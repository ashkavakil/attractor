// Command attractor is the CLI for the Attractor pipeline engine,
// coding agent loop, and unified LLM client.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/ashka-vakil/attractor/pkg/agent"
	"github.com/ashka-vakil/attractor/pkg/llm"
	_ "github.com/ashka-vakil/attractor/pkg/llm/provider/anthropic"
	_ "github.com/ashka-vakil/attractor/pkg/llm/provider/gemini"
	_ "github.com/ashka-vakil/attractor/pkg/llm/provider/openai"
	"github.com/ashka-vakil/attractor/pkg/pipeline"
	"github.com/ashka-vakil/attractor/pkg/pipeline/handler"
	"github.com/ashka-vakil/attractor/pkg/pipeline/transform"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "run":
		cmdRun(os.Args[2:])
	case "agent":
		cmdAgent(os.Args[2:])
	case "serve":
		cmdServe(os.Args[2:])
	case "validate":
		cmdValidate(os.Args[2:])
	case "version":
		fmt.Println("attractor v0.1.0")
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `Usage: attractor <command> [options]

Commands:
  run       Execute a DOT pipeline file
  agent     Start an interactive coding agent session
  serve     Start the HTTP pipeline server
  validate  Validate a DOT pipeline file
  version   Print version
  help      Show this help

Use "attractor <command> -h" for command-specific help.
`)
}

// cmdRun executes a DOT pipeline from a file.
func cmdRun(args []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	logsDir := fs.String("logs", "", "Directory for pipeline logs (default: temp dir)")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Usage: attractor run [options] <pipeline.dot>")
		os.Exit(1)
	}

	client := llm.FromEnv()
	defer client.Close()

	registry := handler.NewRegistry(nil, &handler.AutoApproveInterviewer{})
	resolver := &registryAdapter{registry: registry}

	opts := []pipeline.RunnerOption{}
	if *logsDir != "" {
		opts = append(opts, pipeline.WithLogsRoot(*logsDir))
	}

	runner := pipeline.NewRunner(resolver, opts...)
	runner.RegisterTransform(transform.VariableExpansion())
	runner.RegisterTransform(transform.StylesheetApplication())

	result, err := runner.RunFromFile(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Pipeline completed: status=%s, stages=%d\n", result.Status, len(result.CompletedNodes))
	if result.Status == pipeline.StatusFail {
		os.Exit(1)
	}
}

// cmdAgent starts an interactive coding agent session.
func cmdAgent(args []string) {
	fs := flag.NewFlagSet("agent", flag.ExitOnError)
	model := fs.String("model", "", "Model to use (e.g., claude-opus-4-6, gpt-4.1)")
	provider := fs.String("provider", "", "Provider (anthropic, openai, gemini)")
	maxTurns := fs.Int("max-turns", 0, "Maximum number of turns (0 = unlimited)")
	fs.Parse(args)

	client := llm.FromEnv()
	defer client.Close()
	requireProvider(client)

	// Resolve provider and model
	prov := *provider
	mod := *model
	if prov == "" {
		prov = detectProvider()
	}
	if mod == "" {
		mod = defaultModel(prov)
	}

	// Select profile
	var profile *agent.ProviderProfile
	switch prov {
	case "openai":
		profile = agent.DefaultOpenAIProfile(mod)
	case "gemini":
		profile = agent.DefaultGeminiProfile(mod)
	default:
		profile = agent.DefaultAnthropicProfile(mod)
	}

	config := agent.DefaultSessionConfig()
	if *maxTurns > 0 {
		config.MaxTurns = *maxTurns
	}

	session := agent.NewSession(client, profile, nil, config)
	defer session.Close()

	// Print events
	session.EventEmitter.On(func(e agent.Event) {
		switch e.Type {
		case agent.EventToolCallStarted:
			if name, ok := e.Data["tool_name"].(string); ok {
				fmt.Fprintf(os.Stderr, "  [tool] %s\n", name)
			}
		case agent.EventError:
			if msg, ok := e.Data["error"].(string); ok {
				fmt.Fprintf(os.Stderr, "  [error] %s\n", msg)
			}
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle ctrl+c
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		cancel()
	}()

	// Read prompt from args or stdin
	prompt := ""
	if fs.NArg() > 0 {
		prompt = fs.Arg(0)
	} else {
		fmt.Fprint(os.Stderr, "Enter your prompt: ")
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading stdin: %v\n", err)
			os.Exit(1)
		}
		prompt = string(data)
	}

	if prompt == "" {
		fmt.Fprintln(os.Stderr, "Error: no prompt provided")
		os.Exit(1)
	}

	if err := session.Submit(ctx, prompt); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Print final response
	if len(session.History) > 0 {
		for i := len(session.History) - 1; i >= 0; i-- {
			if at, ok := session.History[i].(*agent.AssistantTurn); ok {
				if at.Content != "" {
					fmt.Println(at.Content)
				}
				break
			}
		}
	}
}

// cmdServe starts the HTTP pipeline server.
func cmdServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", ":8080", "Listen address")
	fs.Parse(args)

	registry := handler.NewRegistry(nil, &handler.AutoApproveInterviewer{})
	resolver := &registryAdapter{registry: registry}
	server := pipeline.NewServer(resolver)

	fmt.Fprintf(os.Stderr, "Listening on %s\n", *addr)
	if err := http.ListenAndServe(*addr, server.Handler()); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// cmdValidate validates a DOT pipeline file.
func cmdValidate(args []string) {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "Usage: attractor validate <pipeline.dot>")
		os.Exit(1)
	}

	data, err := os.ReadFile(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		os.Exit(1)
	}

	graph, err := pipeline.Parse(string(data))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Parse error: %v\n", err)
		os.Exit(1)
	}

	diagnostics := pipeline.Validate(graph)
	hasErrors := false
	for _, d := range diagnostics {
		fmt.Println(d.String())
		if d.Severity == pipeline.SeverityError {
			hasErrors = true
		}
	}

	if hasErrors {
		os.Exit(1)
	}
	if len(diagnostics) == 0 {
		fmt.Println("Valid.")
	}
}

func requireProvider(client *llm.Client) {
	if !client.HasProviders() {
		fmt.Fprintln(os.Stderr, "Error: no LLM provider configured.")
		fmt.Fprintln(os.Stderr, "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY")
		os.Exit(1)
	}
}

// registryAdapter wraps handler.Registry to satisfy pipeline.HandlerResolver,
// bridging the handler.Handler and pipeline.Handler interfaces.
type registryAdapter struct {
	registry *handler.Registry
}

func (r *registryAdapter) Resolve(node *pipeline.Node) pipeline.Handler {
	h := r.registry.Resolve(node)
	if h == nil {
		return nil
	}
	return h
}

func detectProvider() string {
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		return "anthropic"
	}
	if os.Getenv("OPENAI_API_KEY") != "" {
		return "openai"
	}
	if os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "" {
		return "gemini"
	}
	return "anthropic"
}

func defaultModel(provider string) string {
	switch provider {
	case "openai":
		return "gpt-4.1"
	case "gemini":
		return "gemini-2.5-pro"
	default:
		return "claude-sonnet-4-5-20250929"
	}
}
