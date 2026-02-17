package pipeline

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/ashka-vakil/attractor/pkg/pipeline/events"
)

// Runner is a high-level pipeline execution helper.
type Runner struct {
	resolver    HandlerResolver
	emitter     *events.Emitter
	transforms  []interface{ Apply(*Graph) *Graph }
	logsRoot    string
}

// RunnerOption configures a Runner.
type RunnerOption func(*Runner)

// WithLogsRoot sets the base directory for pipeline logs.
func WithLogsRoot(path string) RunnerOption {
	return func(r *Runner) {
		r.logsRoot = path
	}
}

// WithEmitter sets the event emitter.
func WithEmitter(emitter *events.Emitter) RunnerOption {
	return func(r *Runner) {
		r.emitter = emitter
	}
}

// NewRunner creates a new pipeline runner.
func NewRunner(resolver HandlerResolver, opts ...RunnerOption) *Runner {
	r := &Runner{
		resolver: resolver,
		emitter:  events.NewEmitter(),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// OnEvent registers an event listener.
func (r *Runner) OnEvent(listener func(events.Event)) {
	r.emitter.On(listener)
}

// RegisterTransform adds a custom transform.
func (r *Runner) RegisterTransform(t interface{ Apply(*Graph) *Graph }) {
	r.transforms = append(r.transforms, t)
}

// RunFromSource parses, validates, and executes a DOT pipeline.
func (r *Runner) RunFromSource(source string) (*RunResult, error) {
	// 1. Parse
	graph, err := Parse(source)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	return r.RunGraph(graph)
}

// RunFromFile reads a DOT file and executes it.
func (r *Runner) RunFromFile(path string) (*RunResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}
	return r.RunFromSource(string(data))
}

// RunGraph validates and executes a parsed graph.
func (r *Runner) RunGraph(graph *Graph) (*RunResult, error) {
	// Apply transforms
	for _, t := range r.transforms {
		graph = t.Apply(graph)
	}

	// 2. Validate
	diagnostics, err := ValidateOrRaise(graph)
	if err != nil {
		return nil, err
	}

	// Log warnings
	for _, d := range diagnostics {
		if d.Severity == SeverityWarning {
			r.emitter.Emit(events.NewEvent("validation_warning", map[string]interface{}{
				"rule":    d.Rule,
				"message": d.Message,
			}))
		}
	}

	// 3. Initialize logs
	logsRoot := r.logsRoot
	if logsRoot == "" {
		logsRoot = filepath.Join(os.TempDir(), fmt.Sprintf("attractor-run-%d", time.Now().UnixNano()))
	}
	os.MkdirAll(logsRoot, 0o755)

	// Write manifest
	manifest := fmt.Sprintf(`{"name": %q, "goal": %q, "start_time": %q}`,
		graph.Name, graph.Goal, time.Now().Format(time.RFC3339))
	os.WriteFile(filepath.Join(logsRoot, "manifest.json"), []byte(manifest), 0o644)

	// 4. Execute
	engine := NewEngine(EngineConfig{LogsRoot: logsRoot}, r.resolver, r.emitter)
	return engine.Run(graph)
}
