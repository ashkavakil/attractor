// Package handler implements node handlers for the Attractor pipeline engine.
package handler

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ashka-vakil/attractor/pkg/pipeline"
)

// Handler is the interface for node execution.
type Handler interface {
	Execute(node *pipeline.Node, ctx *pipeline.Context, graph *pipeline.Graph, logsRoot string) (*pipeline.Outcome, error)
}

// CodergenBackend is the interface for LLM execution in the codergen handler.
type CodergenBackend interface {
	Run(node *pipeline.Node, prompt string, ctx *pipeline.Context) (interface{}, error)
}

// Registry maps type strings to handler instances.
type Registry struct {
	mu             sync.RWMutex
	handlers       map[string]Handler
	defaultHandler Handler
}

// ShapeToType maps DOT shapes to handler type strings.
var ShapeToType = map[string]string{
	"Mdiamond":       "start",
	"Msquare":        "exit",
	"box":            "codergen",
	"hexagon":        "wait.human",
	"diamond":        "conditional",
	"component":      "parallel",
	"tripleoctagon":  "parallel.fan_in",
	"parallelogram":  "tool",
	"house":          "stack.manager_loop",
}

// NewRegistry creates a new handler registry with all built-in handlers.
func NewRegistry(backend CodergenBackend, interviewer Interviewer) *Registry {
	r := &Registry{
		handlers: make(map[string]Handler),
	}

	codergen := &CodergenHandler{Backend: backend}
	r.defaultHandler = codergen

	r.Register("start", &StartHandler{})
	r.Register("exit", &ExitHandler{})
	r.Register("codergen", codergen)
	r.Register("wait.human", &WaitForHumanHandler{Interviewer: interviewer})
	r.Register("conditional", &ConditionalHandler{})
	r.Register("parallel", &ParallelHandler{})
	r.Register("parallel.fan_in", &FanInHandler{})
	r.Register("tool", &ToolHandler{})
	r.Register("stack.manager_loop", &ManagerLoopHandler{})

	return r
}

// Register adds a handler for the given type string.
func (r *Registry) Register(typeStr string, handler Handler) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.handlers[typeStr] = handler
}

// Resolve finds the handler for a node.
func (r *Registry) Resolve(node *pipeline.Node) Handler {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// 1. Explicit type attribute
	if node.Type != "" {
		if h, ok := r.handlers[node.Type]; ok {
			return h
		}
	}

	// 2. Shape-based resolution
	if handlerType, ok := ShapeToType[node.Shape]; ok {
		if h, ok := r.handlers[handlerType]; ok {
			return h
		}
	}

	// 3. Default
	return r.defaultHandler
}

// --- Start Handler ---

// StartHandler is a no-op handler for the pipeline entry point.
type StartHandler struct{}

func (h *StartHandler) Execute(_ *pipeline.Node, _ *pipeline.Context, _ *pipeline.Graph, _ string) (*pipeline.Outcome, error) {
	return &pipeline.Outcome{Status: pipeline.StatusSuccess}, nil
}

// --- Exit Handler ---

// ExitHandler is a no-op handler for the pipeline exit point.
type ExitHandler struct{}

func (h *ExitHandler) Execute(_ *pipeline.Node, _ *pipeline.Context, _ *pipeline.Graph, _ string) (*pipeline.Outcome, error) {
	return &pipeline.Outcome{Status: pipeline.StatusSuccess}, nil
}

// --- Codergen Handler ---

// CodergenHandler executes LLM tasks.
type CodergenHandler struct {
	Backend CodergenBackend
}

func (h *CodergenHandler) Execute(node *pipeline.Node, ctx *pipeline.Context, graph *pipeline.Graph, logsRoot string) (*pipeline.Outcome, error) {
	// 1. Build prompt
	prompt := node.Prompt
	if prompt == "" {
		prompt = node.Label
	}
	prompt = expandVariables(prompt, graph, ctx)

	// 2. Write prompt to logs
	stageDir := filepath.Join(logsRoot, node.ID)
	os.MkdirAll(stageDir, 0o755)
	os.WriteFile(filepath.Join(stageDir, "prompt.md"), []byte(prompt), 0o644)

	// 3. Call LLM backend
	var responseText string
	if h.Backend != nil {
		result, err := h.Backend.Run(node, prompt, ctx)
		if err != nil {
			return &pipeline.Outcome{
				Status:        pipeline.StatusFail,
				FailureReason: err.Error(),
			}, nil
		}
		// If result is an Outcome, return it directly.
		if outcome, ok := result.(*pipeline.Outcome); ok {
			writeStatus(stageDir, outcome)
			return outcome, nil
		}
		responseText = fmt.Sprint(result)
	} else {
		responseText = "[Simulated] Response for stage: " + node.ID
	}

	// 4. Write response
	os.WriteFile(filepath.Join(stageDir, "response.md"), []byte(responseText), 0o644)

	// 5. Return outcome
	outcome := &pipeline.Outcome{
		Status: pipeline.StatusSuccess,
		Notes:  "Stage completed: " + node.ID,
		ContextUpdates: map[string]interface{}{
			"last_stage":    node.ID,
			"last_response": truncate(responseText, 200),
		},
	}
	writeStatus(stageDir, outcome)
	return outcome, nil
}

// --- Conditional Handler ---

// ConditionalHandler is a pass-through; the engine evaluates edge conditions.
type ConditionalHandler struct{}

func (h *ConditionalHandler) Execute(node *pipeline.Node, _ *pipeline.Context, _ *pipeline.Graph, _ string) (*pipeline.Outcome, error) {
	return &pipeline.Outcome{
		Status: pipeline.StatusSuccess,
		Notes:  "Conditional node evaluated: " + node.ID,
	}, nil
}

// --- Interviewer Interface ---

// QuestionType identifies the type of human interaction question.
type QuestionType int

const (
	QuestionYesNo QuestionType = iota
	QuestionMultipleChoice
	QuestionFreeform
	QuestionConfirmation
)

// Question is a question to present to a human.
type Question struct {
	Text           string
	Type           QuestionType
	Options        []QuestionOption
	Default        *Answer
	TimeoutSeconds float64
	Stage          string
	Metadata       map[string]interface{}
}

// QuestionOption is a choice in a multiple-choice question.
type QuestionOption struct {
	Key   string
	Label string
}

// AnswerValue represents special answer types.
type AnswerValue int

const (
	AnswerYes AnswerValue = iota
	AnswerNo
	AnswerSkipped
	AnswerTimeout
)

// Answer is a human's response to a question.
type Answer struct {
	Value          interface{}
	SelectedOption *QuestionOption
	Text           string
}

// Interviewer is the interface for human interaction.
type Interviewer interface {
	Ask(question *Question) *Answer
	Inform(message, stage string)
}

// --- Wait For Human Handler ---

// WaitForHumanHandler blocks until a human selects an option.
type WaitForHumanHandler struct {
	Interviewer Interviewer
}

func (h *WaitForHumanHandler) Execute(node *pipeline.Node, ctx *pipeline.Context, graph *pipeline.Graph, logsRoot string) (*pipeline.Outcome, error) {
	edges := graph.OutgoingEdges(node.ID)
	if len(edges) == 0 {
		return &pipeline.Outcome{
			Status:        pipeline.StatusFail,
			FailureReason: "No outgoing edges for human gate",
		}, nil
	}

	// Build choices from edges
	var options []QuestionOption
	type choice struct {
		key   string
		label string
		to    string
	}
	var choices []choice

	for _, edge := range edges {
		label := edge.Label
		if label == "" {
			label = edge.To
		}
		key := parseAcceleratorKey(label)
		options = append(options, QuestionOption{Key: key, Label: label})
		choices = append(choices, choice{key: key, label: label, to: edge.To})
	}

	text := node.Label
	if text == "" {
		text = "Select an option:"
	}

	question := &Question{
		Text:    text,
		Type:    QuestionMultipleChoice,
		Options: options,
		Stage:   node.ID,
	}

	answer := h.Interviewer.Ask(question)

	// Handle special answers
	if answer == nil || answer.Value == AnswerTimeout {
		defaultChoice := node.Attrs["human.default_choice"]
		if defaultChoice != "" {
			for _, c := range choices {
				if c.to == defaultChoice || c.key == defaultChoice {
					return &pipeline.Outcome{
						Status:           pipeline.StatusSuccess,
						SuggestedNextIDs: []string{c.to},
						ContextUpdates: map[string]interface{}{
							"human.gate.selected": c.key,
							"human.gate.label":    c.label,
						},
					}, nil
				}
			}
		}
		return &pipeline.Outcome{
			Status:        pipeline.StatusRetry,
			FailureReason: "human gate timeout, no default",
		}, nil
	}

	if answer.Value == AnswerSkipped {
		return &pipeline.Outcome{
			Status:        pipeline.StatusFail,
			FailureReason: "human skipped interaction",
		}, nil
	}

	// Find matching choice
	selectedTo := choices[0].to
	selectedKey := choices[0].key
	selectedLabel := choices[0].label

	answerStr := fmt.Sprint(answer.Value)
	for _, c := range choices {
		if strings.EqualFold(c.key, answerStr) || strings.EqualFold(c.label, answerStr) {
			selectedTo = c.to
			selectedKey = c.key
			selectedLabel = c.label
			break
		}
	}

	return &pipeline.Outcome{
		Status:           pipeline.StatusSuccess,
		SuggestedNextIDs: []string{selectedTo},
		ContextUpdates: map[string]interface{}{
			"human.gate.selected": selectedKey,
			"human.gate.label":    selectedLabel,
		},
	}, nil
}

// --- Parallel Handler ---

// ParallelHandler fans out execution to multiple branches.
type ParallelHandler struct {
	Registry *Registry // set by engine after creation
}

func (h *ParallelHandler) Execute(node *pipeline.Node, ctx *pipeline.Context, graph *pipeline.Graph, logsRoot string) (*pipeline.Outcome, error) {
	edges := graph.OutgoingEdges(node.ID)
	if len(edges) == 0 {
		return &pipeline.Outcome{
			Status:        pipeline.StatusFail,
			FailureReason: "No branches for parallel execution",
		}, nil
	}

	maxParallel := 4
	if v, ok := node.Attrs["max_parallel"]; ok {
		n, _ := strconv.Atoi(v)
		if n > 0 {
			maxParallel = n
		}
	}

	type branchResult struct {
		nodeID  string
		outcome *pipeline.Outcome
	}

	results := make([]branchResult, len(edges))
	sem := make(chan struct{}, maxParallel)
	var wg sync.WaitGroup

	for i, edge := range edges {
		wg.Add(1)
		go func(idx int, e *pipeline.Edge) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			branchCtx := ctx.Clone()
			targetNode := graph.Nodes[e.To]
			if targetNode == nil {
				results[idx] = branchResult{
					nodeID:  e.To,
					outcome: &pipeline.Outcome{Status: pipeline.StatusFail, FailureReason: "node not found"},
				}
				return
			}

			if h.Registry != nil {
				handler := h.Registry.Resolve(targetNode)
				outcome, err := handler.Execute(targetNode, branchCtx, graph, logsRoot)
				if err != nil {
					results[idx] = branchResult{
						nodeID:  e.To,
						outcome: &pipeline.Outcome{Status: pipeline.StatusFail, FailureReason: err.Error()},
					}
					return
				}
				results[idx] = branchResult{nodeID: e.To, outcome: outcome}
			} else {
				results[idx] = branchResult{
					nodeID:  e.To,
					outcome: &pipeline.Outcome{Status: pipeline.StatusSuccess, Notes: "Branch: " + e.To},
				}
			}
		}(i, edge)
	}

	wg.Wait()

	// Evaluate join policy
	successCount := 0
	failCount := 0
	for _, r := range results {
		if r.outcome.Status == pipeline.StatusSuccess || r.outcome.Status == pipeline.StatusPartialSuccess {
			successCount++
		} else if r.outcome.Status == pipeline.StatusFail {
			failCount++
		}
	}

	// Serialize results for fan-in
	serialized, _ := json.Marshal(results)
	ctx.Set("parallel.results", string(serialized))

	joinPolicy := node.Attrs["join_policy"]
	if joinPolicy == "" {
		joinPolicy = "wait_all"
	}

	switch joinPolicy {
	case "wait_all":
		if failCount == 0 {
			return &pipeline.Outcome{Status: pipeline.StatusSuccess}, nil
		}
		return &pipeline.Outcome{Status: pipeline.StatusPartialSuccess}, nil
	case "first_success":
		if successCount > 0 {
			return &pipeline.Outcome{Status: pipeline.StatusSuccess}, nil
		}
		return &pipeline.Outcome{Status: pipeline.StatusFail}, nil
	default:
		if failCount == 0 {
			return &pipeline.Outcome{Status: pipeline.StatusSuccess}, nil
		}
		return &pipeline.Outcome{Status: pipeline.StatusPartialSuccess}, nil
	}
}

// --- Fan-In Handler ---

// FanInHandler consolidates parallel results.
type FanInHandler struct{}

func (h *FanInHandler) Execute(node *pipeline.Node, ctx *pipeline.Context, graph *pipeline.Graph, logsRoot string) (*pipeline.Outcome, error) {
	resultsJSON := ctx.GetString("parallel.results")
	if resultsJSON == "" {
		return &pipeline.Outcome{
			Status:        pipeline.StatusFail,
			FailureReason: "No parallel results to evaluate",
		}, nil
	}

	return &pipeline.Outcome{
		Status: pipeline.StatusSuccess,
		Notes:  "Fan-in completed",
		ContextUpdates: map[string]interface{}{
			"parallel.fan_in.complete": "true",
		},
	}, nil
}

// --- Tool Handler ---

// ToolHandler executes external commands.
type ToolHandler struct{}

func (h *ToolHandler) Execute(node *pipeline.Node, ctx *pipeline.Context, graph *pipeline.Graph, logsRoot string) (*pipeline.Outcome, error) {
	command := node.Attrs["tool_command"]
	if command == "" {
		return &pipeline.Outcome{
			Status:        pipeline.StatusFail,
			FailureReason: "No tool_command specified",
		}, nil
	}

	timeout := node.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	cmd := exec.Command("sh", "-c", command)
	cmd.Env = os.Environ()

	output, err := cmd.Output()
	if err != nil {
		return &pipeline.Outcome{
			Status:        pipeline.StatusFail,
			FailureReason: fmt.Sprintf("tool execution failed: %v", err),
		}, nil
	}

	return &pipeline.Outcome{
		Status: pipeline.StatusSuccess,
		ContextUpdates: map[string]interface{}{
			"tool.output": string(output),
		},
		Notes: "Tool completed: " + command,
	}, nil
}

// --- Manager Loop Handler ---

// ManagerLoopHandler orchestrates sprint-based iteration over a child pipeline.
type ManagerLoopHandler struct{}

func (h *ManagerLoopHandler) Execute(node *pipeline.Node, ctx *pipeline.Context, graph *pipeline.Graph, logsRoot string) (*pipeline.Outcome, error) {
	maxCycles := 1000
	if v, ok := node.Attrs["manager.max_cycles"]; ok {
		n, _ := strconv.Atoi(v)
		if n > 0 {
			maxCycles = n
		}
	}

	pollInterval := 45 * time.Second
	if v, ok := node.Attrs["manager.poll_interval"]; ok {
		if d, err := time.ParseDuration(v); err == nil {
			pollInterval = d
		}
	}

	for cycle := 0; cycle < maxCycles; cycle++ {
		childStatus := ctx.GetString("context.stack.child.status")
		if childStatus == "completed" {
			childOutcome := ctx.GetString("context.stack.child.outcome")
			if childOutcome == "success" {
				return &pipeline.Outcome{Status: pipeline.StatusSuccess, Notes: "Child completed"}, nil
			}
		}
		if childStatus == "failed" {
			return &pipeline.Outcome{Status: pipeline.StatusFail, FailureReason: "Child failed"}, nil
		}

		time.Sleep(pollInterval)
	}

	return &pipeline.Outcome{
		Status:        pipeline.StatusFail,
		FailureReason: "Max cycles exceeded",
	}, nil
}

// --- Built-in Interviewer Implementations ---

// AutoApproveInterviewer always selects YES or the first option.
type AutoApproveInterviewer struct{}

func (a *AutoApproveInterviewer) Ask(question *Question) *Answer {
	switch question.Type {
	case QuestionYesNo, QuestionConfirmation:
		return &Answer{Value: AnswerYes}
	case QuestionMultipleChoice:
		if len(question.Options) > 0 {
			return &Answer{
				Value:          question.Options[0].Key,
				SelectedOption: &question.Options[0],
			}
		}
	}
	return &Answer{Value: "auto-approved", Text: "auto-approved"}
}

func (a *AutoApproveInterviewer) Inform(message, stage string) {}

// ConsoleInterviewer reads from standard input.
type ConsoleInterviewer struct{}

func (c *ConsoleInterviewer) Ask(question *Question) *Answer {
	fmt.Printf("[?] %s\n", question.Text)
	switch question.Type {
	case QuestionMultipleChoice:
		for _, opt := range question.Options {
			fmt.Printf("  [%s] %s\n", opt.Key, opt.Label)
		}
		fmt.Print("Select: ")
		var input string
		fmt.Scanln(&input)
		input = strings.TrimSpace(input)
		for _, opt := range question.Options {
			if strings.EqualFold(opt.Key, input) {
				return &Answer{Value: opt.Key, SelectedOption: &opt}
			}
		}
		if len(question.Options) > 0 {
			return &Answer{Value: question.Options[0].Key, SelectedOption: &question.Options[0]}
		}
	case QuestionYesNo:
		fmt.Print("[Y/N]: ")
		var input string
		fmt.Scanln(&input)
		if strings.EqualFold(strings.TrimSpace(input), "y") {
			return &Answer{Value: AnswerYes}
		}
		return &Answer{Value: AnswerNo}
	case QuestionFreeform:
		fmt.Print("> ")
		var input string
		fmt.Scanln(&input)
		return &Answer{Text: strings.TrimSpace(input)}
	}
	return &Answer{Value: AnswerSkipped}
}

func (c *ConsoleInterviewer) Inform(message, stage string) {
	fmt.Printf("[%s] %s\n", stage, message)
}

// CallbackInterviewer delegates to a callback function.
type CallbackInterviewer struct {
	Callback func(*Question) *Answer
}

func (c *CallbackInterviewer) Ask(question *Question) *Answer {
	return c.Callback(question)
}

func (c *CallbackInterviewer) Inform(message, stage string) {}

// QueueInterviewer reads from a pre-filled answer queue.
type QueueInterviewer struct {
	mu      sync.Mutex
	Answers []*Answer
}

func (q *QueueInterviewer) Ask(question *Question) *Answer {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.Answers) > 0 {
		a := q.Answers[0]
		q.Answers = q.Answers[1:]
		return a
	}
	return &Answer{Value: AnswerSkipped}
}

func (q *QueueInterviewer) Inform(message, stage string) {}

// RecordingInterviewer wraps another interviewer and records interactions.
type RecordingInterviewer struct {
	Inner      Interviewer
	mu         sync.Mutex
	Recordings []struct {
		Question *Question
		Answer   *Answer
	}
}

func (r *RecordingInterviewer) Ask(question *Question) *Answer {
	answer := r.Inner.Ask(question)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Recordings = append(r.Recordings, struct {
		Question *Question
		Answer   *Answer
	}{question, answer})
	return answer
}

func (r *RecordingInterviewer) Inform(message, stage string) {
	r.Inner.Inform(message, stage)
}

// --- Helpers ---

var acceleratorPattern = regexp.MustCompile(`^\[([A-Za-z])\]\s|^([A-Za-z])\)\s|^([A-Za-z])\s-\s`)

func parseAcceleratorKey(label string) string {
	matches := acceleratorPattern.FindStringSubmatch(label)
	if matches != nil {
		for _, m := range matches[1:] {
			if m != "" {
				return strings.ToUpper(m)
			}
		}
	}
	if len(label) > 0 {
		return strings.ToUpper(string(label[0]))
	}
	return ""
}

func expandVariables(prompt string, graph *pipeline.Graph, ctx *pipeline.Context) string {
	prompt = strings.ReplaceAll(prompt, "$goal", graph.Goal)
	return prompt
}

func writeStatus(stageDir string, outcome *pipeline.Outcome) {
	data, _ := json.MarshalIndent(outcome, "", "  ")
	os.WriteFile(filepath.Join(stageDir, "status.json"), data, 0o644)
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}
