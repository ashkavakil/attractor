package handler

import (
	"testing"

	"github.com/ashka-vakil/attractor/pkg/pipeline"
)

func TestStartHandler(t *testing.T) {
	h := &StartHandler{}
	node := &pipeline.Node{ID: "start", Shape: "Mdiamond", Attrs: map[string]string{}}
	outcome, err := h.Execute(node, pipeline.NewContext(), &pipeline.Graph{}, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", outcome.Status)
	}
}

func TestExitHandler(t *testing.T) {
	h := &ExitHandler{}
	node := &pipeline.Node{ID: "exit", Shape: "Msquare", Attrs: map[string]string{}}
	outcome, err := h.Execute(node, pipeline.NewContext(), &pipeline.Graph{}, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", outcome.Status)
	}
}

func TestConditionalHandler(t *testing.T) {
	h := &ConditionalHandler{}
	node := &pipeline.Node{ID: "gate", Shape: "diamond", Attrs: map[string]string{}}
	outcome, err := h.Execute(node, pipeline.NewContext(), &pipeline.Graph{}, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", outcome.Status)
	}
}

func TestCodergenHandlerSimulation(t *testing.T) {
	h := &CodergenHandler{Backend: nil} // simulation mode
	node := &pipeline.Node{
		ID:     "plan",
		Label:  "Plan",
		Prompt: "Plan the implementation of $goal",
		Attrs:  map[string]string{},
	}
	graph := &pipeline.Graph{Goal: "a test feature"}
	logsRoot := t.TempDir()

	outcome, err := h.Execute(node, pipeline.NewContext(), graph, logsRoot)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", outcome.Status)
	}
	if outcome.ContextUpdates == nil {
		t.Error("expected context_updates")
	}
	if outcome.ContextUpdates["last_stage"] != "plan" {
		t.Errorf("expected last_stage=plan, got %v", outcome.ContextUpdates["last_stage"])
	}
}

func TestCodergenHandlerWithBackend(t *testing.T) {
	backend := &mockBackend{response: "code generated"}
	h := &CodergenHandler{Backend: backend}
	node := &pipeline.Node{
		ID:     "impl",
		Label:  "Implement",
		Prompt: "Write the code",
		Attrs:  map[string]string{},
	}
	graph := &pipeline.Graph{Goal: "test"}
	logsRoot := t.TempDir()

	outcome, err := h.Execute(node, pipeline.NewContext(), graph, logsRoot)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", outcome.Status)
	}
}

type mockBackend struct {
	response string
}

func (m *mockBackend) Run(node *pipeline.Node, prompt string, ctx *pipeline.Context) (interface{}, error) {
	return m.response, nil
}

func TestCodergenHandlerGoalExpansion(t *testing.T) {
	h := &CodergenHandler{Backend: nil}
	node := &pipeline.Node{
		ID:     "plan",
		Prompt: "Plan for: $goal",
		Attrs:  map[string]string{},
	}
	graph := &pipeline.Graph{Goal: "build a REST API"}
	logsRoot := t.TempDir()

	outcome, err := h.Execute(node, pipeline.NewContext(), graph, logsRoot)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", outcome.Status)
	}
}

func TestWaitForHumanHandlerNoEdges(t *testing.T) {
	h := &WaitForHumanHandler{Interviewer: &AutoApproveInterviewer{}}
	node := &pipeline.Node{ID: "gate", Label: "Choose", Attrs: map[string]string{}}
	graph := &pipeline.Graph{
		Nodes: map[string]*pipeline.Node{"gate": node},
		Edges: []*pipeline.Edge{}, // no outgoing edges
	}

	outcome, err := h.Execute(node, pipeline.NewContext(), graph, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusFail {
		t.Errorf("expected FAIL when no outgoing edges, got %s", outcome.Status)
	}
}

func TestWaitForHumanHandlerAutoApprove(t *testing.T) {
	h := &WaitForHumanHandler{Interviewer: &AutoApproveInterviewer{}}
	node := &pipeline.Node{ID: "gate", Label: "Review Changes", Attrs: map[string]string{}}
	graph := &pipeline.Graph{
		Nodes: map[string]*pipeline.Node{
			"gate":    node,
			"approve": {ID: "approve", Attrs: map[string]string{}},
			"fix":     {ID: "fix", Attrs: map[string]string{}},
		},
		Edges: []*pipeline.Edge{
			{From: "gate", To: "approve", Label: "[A] Approve"},
			{From: "gate", To: "fix", Label: "[F] Fix"},
		},
	}

	outcome, err := h.Execute(node, pipeline.NewContext(), graph, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", outcome.Status)
	}
	if len(outcome.SuggestedNextIDs) == 0 {
		t.Error("expected suggested next IDs")
	}
	// Auto-approve selects first option
	if outcome.SuggestedNextIDs[0] != "approve" {
		t.Errorf("expected first option 'approve', got %q", outcome.SuggestedNextIDs[0])
	}
}

func TestQueueInterviewer(t *testing.T) {
	q := &QueueInterviewer{
		Answers: []*Answer{
			{Value: "F"},
			{Value: "A"},
		},
	}

	a1 := q.Ask(&Question{Text: "Q1"})
	if a1.Value != "F" {
		t.Errorf("expected F, got %v", a1.Value)
	}

	a2 := q.Ask(&Question{Text: "Q2"})
	if a2.Value != "A" {
		t.Errorf("expected A, got %v", a2.Value)
	}

	// Queue exhausted
	a3 := q.Ask(&Question{Text: "Q3"})
	if a3.Value != AnswerSkipped {
		t.Errorf("expected SKIPPED, got %v", a3.Value)
	}
}

func TestRecordingInterviewer(t *testing.T) {
	inner := &AutoApproveInterviewer{}
	recorder := &RecordingInterviewer{Inner: inner}

	q := &Question{Text: "Approve?", Type: QuestionYesNo}
	recorder.Ask(q)

	if len(recorder.Recordings) != 1 {
		t.Errorf("expected 1 recording, got %d", len(recorder.Recordings))
	}
}

func TestCallbackInterviewer(t *testing.T) {
	called := false
	cb := &CallbackInterviewer{
		Callback: func(q *Question) *Answer {
			called = true
			return &Answer{Value: "custom"}
		},
	}

	answer := cb.Ask(&Question{Text: "Test?"})
	if !called {
		t.Error("callback was not called")
	}
	if answer.Value != "custom" {
		t.Errorf("expected 'custom', got %v", answer.Value)
	}
}

func TestHandlerRegistry(t *testing.T) {
	registry := NewRegistry(nil, &AutoApproveInterviewer{})

	// Test shape-based resolution
	boxNode := &pipeline.Node{ID: "test", Shape: "box", Attrs: map[string]string{}}
	handler := registry.Resolve(boxNode)
	if _, ok := handler.(*CodergenHandler); !ok {
		t.Errorf("expected CodergenHandler for box shape")
	}

	startNode := &pipeline.Node{ID: "start", Shape: "Mdiamond", Attrs: map[string]string{}}
	handler = registry.Resolve(startNode)
	if _, ok := handler.(*StartHandler); !ok {
		t.Errorf("expected StartHandler for Mdiamond shape")
	}

	// Test explicit type override
	typedNode := &pipeline.Node{ID: "gate", Shape: "box", Type: "wait.human", Attrs: map[string]string{}}
	handler = registry.Resolve(typedNode)
	if _, ok := handler.(*WaitForHumanHandler); !ok {
		t.Errorf("expected WaitForHumanHandler for type=wait.human")
	}
}

func TestCustomHandlerRegistration(t *testing.T) {
	registry := NewRegistry(nil, &AutoApproveInterviewer{})

	called := false
	custom := &testCustomHandler{called: &called}
	registry.Register("my_type", custom)

	node := &pipeline.Node{ID: "test", Shape: "box", Type: "my_type", Attrs: map[string]string{}}
	handler := registry.Resolve(node)

	outcome, _ := handler.Execute(node, pipeline.NewContext(), &pipeline.Graph{}, "")
	if !called {
		t.Error("custom handler was not called")
	}
	if outcome.Status != pipeline.StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", outcome.Status)
	}
}

type testCustomHandler struct {
	called *bool
}

func (h *testCustomHandler) Execute(node *pipeline.Node, ctx *pipeline.Context, graph *pipeline.Graph, logsRoot string) (*pipeline.Outcome, error) {
	*h.called = true
	return &pipeline.Outcome{Status: pipeline.StatusSuccess}, nil
}

func TestToolHandlerNoCommand(t *testing.T) {
	h := &ToolHandler{}
	node := &pipeline.Node{ID: "tool", Shape: "parallelogram", Attrs: map[string]string{}}
	outcome, err := h.Execute(node, pipeline.NewContext(), &pipeline.Graph{}, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusFail {
		t.Errorf("expected FAIL when no tool_command, got %s", outcome.Status)
	}
}

func TestToolHandlerWithCommand(t *testing.T) {
	h := &ToolHandler{}
	node := &pipeline.Node{
		ID:    "tool",
		Shape: "parallelogram",
		Attrs: map[string]string{
			"tool_command": "echo hello",
		},
	}
	outcome, err := h.Execute(node, pipeline.NewContext(), &pipeline.Graph{}, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != pipeline.StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", outcome.Status)
	}
}

func TestAcceleratorKeyParsing(t *testing.T) {
	tests := []struct {
		label    string
		expected string
	}{
		{"[Y] Yes, deploy", "Y"},
		{"[A] Approve", "A"},
		{"N) No", "N"},
		{"F - Fix", "F"},
		{"Yes", "Y"},
	}

	for _, tt := range tests {
		key := parseAcceleratorKey(tt.label)
		if key != tt.expected {
			t.Errorf("parseAcceleratorKey(%q) = %q, want %q", tt.label, key, tt.expected)
		}
	}
}
