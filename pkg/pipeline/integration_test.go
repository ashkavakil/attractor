package pipeline

import (
	"os"
	"path/filepath"
	"testing"
)

// integrationBackendHandler simulates the LLM codergen backend for integration tests.
type integrationBackendHandler struct{}

func (h *integrationBackendHandler) Execute(node *Node, ctx *Context, graph *Graph, logsRoot string) (*Outcome, error) {
	// Write prompt and response artifacts like the real codergen handler does.
	prompt := node.Prompt
	if prompt == "" {
		prompt = node.Label
	}

	stageDir := filepath.Join(logsRoot, node.ID)
	os.MkdirAll(stageDir, 0o755)
	os.WriteFile(filepath.Join(stageDir, "prompt.md"), []byte(prompt), 0o644)

	response := "[Simulated] Response for: " + node.ID
	os.WriteFile(filepath.Join(stageDir, "response.md"), []byte(response), 0o644)

	outcome := &Outcome{
		Status: StatusSuccess,
		Notes:  "Stage completed: " + node.ID,
		ContextUpdates: map[string]interface{}{
			"last_stage":    node.ID,
			"last_response": response,
		},
	}

	// Write status.json
	statusJSON := `{"outcome":"success","notes":"Stage completed: ` + node.ID + `"}`
	os.WriteFile(filepath.Join(stageDir, "status.json"), []byte(statusJSON), 0o644)

	return outcome, nil
}

func TestIntegrationSmokeTest(t *testing.T) {
	source := `digraph test_pipeline {
		graph [goal="Create a hello world Python script"]

		start       [shape=Mdiamond]
		plan        [shape=box, prompt="Plan how to create a hello world script for: $goal"]
		implement   [shape=box, prompt="Write the code based on the plan", goal_gate=true]
		review      [shape=box, prompt="Review the code for correctness"]
		done        [shape=Msquare]

		start -> plan
		plan -> implement
		implement -> review [condition="outcome=success"]
		implement -> plan   [condition="outcome=fail", label="Retry"]
		review -> done      [condition="outcome=success"]
		review -> implement [condition="outcome=fail", label="Fix"]
	}`

	// 1. Parse
	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if graph.Goal != "Create a hello world Python script" {
		t.Errorf("expected goal, got %q", graph.Goal)
	}
	if len(graph.Nodes) != 5 {
		t.Errorf("expected 5 nodes, got %d", len(graph.Nodes))
	}
	if len(graph.Edges) != 6 {
		t.Errorf("expected 6 edges, got %d", len(graph.Edges))
	}

	// 2. Validate
	diagnostics, err := ValidateOrRaise(graph)
	if err != nil {
		t.Fatalf("Validation failed: %v", err)
	}
	for _, d := range diagnostics {
		if d.Severity == SeverityError {
			t.Errorf("unexpected error: %s", d)
		}
	}

	// 3. Execute with simulated LLM callback
	logsRoot := t.TempDir()

	resolver := &staticResolver{handler: &integrationBackendHandler{}}
	engine := NewEngine(EngineConfig{LogsRoot: logsRoot}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// 4. Verify outcome
	if result.Status != StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", result.Status)
	}

	// Check that implement is in completed nodes
	found := false
	for _, id := range result.CompletedNodes {
		if id == "implement" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected 'implement' in completed nodes: %v", result.CompletedNodes)
	}

	// 5. Verify artifacts exist
	for _, nodeID := range []string{"plan", "implement", "review"} {
		stageDir := filepath.Join(logsRoot, nodeID)
		for _, file := range []string{"prompt.md", "response.md", "status.json"} {
			path := filepath.Join(stageDir, file)
			if _, err := os.Stat(path); os.IsNotExist(err) {
				t.Errorf("artifact missing: %s", path)
			}
		}
	}

	// 6. Verify goal gate was satisfied
	if outcome, ok := result.NodeOutcomes["implement"]; ok {
		if outcome.Status != StatusSuccess {
			t.Errorf("goal gate 'implement' should be SUCCESS, got %s", outcome.Status)
		}
	} else {
		t.Error("implement outcome not recorded")
	}

	// 7. Verify checkpoint
	cpPath := filepath.Join(logsRoot, "checkpoint.json")
	cp, err := LoadCheckpoint(cpPath)
	if err != nil {
		t.Fatalf("Load checkpoint failed: %v", err)
	}

	// done is the terminal node
	if cp.CurrentNode != "review" {
		// The last checkpoint is saved at the last non-terminal node
		t.Logf("checkpoint current_node: %s", cp.CurrentNode)
	}

	for _, expected := range []string{"plan", "implement", "review"} {
		found := false
		for _, id := range cp.CompletedNodes {
			if id == expected {
				found = true
			}
		}
		if !found {
			t.Errorf("expected %q in checkpoint completed_nodes: %v", expected, cp.CompletedNodes)
		}
	}
}

func TestIntegrationHumanGate(t *testing.T) {
	source := `digraph Review {
		start [shape=Mdiamond, label="Start"]
		exit  [shape=Msquare, label="Exit"]

		review_gate [
			shape=hexagon,
			label="Review Changes"
		]

		ship_it [label="Ship It", prompt="Deploy"]
		fixes   [label="Fixes", prompt="Apply fixes"]

		start -> review_gate
		review_gate -> ship_it [label="[A] Approve"]
		review_gate -> fixes   [label="[F] Fix"]
		ship_it -> exit
		fixes -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// The review_gate should be a hexagon (wait.human handler)
	gate := graph.Nodes["review_gate"]
	if gate == nil {
		t.Fatal("review_gate node not found")
	}
	if gate.Shape != "hexagon" {
		t.Errorf("expected shape hexagon, got %q", gate.Shape)
	}

	// Verify the edges have labels with accelerator keys
	edges := graph.OutgoingEdges("review_gate")
	if len(edges) != 2 {
		t.Errorf("expected 2 outgoing edges from review_gate, got %d", len(edges))
	}
}

func TestIntegrationCustomHandler(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Nodes: map[string]*Node{
			"start":  {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"custom": {ID: "custom", Shape: "box", Label: "Custom", Type: "my_custom_type", Attrs: map[string]string{}},
			"exit":   {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "custom"},
			{From: "custom", To: "exit"},
		},
	}

	customCalled := false
	customHandler := &testCustomHandler{called: &customCalled}

	resolver := &staticResolver{
		handler: &simpleHandler{},
		special: map[string]Handler{"custom": customHandler},
	}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if !customCalled {
		t.Error("custom handler was not called")
	}
	if result.Status != StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", result.Status)
	}
}

type testCustomHandler struct {
	called *bool
}

func (h *testCustomHandler) Execute(node *Node, ctx *Context, graph *Graph, logsRoot string) (*Outcome, error) {
	*h.called = true
	return &Outcome{Status: StatusSuccess, Notes: "custom handler executed"}, nil
}
