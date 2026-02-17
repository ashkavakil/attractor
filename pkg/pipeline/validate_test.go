package pipeline

import (
	"testing"
)

func makeSimpleGraph() *Graph {
	return &Graph{
		Name:  "test",
		Nodes: map[string]*Node{
			"start": {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"exit":  {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
			"a":     {ID: "a", Shape: "box", Label: "A", Prompt: "Do A", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
			{From: "a", To: "exit"},
		},
	}
}

func TestValidateMissingStartNode(t *testing.T) {
	graph := &Graph{
		Nodes: map[string]*Node{
			"exit": {ID: "exit", Shape: "Msquare", Attrs: map[string]string{}},
			"a":    {ID: "a", Shape: "box", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "a", To: "exit"},
		},
	}

	diagnostics := Validate(graph)
	found := false
	for _, d := range diagnostics {
		if d.Rule == "start_node" && d.Severity == SeverityError {
			found = true
		}
	}
	if !found {
		t.Error("expected start_node error diagnostic")
	}
}

func TestValidateMissingExitNode(t *testing.T) {
	graph := &Graph{
		Nodes: map[string]*Node{
			"start": {ID: "start", Shape: "Mdiamond", Attrs: map[string]string{}},
			"a":     {ID: "a", Shape: "box", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
		},
	}

	diagnostics := Validate(graph)
	found := false
	for _, d := range diagnostics {
		if d.Rule == "terminal_node" && d.Severity == SeverityError {
			found = true
		}
	}
	if !found {
		t.Error("expected terminal_node error diagnostic")
	}
}

func TestValidateStartNoIncoming(t *testing.T) {
	graph := makeSimpleGraph()
	graph.Edges = append(graph.Edges, &Edge{From: "a", To: "start"})

	diagnostics := Validate(graph)
	found := false
	for _, d := range diagnostics {
		if d.Rule == "start_no_incoming" && d.Severity == SeverityError {
			found = true
		}
	}
	if !found {
		t.Error("expected start_no_incoming error")
	}
}

func TestValidateExitNoOutgoing(t *testing.T) {
	graph := makeSimpleGraph()
	graph.Edges = append(graph.Edges, &Edge{From: "exit", To: "a"})

	diagnostics := Validate(graph)
	found := false
	for _, d := range diagnostics {
		if d.Rule == "exit_no_outgoing" && d.Severity == SeverityError {
			found = true
		}
	}
	if !found {
		t.Error("expected exit_no_outgoing error")
	}
}

func TestValidateReachability(t *testing.T) {
	graph := makeSimpleGraph()
	graph.Nodes["orphan"] = &Node{ID: "orphan", Shape: "box", Attrs: map[string]string{}}

	diagnostics := Validate(graph)
	found := false
	for _, d := range diagnostics {
		if d.Rule == "reachability" && d.NodeID == "orphan" {
			found = true
		}
	}
	if !found {
		t.Error("expected reachability error for orphan node")
	}
}

func TestValidateEdgeTargetExists(t *testing.T) {
	graph := makeSimpleGraph()
	graph.Edges = append(graph.Edges, &Edge{From: "a", To: "nonexistent"})

	diagnostics := Validate(graph)
	found := false
	for _, d := range diagnostics {
		if d.Rule == "edge_target_exists" {
			found = true
		}
	}
	if !found {
		t.Error("expected edge_target_exists error")
	}
}

func TestValidateConditionSyntax(t *testing.T) {
	graph := makeSimpleGraph()
	// Valid condition should not produce error
	graph.Edges[1].Condition = "outcome=success"

	diagnostics := Validate(graph)
	for _, d := range diagnostics {
		if d.Rule == "condition_syntax" {
			t.Errorf("unexpected condition_syntax error: %s", d.Message)
		}
	}
}

func TestValidateOrRaise(t *testing.T) {
	graph := makeSimpleGraph()
	_, err := ValidateOrRaise(graph)
	if err != nil {
		t.Errorf("expected no error for valid graph, got: %v", err)
	}
}

func TestValidateOrRaiseFailure(t *testing.T) {
	graph := &Graph{
		Nodes: map[string]*Node{
			"a": {ID: "a", Shape: "box", Attrs: map[string]string{}},
		},
		Edges: []*Edge{},
	}

	_, err := ValidateOrRaise(graph)
	if err == nil {
		t.Error("expected validation error for graph without start/exit nodes")
	}
}

func TestValidatePromptOnLLMNodes(t *testing.T) {
	graph := makeSimpleGraph()
	// Node "a" has a prompt, so no warning.
	graph.Nodes["a"].Prompt = ""
	graph.Nodes["a"].Label = "a" // label == ID

	diagnostics := Validate(graph)
	found := false
	for _, d := range diagnostics {
		if d.Rule == "prompt_on_llm_nodes" && d.NodeID == "a" {
			found = true
		}
	}
	if !found {
		t.Error("expected prompt_on_llm_nodes warning for node without prompt or descriptive label")
	}
}

func TestValidateGoalGateHasRetry(t *testing.T) {
	graph := makeSimpleGraph()
	graph.Nodes["a"].GoalGate = true

	diagnostics := Validate(graph)
	found := false
	for _, d := range diagnostics {
		if d.Rule == "goal_gate_has_retry" {
			found = true
		}
	}
	if !found {
		t.Error("expected goal_gate_has_retry warning")
	}
}
