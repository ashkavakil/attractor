package pipeline

import (
	"testing"
)

func TestParseSimpleLinearPipeline(t *testing.T) {
	source := `digraph Simple {
		graph [goal="Run tests and report"]

		start [shape=Mdiamond, label="Start"]
		exit  [shape=Msquare, label="Exit"]
		run_tests [label="Run Tests", prompt="Run the test suite"]
		report    [label="Report", prompt="Summarize results"]

		start -> run_tests -> report -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if graph.Goal != "Run tests and report" {
		t.Errorf("expected goal %q, got %q", "Run tests and report", graph.Goal)
	}

	if len(graph.Nodes) != 4 {
		t.Errorf("expected 4 nodes, got %d", len(graph.Nodes))
	}

	if len(graph.Edges) != 3 {
		t.Errorf("expected 3 edges, got %d", len(graph.Edges))
	}

	// Check start node
	start := graph.Nodes["start"]
	if start == nil {
		t.Fatal("start node not found")
	}
	if start.Shape != "Mdiamond" {
		t.Errorf("expected start shape Mdiamond, got %q", start.Shape)
	}

	// Check exit node
	exit := graph.Nodes["exit"]
	if exit == nil {
		t.Fatal("exit node not found")
	}
	if exit.Shape != "Msquare" {
		t.Errorf("expected exit shape Msquare, got %q", exit.Shape)
	}

	// Check prompt
	rt := graph.Nodes["run_tests"]
	if rt == nil {
		t.Fatal("run_tests node not found")
	}
	if rt.Prompt != "Run the test suite" {
		t.Errorf("expected prompt %q, got %q", "Run the test suite", rt.Prompt)
	}
}

func TestParseGraphLevelAttributes(t *testing.T) {
	source := `digraph Test {
		graph [goal="Test goal", label="Test Label", default_max_retry=3]
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		start -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if graph.Goal != "Test goal" {
		t.Errorf("expected goal %q, got %q", "Test goal", graph.Goal)
	}
	if graph.Label != "Test Label" {
		t.Errorf("expected label %q, got %q", "Test Label", graph.Label)
	}
	if graph.DefaultMaxRetry != 3 {
		t.Errorf("expected default_max_retry 3, got %d", graph.DefaultMaxRetry)
	}
}

func TestParseMultilineNodeAttributes(t *testing.T) {
	source := `digraph Test {
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		my_node [
			label="My Node",
			shape=box,
			prompt="Do something complex",
			max_retries=3,
			goal_gate=true
		]
		start -> my_node -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	node := graph.Nodes["my_node"]
	if node == nil {
		t.Fatal("my_node not found")
	}
	if node.Label != "My Node" {
		t.Errorf("expected label %q, got %q", "My Node", node.Label)
	}
	if node.MaxRetries != 3 {
		t.Errorf("expected max_retries 3, got %d", node.MaxRetries)
	}
	if !node.GoalGate {
		t.Error("expected goal_gate=true")
	}
}

func TestParseChainedEdges(t *testing.T) {
	source := `digraph Test {
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		a [label="A"]
		b [label="B"]
		c [label="C"]
		start -> a -> b -> c -> exit [label="next"]
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// start->a, a->b, b->c, c->exit = 4 edges
	if len(graph.Edges) != 4 {
		t.Errorf("expected 4 edges from chained declaration, got %d", len(graph.Edges))
	}

	// All should have label "next"
	for _, e := range graph.Edges {
		if e.Label != "next" {
			t.Errorf("expected all chained edges to have label 'next', got %q for %s->%s", e.Label, e.From, e.To)
		}
	}
}

func TestParseEdgeAttributes(t *testing.T) {
	source := `digraph Test {
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		a [label="A"]
		start -> a
		a -> exit [label="Done", condition="outcome=success", weight=5]
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Find the a->exit edge
	for _, e := range graph.Edges {
		if e.From == "a" && e.To == "exit" {
			if e.Label != "Done" {
				t.Errorf("expected label %q, got %q", "Done", e.Label)
			}
			if e.Condition != "outcome=success" {
				t.Errorf("expected condition %q, got %q", "outcome=success", e.Condition)
			}
			if e.Weight != 5 {
				t.Errorf("expected weight 5, got %d", e.Weight)
			}
			return
		}
	}
	t.Error("a->exit edge not found")
}

func TestParseNodeDefaults(t *testing.T) {
	source := `digraph Test {
		node [shape=box, timeout="900s"]
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		a [label="A"]
		start -> a -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	a := graph.Nodes["a"]
	if a == nil {
		t.Fatal("node a not found")
	}
	if a.Shape != "box" {
		t.Errorf("expected default shape box, got %q", a.Shape)
	}
}

func TestParseComments(t *testing.T) {
	source := `// This is a comment
	digraph Test {
		/* Block comment */
		start [shape=Mdiamond] // inline comment
		exit  [shape=Msquare]
		start -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if len(graph.Nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(graph.Nodes))
	}
}

func TestParseQuotedAndUnquoted(t *testing.T) {
	source := `digraph Test {
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		a [label="Quoted Label", shape=box]
		start -> a -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	a := graph.Nodes["a"]
	if a.Label != "Quoted Label" {
		t.Errorf("expected %q, got %q", "Quoted Label", a.Label)
	}
	if a.Shape != "box" {
		t.Errorf("expected shape box (unquoted), got %q", a.Shape)
	}
}

func TestParseSubgraph(t *testing.T) {
	source := `digraph Test {
		start [shape=Mdiamond]
		exit  [shape=Msquare]

		subgraph cluster_loop {
			node [timeout="900s"]
			Plan      [label="Plan"]
			Implement [label="Implement"]
		}

		start -> Plan -> Implement -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if _, ok := graph.Nodes["Plan"]; !ok {
		t.Error("Plan node not found")
	}
	if _, ok := graph.Nodes["Implement"]; !ok {
		t.Error("Implement node not found")
	}
}

func TestParseBranchingWorkflow(t *testing.T) {
	source := `digraph Branch {
		graph [goal="Implement and validate a feature"]
		node [shape=box, timeout="900s"]

		start     [shape=Mdiamond, label="Start"]
		exit      [shape=Msquare, label="Exit"]
		plan      [label="Plan", prompt="Plan the implementation"]
		implement [label="Implement", prompt="Implement the plan"]
		validate  [label="Validate", prompt="Run tests"]
		gate      [shape=diamond, label="Tests passing?"]

		start -> plan -> implement -> validate -> gate
		gate -> exit      [label="Yes", condition="outcome=success"]
		gate -> implement [label="No", condition="outcome!=success"]
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if len(graph.Nodes) != 6 {
		t.Errorf("expected 6 nodes, got %d", len(graph.Nodes))
	}

	gate := graph.Nodes["gate"]
	if gate.Shape != "diamond" {
		t.Errorf("expected gate shape diamond, got %q", gate.Shape)
	}

	// Should have 6 edges total
	if len(graph.Edges) != 6 {
		t.Errorf("expected 6 edges, got %d", len(graph.Edges))
	}
}

func TestParseModelStylesheet(t *testing.T) {
	source := `digraph Test {
		graph [
			goal="Test",
			model_stylesheet="* { llm_model: claude-sonnet-4-5; } .code { llm_model: claude-opus-4-6; } #review { reasoning_effort: high; }"
		]
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		start -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if graph.ModelStylesheet == "" {
		t.Error("model_stylesheet not parsed")
	}
}

func TestParseTopLevelKeyValue(t *testing.T) {
	source := `digraph Test {
		rankdir=LR
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		start -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if graph.Attrs["rankdir"] != "LR" {
		t.Errorf("expected rankdir=LR, got %q", graph.Attrs["rankdir"])
	}
}
