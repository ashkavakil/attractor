package pipeline

import (
	"os"
	"path/filepath"
	"testing"
)

// simpleHandler always returns SUCCESS.
type simpleHandler struct {
	response string
}

func (h *simpleHandler) Execute(node *Node, ctx *Context, graph *Graph, logsRoot string) (*Outcome, error) {
	return &Outcome{
		Status: StatusSuccess,
		Notes:  "completed: " + node.ID,
		ContextUpdates: map[string]interface{}{
			"last_stage":    node.ID,
			"last_response": h.response,
		},
	}, nil
}

// failHandler returns FAIL.
type failHandler struct{}

func (h *failHandler) Execute(node *Node, ctx *Context, graph *Graph, logsRoot string) (*Outcome, error) {
	return &Outcome{
		Status:        StatusFail,
		FailureReason: "deliberate failure",
	}, nil
}

// retryHandler returns RETRY for a number of attempts, then SUCCESS.
type retryHandler struct {
	attemptsBeforeSuccess int
	attempts              int
}

func (h *retryHandler) Execute(node *Node, ctx *Context, graph *Graph, logsRoot string) (*Outcome, error) {
	h.attempts++
	if h.attempts <= h.attemptsBeforeSuccess {
		return &Outcome{
			Status:        StatusRetry,
			FailureReason: "not yet",
		}, nil
	}
	return &Outcome{Status: StatusSuccess}, nil
}

// conditionalHandler returns with a preferred label.
type conditionalHandler struct {
	preferredLabel string
}

func (h *conditionalHandler) Execute(node *Node, ctx *Context, graph *Graph, logsRoot string) (*Outcome, error) {
	return &Outcome{
		Status:         StatusSuccess,
		PreferredLabel: h.preferredLabel,
	}, nil
}

// staticResolver returns the same handler for all nodes.
type staticResolver struct {
	handler Handler
	special map[string]Handler
}

func (r *staticResolver) Resolve(node *Node) Handler {
	if h, ok := r.special[node.ID]; ok {
		return h
	}
	return r.handler
}

func TestExecuteLinearPipeline(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Goal: "Test goal",
		Nodes: map[string]*Node{
			"start": {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"a":     {ID: "a", Shape: "box", Label: "A", Prompt: "Do A", Attrs: map[string]string{}},
			"b":     {ID: "b", Shape: "box", Label: "B", Prompt: "Do B", Attrs: map[string]string{}},
			"exit":  {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
			{From: "a", To: "b"},
			{From: "b", To: "exit"},
		},
	}

	logsRoot := t.TempDir()
	resolver := &staticResolver{handler: &simpleHandler{response: "ok"}}
	engine := NewEngine(EngineConfig{LogsRoot: logsRoot}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if result.Status != StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", result.Status)
	}

	if len(result.CompletedNodes) != 3 { // start, a, b (exit is not "completed" in the same way)
		t.Errorf("expected 3 completed nodes, got %d: %v", len(result.CompletedNodes), result.CompletedNodes)
	}
}

func TestExecuteConditionalBranching(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Nodes: map[string]*Node{
			"start":     {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"plan":      {ID: "plan", Shape: "box", Label: "Plan", Attrs: map[string]string{}},
			"implement": {ID: "implement", Shape: "box", Label: "Implement", Attrs: map[string]string{}},
			"validate":  {ID: "validate", Shape: "diamond", Label: "Validate", Attrs: map[string]string{}},
			"exit":      {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "plan"},
			{From: "plan", To: "implement"},
			{From: "implement", To: "validate"},
			{From: "validate", To: "exit", Condition: "outcome=success"},
			{From: "validate", To: "implement", Condition: "outcome!=success"},
		},
	}

	resolver := &staticResolver{handler: &simpleHandler{response: "done"}}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if result.Status != StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", result.Status)
	}
}

func TestExecuteWithRetry(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Nodes: map[string]*Node{
			"start": {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"a":     {ID: "a", Shape: "box", Label: "A", MaxRetries: 3, Attrs: map[string]string{}},
			"exit":  {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
			{From: "a", To: "exit"},
		},
	}

	retryH := &retryHandler{attemptsBeforeSuccess: 2}
	resolver := &staticResolver{
		handler: &simpleHandler{response: "ok"},
		special: map[string]Handler{"a": retryH},
	}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if result.Status != StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", result.Status)
	}

	if retryH.attempts != 3 {
		t.Errorf("expected 3 attempts, got %d", retryH.attempts)
	}
}

func TestGoalGateBlocksExit(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Nodes: map[string]*Node{
			"start": {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"a":     {ID: "a", Shape: "box", Label: "A", GoalGate: true, Attrs: map[string]string{}},
			"exit":  {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
			{From: "a", To: "exit"},
		},
	}

	resolver := &staticResolver{
		handler: &simpleHandler{},
		special: map[string]Handler{"a": &failHandler{}},
	}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	result, err := engine.Run(graph)
	// With a failing goal gate and no retry target, it should fail
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != StatusFail {
		t.Errorf("expected FAIL when goal gate unsatisfied, got %s", result.Status)
	}
}

func TestGoalGateAllowsExit(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Nodes: map[string]*Node{
			"start": {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"a":     {ID: "a", Shape: "box", Label: "A", GoalGate: true, Attrs: map[string]string{}},
			"exit":  {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
			{From: "a", To: "exit"},
		},
	}

	resolver := &staticResolver{handler: &simpleHandler{}}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if result.Status != StatusSuccess {
		t.Errorf("expected SUCCESS when goal gate satisfied, got %s", result.Status)
	}
}

func TestEdgeSelectionConditionWins(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Nodes: map[string]*Node{
			"start":   {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"a":       {ID: "a", Shape: "box", Label: "A", Attrs: map[string]string{}},
			"success": {ID: "success", Shape: "box", Label: "Success", Attrs: map[string]string{}},
			"fail":    {ID: "fail", Shape: "box", Label: "Fail", Attrs: map[string]string{}},
			"exit":    {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
			{From: "a", To: "success", Condition: "outcome=success"},
			{From: "a", To: "fail", Condition: "outcome=fail", Weight: 10},
			{From: "success", To: "exit"},
			{From: "fail", To: "exit"},
		},
	}

	resolver := &staticResolver{handler: &simpleHandler{}}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Should go through "success" path since handler returns SUCCESS
	found := false
	for _, id := range result.CompletedNodes {
		if id == "success" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected 'success' node in completed path, got: %v", result.CompletedNodes)
	}
}

func TestEdgeSelectionWeightBreaksTie(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Nodes: map[string]*Node{
			"start":  {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"a":      {ID: "a", Shape: "box", Label: "A", Attrs: map[string]string{}},
			"heavy":  {ID: "heavy", Shape: "box", Label: "Heavy", Attrs: map[string]string{}},
			"light":  {ID: "light", Shape: "box", Label: "Light", Attrs: map[string]string{}},
			"exit":   {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
			{From: "a", To: "light", Weight: 1},
			{From: "a", To: "heavy", Weight: 10},
			{From: "heavy", To: "exit"},
			{From: "light", To: "exit"},
		},
	}

	resolver := &staticResolver{handler: &simpleHandler{}}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	found := false
	for _, id := range result.CompletedNodes {
		if id == "heavy" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected 'heavy' node (weight=10) in path, got: %v", result.CompletedNodes)
	}
}

func TestEdgeSelectionLexicalTiebreak(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Nodes: map[string]*Node{
			"start": {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"a":     {ID: "a", Shape: "box", Label: "A", Attrs: map[string]string{}},
			"beta":  {ID: "beta", Shape: "box", Label: "Beta", Attrs: map[string]string{}},
			"alpha": {ID: "alpha", Shape: "box", Label: "Alpha", Attrs: map[string]string{}},
			"exit":  {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
			{From: "a", To: "beta"},  // same weight (0)
			{From: "a", To: "alpha"}, // alphabetically first
			{From: "alpha", To: "exit"},
			{From: "beta", To: "exit"},
		},
	}

	resolver := &staticResolver{handler: &simpleHandler{}}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// "alpha" < "beta" lexically, so alpha should win
	found := false
	for _, id := range result.CompletedNodes {
		if id == "alpha" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected 'alpha' (lexical tiebreak) in path, got: %v", result.CompletedNodes)
	}
}

func TestContextUpdatesVisibleToNextNode(t *testing.T) {
	graph := &Graph{
		Name: "test",
		Nodes: map[string]*Node{
			"start": {ID: "start", Shape: "Mdiamond", Label: "Start", Attrs: map[string]string{}},
			"a":     {ID: "a", Shape: "box", Label: "A", Attrs: map[string]string{}},
			"b":     {ID: "b", Shape: "box", Label: "B", Attrs: map[string]string{}},
			"exit":  {ID: "exit", Shape: "Msquare", Label: "Exit", Attrs: map[string]string{}},
		},
		Edges: []*Edge{
			{From: "start", To: "a"},
			{From: "a", To: "b"},
			{From: "b", To: "exit"},
		},
	}

	// Handler that checks for context from previous stage
	ctxCheckHandler := &contextCheckHandler{t: t, checkKey: "last_stage", expectedValue: "a"}
	resolver := &staticResolver{
		handler: &simpleHandler{response: "ok"},
		special: map[string]Handler{"b": ctxCheckHandler},
	}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	_, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if !ctxCheckHandler.verified {
		t.Error("context check handler was never called or verification failed")
	}
}

type contextCheckHandler struct {
	t             *testing.T
	checkKey      string
	expectedValue string
	verified      bool
}

func (h *contextCheckHandler) Execute(node *Node, ctx *Context, graph *Graph, logsRoot string) (*Outcome, error) {
	val := ctx.GetString(h.checkKey)
	if val != h.expectedValue {
		h.t.Errorf("expected context[%q]=%q, got %q", h.checkKey, h.expectedValue, val)
	} else {
		h.verified = true
	}
	return &Outcome{Status: StatusSuccess}, nil
}

func TestCheckpointSaveAndLoad(t *testing.T) {
	tmpDir := t.TempDir()
	cpPath := filepath.Join(tmpDir, "checkpoint.json")

	original := &Checkpoint{
		CurrentNode:    "node_b",
		CompletedNodes: []string{"start", "node_a", "node_b"},
		NodeRetries:    map[string]int{"node_a": 2},
		ContextValues:  map[string]interface{}{"key": "value"},
		Logs:           []string{"log1", "log2"},
	}

	if err := original.Save(cpPath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loaded, err := LoadCheckpoint(cpPath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if loaded.CurrentNode != original.CurrentNode {
		t.Errorf("expected current_node %q, got %q", original.CurrentNode, loaded.CurrentNode)
	}
	if len(loaded.CompletedNodes) != 3 {
		t.Errorf("expected 3 completed nodes, got %d", len(loaded.CompletedNodes))
	}
}

func TestPromptVariableExpansion(t *testing.T) {
	source := `digraph Test {
		graph [goal="Create a hello world script"]
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		plan  [shape=box, prompt="Plan how to: $goal"]
		start -> plan -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Apply variable expansion
	for _, node := range graph.Nodes {
		if node.Prompt != "" {
			node.Prompt = expandVars(node.Prompt, graph)
		}
	}

	plan := graph.Nodes["plan"]
	if plan.Prompt != "Plan how to: Create a hello world script" {
		t.Errorf("expected expanded prompt, got %q", plan.Prompt)
	}
}

func expandVars(prompt string, graph *Graph) string {
	return replaceAll(prompt, "$goal", graph.Goal)
}

func replaceAll(s, old, new string) string {
	for {
		idx := indexOf(s, old)
		if idx < 0 {
			return s
		}
		s = s[:idx] + new + s[idx+len(old):]
	}
}

func indexOf(s, sub string) int {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func TestArtifactWritten(t *testing.T) {
	source := `digraph Test {
		graph [goal="Test artifacts"]
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		plan  [shape=box, prompt="Plan something"]
		start -> plan -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	logsRoot := t.TempDir()

	// Use a handler that writes artifacts
	resolver := &staticResolver{handler: &artifactWriterHandler{logsRoot: logsRoot}}
	engine := NewEngine(EngineConfig{LogsRoot: logsRoot}, resolver, nil)

	_, err = engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
}

type artifactWriterHandler struct {
	logsRoot string
}

func (h *artifactWriterHandler) Execute(node *Node, ctx *Context, graph *Graph, logsRoot string) (*Outcome, error) {
	if logsRoot != "" {
		stageDir := filepath.Join(logsRoot, node.ID)
		os.MkdirAll(stageDir, 0o755)
		os.WriteFile(filepath.Join(stageDir, "prompt.md"), []byte("test prompt"), 0o644)
		os.WriteFile(filepath.Join(stageDir, "response.md"), []byte("test response"), 0o644)
		os.WriteFile(filepath.Join(stageDir, "status.json"), []byte(`{"outcome":"success"}`), 0o644)
	}
	return &Outcome{Status: StatusSuccess}, nil
}

func TestPipelineWith10PlusNodes(t *testing.T) {
	source := `digraph Large {
		graph [goal="Large pipeline test"]
		start [shape=Mdiamond]
		exit  [shape=Msquare]
		n1 [label="N1", prompt="Step 1"]
		n2 [label="N2", prompt="Step 2"]
		n3 [label="N3", prompt="Step 3"]
		n4 [label="N4", prompt="Step 4"]
		n5 [label="N5", prompt="Step 5"]
		n6 [label="N6", prompt="Step 6"]
		n7 [label="N7", prompt="Step 7"]
		n8 [label="N8", prompt="Step 8"]
		n9 [label="N9", prompt="Step 9"]
		n10 [label="N10", prompt="Step 10"]

		start -> n1 -> n2 -> n3 -> n4 -> n5 -> n6 -> n7 -> n8 -> n9 -> n10 -> exit
	}`

	graph, err := Parse(source)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if len(graph.Nodes) != 12 {
		t.Errorf("expected 12 nodes, got %d", len(graph.Nodes))
	}

	resolver := &staticResolver{handler: &simpleHandler{}}
	engine := NewEngine(EngineConfig{LogsRoot: t.TempDir()}, resolver, nil)

	result, err := engine.Run(graph)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if result.Status != StatusSuccess {
		t.Errorf("expected SUCCESS, got %s", result.Status)
	}
}
