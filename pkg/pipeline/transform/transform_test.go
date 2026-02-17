package transform

import (
	"testing"

	"github.com/ashka-vakil/attractor/pkg/pipeline"
)

func TestVariableExpansion(t *testing.T) {
	graph := &pipeline.Graph{
		Goal: "Create a web app",
		Nodes: map[string]*pipeline.Node{
			"plan": {ID: "plan", Prompt: "Plan for: $goal", Attrs: map[string]string{}},
			"impl": {ID: "impl", Prompt: "Implement the plan", Attrs: map[string]string{}},
		},
	}

	result := VariableExpansion().Apply(graph)

	if result.Nodes["plan"].Prompt != "Plan for: Create a web app" {
		t.Errorf("expected expanded prompt, got %q", result.Nodes["plan"].Prompt)
	}
	if result.Nodes["impl"].Prompt != "Implement the plan" {
		t.Errorf("prompt without $goal should be unchanged, got %q", result.Nodes["impl"].Prompt)
	}
}

func TestStylesheetApplication(t *testing.T) {
	graph := &pipeline.Graph{
		ModelStylesheet: `* { llm_model: claude-sonnet-4-5; } .code { llm_model: claude-opus-4-6; }`,
		Nodes: map[string]*pipeline.Node{
			"a": {ID: "a", Class: "", Attrs: map[string]string{}},
			"b": {ID: "b", Class: "code", Attrs: map[string]string{}},
		},
	}

	result := StylesheetApplication().Apply(graph)

	if result.Nodes["a"].LLMModel != "claude-sonnet-4-5" {
		t.Errorf("expected 'claude-sonnet-4-5', got %q", result.Nodes["a"].LLMModel)
	}
	if result.Nodes["b"].LLMModel != "claude-opus-4-6" {
		t.Errorf("expected 'claude-opus-4-6', got %q", result.Nodes["b"].LLMModel)
	}
}

func TestApplyAll(t *testing.T) {
	graph := &pipeline.Graph{
		Goal:            "Test",
		ModelStylesheet: `* { llm_model: test-model; }`,
		Nodes: map[string]*pipeline.Node{
			"a": {ID: "a", Prompt: "Do $goal", Attrs: map[string]string{}},
		},
	}

	result := ApplyAll(graph, DefaultTransforms()...)

	if result.Nodes["a"].Prompt != "Do Test" {
		t.Errorf("expected expanded prompt, got %q", result.Nodes["a"].Prompt)
	}
	if result.Nodes["a"].LLMModel != "test-model" {
		t.Errorf("expected 'test-model', got %q", result.Nodes["a"].LLMModel)
	}
}

func TestCustomTransform(t *testing.T) {
	customTransform := TransformFunc(func(graph *pipeline.Graph) *pipeline.Graph {
		for _, node := range graph.Nodes {
			if node.Prompt == "" {
				node.Prompt = "default prompt for " + node.ID
			}
		}
		return graph
	})

	graph := &pipeline.Graph{
		Nodes: map[string]*pipeline.Node{
			"a": {ID: "a", Prompt: "", Attrs: map[string]string{}},
			"b": {ID: "b", Prompt: "existing", Attrs: map[string]string{}},
		},
	}

	result := customTransform.Apply(graph)

	if result.Nodes["a"].Prompt != "default prompt for a" {
		t.Errorf("expected custom transform to set prompt, got %q", result.Nodes["a"].Prompt)
	}
	if result.Nodes["b"].Prompt != "existing" {
		t.Errorf("expected existing prompt preserved, got %q", result.Nodes["b"].Prompt)
	}
}

func TestDefaultTransforms(t *testing.T) {
	transforms := DefaultTransforms()
	if len(transforms) != 2 {
		t.Errorf("expected 2 default transforms, got %d", len(transforms))
	}
}

func TestEmptyStylesheetNoop(t *testing.T) {
	graph := &pipeline.Graph{
		ModelStylesheet: "",
		Nodes: map[string]*pipeline.Node{
			"a": {ID: "a", LLMModel: "original", Attrs: map[string]string{}},
		},
	}

	result := StylesheetApplication().Apply(graph)

	if result.Nodes["a"].LLMModel != "original" {
		t.Errorf("expected original model preserved when no stylesheet, got %q", result.Nodes["a"].LLMModel)
	}
}
