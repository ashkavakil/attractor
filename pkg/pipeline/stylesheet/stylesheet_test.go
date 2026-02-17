package stylesheet

import (
	"testing"

	"github.com/ashka-vakil/attractor/pkg/pipeline"
)

func TestParseEmptyStylesheet(t *testing.T) {
	ss, err := Parse("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ss.Rules) != 0 {
		t.Errorf("expected 0 rules, got %d", len(ss.Rules))
	}
}

func TestParseUniversalSelector(t *testing.T) {
	ss, err := Parse(`* { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(ss.Rules) != 1 {
		t.Fatalf("expected 1 rule, got %d", len(ss.Rules))
	}
	if ss.Rules[0].SelectorType != SelectorUniversal {
		t.Errorf("expected universal selector")
	}
	if ss.Rules[0].Properties["llm_model"] != "claude-sonnet-4-5" {
		t.Errorf("expected llm_model claude-sonnet-4-5, got %q", ss.Rules[0].Properties["llm_model"])
	}
}

func TestParseClassSelector(t *testing.T) {
	ss, err := Parse(`.code { llm_model: claude-opus-4-6; }`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if ss.Rules[0].SelectorType != SelectorClass {
		t.Errorf("expected class selector")
	}
	if ss.Rules[0].Selector != "code" {
		t.Errorf("expected selector 'code', got %q", ss.Rules[0].Selector)
	}
}

func TestParseIDSelector(t *testing.T) {
	ss, err := Parse(`#review { reasoning_effort: high; }`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if ss.Rules[0].SelectorType != SelectorID {
		t.Errorf("expected ID selector")
	}
	if ss.Rules[0].Selector != "review" {
		t.Errorf("expected selector 'review', got %q", ss.Rules[0].Selector)
	}
}

func TestParseShapeSelector(t *testing.T) {
	ss, err := Parse(`box { llm_model: claude-opus-4-6; }`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if ss.Rules[0].SelectorType != SelectorShape {
		t.Errorf("expected shape selector, got %d", ss.Rules[0].SelectorType)
	}
	if ss.Rules[0].Selector != "box" {
		t.Errorf("expected selector 'box', got %q", ss.Rules[0].Selector)
	}
	if ss.Rules[0].Specificity != SpecificityShape {
		t.Errorf("expected specificity %d, got %d", SpecificityShape, ss.Rules[0].Specificity)
	}
}

func TestShapeSelectorApply(t *testing.T) {
	ss, err := Parse(`box { llm_model: box-model; }`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	graph := &pipeline.Graph{
		Nodes: map[string]*pipeline.Node{
			"a": {ID: "a", Shape: "box", Attrs: map[string]string{}},
			"b": {ID: "b", Shape: "diamond", Attrs: map[string]string{}},
		},
	}

	ss.Apply(graph)

	if graph.Nodes["a"].LLMModel != "box-model" {
		t.Errorf("expected 'box-model' for box-shaped node, got %q", graph.Nodes["a"].LLMModel)
	}
	if graph.Nodes["b"].LLMModel != "" {
		t.Errorf("expected empty model for diamond-shaped node, got %q", graph.Nodes["b"].LLMModel)
	}
}

func TestSpecificityOrder(t *testing.T) {
	// universal(0) < shape(1) < class(2) < ID(3)
	ss, err := Parse(`
		* { llm_model: universal; }
		box { llm_model: shape-model; }
		.code { llm_model: class-model; }
		#specific { llm_model: id-model; }
	`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	graph := &pipeline.Graph{
		Nodes: map[string]*pipeline.Node{
			"generic":  {ID: "generic", Shape: "diamond", Class: "", Attrs: map[string]string{}},
			"boxed":    {ID: "boxed", Shape: "box", Class: "", Attrs: map[string]string{}},
			"coder":    {ID: "coder", Shape: "box", Class: "code", Attrs: map[string]string{}},
			"specific": {ID: "specific", Shape: "box", Class: "code", Attrs: map[string]string{}},
		},
	}

	ss.Apply(graph)

	// generic: only * matches -> universal
	if graph.Nodes["generic"].LLMModel != "universal" {
		t.Errorf("expected 'universal' for generic, got %q", graph.Nodes["generic"].LLMModel)
	}

	// boxed: * and box match -> shape-model wins
	if graph.Nodes["boxed"].LLMModel != "shape-model" {
		t.Errorf("expected 'shape-model' for boxed, got %q", graph.Nodes["boxed"].LLMModel)
	}

	// coder: *, box, .code match -> class-model wins (higher specificity)
	if graph.Nodes["coder"].LLMModel != "class-model" {
		t.Errorf("expected 'class-model' for coder, got %q", graph.Nodes["coder"].LLMModel)
	}

	// specific: *, box, .code, #specific match -> id-model wins (highest specificity)
	if graph.Nodes["specific"].LLMModel != "id-model" {
		t.Errorf("expected 'id-model' for specific, got %q", graph.Nodes["specific"].LLMModel)
	}
}

func TestExplicitNodeAttributeOverridesStylesheet(t *testing.T) {
	ss, err := Parse(`* { llm_model: default-model; }`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	graph := &pipeline.Graph{
		Nodes: map[string]*pipeline.Node{
			"a": {ID: "a", LLMModel: "explicit-model", Attrs: map[string]string{}},
			"b": {ID: "b", Attrs: map[string]string{}},
		},
	}

	ss.Apply(graph)

	if graph.Nodes["a"].LLMModel != "explicit-model" {
		t.Errorf("expected 'explicit-model', got %q", graph.Nodes["a"].LLMModel)
	}
	if graph.Nodes["b"].LLMModel != "default-model" {
		t.Errorf("expected 'default-model', got %q", graph.Nodes["b"].LLMModel)
	}
}

func TestValidateStylesheet(t *testing.T) {
	ss, err := Parse(`* { llm_model: test; llm_provider: openai; reasoning_effort: high; }`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if err := ss.Validate(); err != nil {
		t.Errorf("expected valid stylesheet, got: %v", err)
	}
}

func TestModelAliasProperty(t *testing.T) {
	// DoD example uses "model" as alias for "llm_model"
	ss, err := Parse(`box { model: claude-opus-4-6; }`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	graph := &pipeline.Graph{
		Nodes: map[string]*pipeline.Node{
			"a": {ID: "a", Shape: "box", Attrs: map[string]string{}},
		},
	}

	ss.Apply(graph)
	if graph.Nodes["a"].LLMModel != "claude-opus-4-6" {
		t.Errorf("expected 'claude-opus-4-6' via model alias, got %q", graph.Nodes["a"].LLMModel)
	}
}

func TestEqualsDeclarationSyntax(t *testing.T) {
	// DoD examples use "=" in declarations too
	ss, err := Parse(`#review { reasoning_effort = high; }`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if ss.Rules[0].Properties["reasoning_effort"] != "high" {
		t.Errorf("expected 'high', got %q", ss.Rules[0].Properties["reasoning_effort"])
	}
}

func TestMultipleRules(t *testing.T) {
	ss, err := Parse(`
		* { llm_model: base; llm_provider: anthropic; }
		.fast { llm_model: flash; llm_provider: gemini; }
		#critical { reasoning_effort: high; }
	`)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(ss.Rules) != 3 {
		t.Errorf("expected 3 rules, got %d", len(ss.Rules))
	}
}
