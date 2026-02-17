package llm

import "testing"

func TestGetModelInfo(t *testing.T) {
	info, ok := GetModelInfo("claude-opus-4-6")
	if !ok {
		t.Fatal("expected to find claude-opus-4-6")
	}
	if info.Provider != "anthropic" {
		t.Errorf("expected provider anthropic, got %q", info.Provider)
	}
	if info.ContextWindow != 1000000 {
		t.Errorf("expected 1M context window, got %d", info.ContextWindow)
	}
}

func TestGetModelInfoNotFound(t *testing.T) {
	_, ok := GetModelInfo("nonexistent-model")
	if ok {
		t.Error("expected model not found")
	}
}

func TestListModels(t *testing.T) {
	all := ListModels("")
	if len(all) == 0 {
		t.Error("expected non-empty model list")
	}

	anthropic := ListModels("anthropic")
	if len(anthropic) == 0 {
		t.Error("expected anthropic models")
	}
	for _, m := range anthropic {
		if m.Provider != "anthropic" {
			t.Errorf("expected anthropic provider, got %q", m.Provider)
		}
	}
}

func TestListModelsEmpty(t *testing.T) {
	result := ListModels("nonexistent-provider")
	if len(result) != 0 {
		t.Errorf("expected empty list for unknown provider, got %d", len(result))
	}
}
