package llm

// ModelInfo describes a known model.
type ModelInfo struct {
	ID             string `json:"id"`
	Provider       string `json:"provider"`
	DisplayName    string `json:"display_name"`
	ContextWindow  int    `json:"context_window"`
	MaxOutput      int    `json:"max_output"`
	SupportsVision bool   `json:"supports_vision"`
	SupportsTools  bool   `json:"supports_tools"`
	SupportsReasoning bool `json:"supports_reasoning"`
}

// knownModels is the built-in model catalog.
var knownModels = []ModelInfo{
	// OpenAI
	{ID: "gpt-5.2", Provider: "openai", DisplayName: "GPT-5.2", ContextWindow: 128000, MaxOutput: 16384, SupportsVision: true, SupportsTools: true, SupportsReasoning: true},
	{ID: "gpt-4.1", Provider: "openai", DisplayName: "GPT-4.1", ContextWindow: 1000000, MaxOutput: 32768, SupportsVision: true, SupportsTools: true},
	{ID: "gpt-4o", Provider: "openai", DisplayName: "GPT-4o", ContextWindow: 128000, MaxOutput: 16384, SupportsVision: true, SupportsTools: true},
	{ID: "o3", Provider: "openai", DisplayName: "o3", ContextWindow: 200000, MaxOutput: 100000, SupportsTools: true, SupportsReasoning: true},

	// Anthropic
	{ID: "claude-opus-4-6", Provider: "anthropic", DisplayName: "Claude Opus 4.6", ContextWindow: 1000000, MaxOutput: 32000, SupportsVision: true, SupportsTools: true, SupportsReasoning: true},
	{ID: "claude-sonnet-4-5-20250929", Provider: "anthropic", DisplayName: "Claude Sonnet 4.5", ContextWindow: 200000, MaxOutput: 16384, SupportsVision: true, SupportsTools: true, SupportsReasoning: true},
	{ID: "claude-haiku-4-5-20251001", Provider: "anthropic", DisplayName: "Claude Haiku 4.5", ContextWindow: 200000, MaxOutput: 8192, SupportsVision: true, SupportsTools: true},

	// Gemini
	{ID: "gemini-2.5-pro", Provider: "gemini", DisplayName: "Gemini 2.5 Pro", ContextWindow: 1000000, MaxOutput: 65536, SupportsVision: true, SupportsTools: true, SupportsReasoning: true},
	{ID: "gemini-2.5-flash", Provider: "gemini", DisplayName: "Gemini 2.5 Flash", ContextWindow: 1000000, MaxOutput: 65536, SupportsVision: true, SupportsTools: true},
}

// GetModelInfo returns info about a known model by ID.
func GetModelInfo(modelID string) (ModelInfo, bool) {
	for _, m := range knownModels {
		if m.ID == modelID {
			return m, true
		}
	}
	return ModelInfo{}, false
}

// ListModels returns all known models, optionally filtered by provider.
func ListModels(provider string) []ModelInfo {
	if provider == "" {
		result := make([]ModelInfo, len(knownModels))
		copy(result, knownModels)
		return result
	}
	var result []ModelInfo
	for _, m := range knownModels {
		if m.Provider == provider {
			result = append(result, m)
		}
	}
	return result
}
