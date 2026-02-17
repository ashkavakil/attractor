package agent

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/ashka-vakil/attractor/pkg/llm"
)

// mockLLMAdapter is a test adapter that returns a configured response.
type mockLLMAdapter struct {
	responses []*llm.Response
	callIdx   int
}

func (m *mockLLMAdapter) Name() string { return "mock" }
func (m *mockLLMAdapter) Close() error { return nil }
func (m *mockLLMAdapter) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if m.callIdx >= len(m.responses) {
		return &llm.Response{
			Content:      "Done.",
			FinishReason: llm.FinishReasonStop,
			CreatedAt:    time.Now(),
		}, nil
	}
	resp := m.responses[m.callIdx]
	m.callIdx++
	return resp, nil
}
func (m *mockLLMAdapter) Stream(ctx context.Context, req *llm.Request) (<-chan llm.StreamEvent, error) {
	ch := make(chan llm.StreamEvent, 1)
	close(ch)
	return ch, nil
}

// mockEnv is a test execution environment.
type mockEnv struct {
	results map[string]string
}

func (m *mockEnv) Execute(ctx context.Context, toolName string, arguments json.RawMessage) (string, error) {
	if result, ok := m.results[toolName]; ok {
		return result, nil
	}
	return "ok", nil
}

func TestSessionCreation(t *testing.T) {
	client := llm.NewClient(llm.WithProvider("mock", &mockLLMAdapter{}))
	profile := DefaultAnthropicProfile("claude-sonnet-4-5-20250929")
	config := DefaultSessionConfig()

	session := NewSession(client, profile, &mockEnv{results: map[string]string{}}, config)

	if session.ID == "" {
		t.Error("session should have an ID")
	}
	if session.State != StateIdle {
		t.Errorf("expected IDLE, got %s", session.State)
	}
}

func TestSessionSubmitNaturalCompletion(t *testing.T) {
	adapter := &mockLLMAdapter{
		responses: []*llm.Response{
			{Content: "Hello, I can help with that!", FinishReason: llm.FinishReasonStop, CreatedAt: time.Now()},
		},
	}
	client := llm.NewClient(llm.WithProvider("mock", adapter))
	profile := DefaultAnthropicProfile("test-model")
	session := NewSession(client, profile, &mockEnv{results: map[string]string{}}, DefaultSessionConfig())

	err := session.Submit(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	if session.State != StateIdle {
		t.Errorf("expected IDLE after completion, got %s", session.State)
	}
	if len(session.History) < 2 {
		t.Errorf("expected at least 2 turns (user + assistant), got %d", len(session.History))
	}
}

func TestSessionToolCallLoop(t *testing.T) {
	adapter := &mockLLMAdapter{
		responses: []*llm.Response{
			{
				Content:      "",
				FinishReason: llm.FinishReasonToolCalls,
				ToolCalls: []llm.ToolCall{
					{ID: "call-1", Name: "read_file", Arguments: json.RawMessage(`{"path":"test.txt"}`)},
				},
				CreatedAt: time.Now(),
			},
			{
				Content:      "The file contains test data.",
				FinishReason: llm.FinishReasonStop,
				CreatedAt:    time.Now(),
			},
		},
	}
	client := llm.NewClient(llm.WithProvider("mock", adapter))
	profile := DefaultAnthropicProfile("test-model")
	env := &mockEnv{results: map[string]string{
		"read_file": "test content",
	}}
	session := NewSession(client, profile, env, DefaultSessionConfig())

	err := session.Submit(context.Background(), "Read test.txt")
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	// Should have: user turn, assistant turn (tool call), tool results, assistant turn (text)
	if len(session.History) < 4 {
		t.Errorf("expected at least 4 turns, got %d", len(session.History))
	}
}

func TestSessionSteering(t *testing.T) {
	callCount := 0
	adapter := &mockLLMAdapter{
		responses: []*llm.Response{
			{
				Content:      "",
				FinishReason: llm.FinishReasonToolCalls,
				ToolCalls: []llm.ToolCall{
					{ID: "call-1", Name: "bash", Arguments: json.RawMessage(`{"command":"echo hello"}`)},
				},
				CreatedAt: time.Now(),
			},
			{
				Content:      "Adjusted approach.",
				FinishReason: llm.FinishReasonStop,
				CreatedAt:    time.Now(),
			},
		},
	}
	client := llm.NewClient(llm.WithProvider("mock", adapter))
	profile := DefaultAnthropicProfile("test-model")
	env := &mockEnv{results: map[string]string{"bash": "hello"}}
	session := NewSession(client, profile, env, DefaultSessionConfig())

	// Add steering before submit
	session.Steer("Actually, do something different")

	err := session.Submit(context.Background(), "Do something")
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}
	_ = callCount
}

func TestSessionMaxTurns(t *testing.T) {
	adapter := &mockLLMAdapter{
		responses: []*llm.Response{
			{Content: "turn 1", FinishReason: llm.FinishReasonStop, CreatedAt: time.Now()},
		},
	}
	client := llm.NewClient(llm.WithProvider("mock", adapter))
	profile := DefaultAnthropicProfile("test-model")
	config := DefaultSessionConfig()
	config.MaxTurns = 1

	session := NewSession(client, profile, &mockEnv{results: map[string]string{}}, config)
	err := session.Submit(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}
}

func TestSessionClose(t *testing.T) {
	client := llm.NewClient(llm.WithProvider("mock", &mockLLMAdapter{}))
	profile := DefaultAnthropicProfile("test-model")
	session := NewSession(client, profile, &mockEnv{results: map[string]string{}}, DefaultSessionConfig())

	session.Close()
	if session.State != StateClosed {
		t.Errorf("expected CLOSED, got %s", session.State)
	}
}

func TestSessionEventEmission(t *testing.T) {
	adapter := &mockLLMAdapter{
		responses: []*llm.Response{
			{Content: "Done!", FinishReason: llm.FinishReasonStop, CreatedAt: time.Now()},
		},
	}
	client := llm.NewClient(llm.WithProvider("mock", adapter))
	profile := DefaultAnthropicProfile("test-model")
	session := NewSession(client, profile, &mockEnv{results: map[string]string{}}, DefaultSessionConfig())

	var events []Event
	session.EventEmitter.On(func(e Event) {
		events = append(events, e)
	})

	session.Submit(context.Background(), "Hello")

	if len(events) == 0 {
		t.Error("expected events to be emitted")
	}

	// Should have at least session_started and turn_started/turn_completed
	foundStart := false
	for _, e := range events {
		if e.Type == EventSessionStarted {
			foundStart = true
		}
	}
	if !foundStart {
		t.Error("expected session_started event")
	}
}

func TestProviderProfiles(t *testing.T) {
	tests := []struct {
		name    string
		profile *ProviderProfile
	}{
		{"anthropic", DefaultAnthropicProfile("claude-sonnet-4-5-20250929")},
		{"openai", DefaultOpenAIProfile("gpt-5.2")},
		{"gemini", DefaultGeminiProfile("gemini-2.5-pro")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.profile.Provider != tt.name {
				t.Errorf("expected provider %q, got %q", tt.name, tt.profile.Provider)
			}
			if len(tt.profile.Tools) == 0 {
				t.Error("expected tools in profile")
			}
			if tt.profile.SystemPrompt == "" {
				t.Error("expected system prompt")
			}
		})
	}
}

func TestOpenAIProfileHasApplyPatch(t *testing.T) {
	profile := DefaultOpenAIProfile("gpt-5.2")
	found := false
	for _, tool := range profile.Tools {
		if tool.Name == "apply_patch" {
			found = true
		}
	}
	if !found {
		t.Error("OpenAI profile should include apply_patch tool")
	}
}

func TestRegisterCustomTool(t *testing.T) {
	profile := DefaultAnthropicProfile("test-model")
	originalCount := len(profile.Tools)

	customTool := llm.Tool{
		Name:        "custom_tool",
		Description: "A custom tool",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}
	profile.RegisterTool(customTool)

	if len(profile.Tools) != originalCount+1 {
		t.Errorf("expected %d tools after registration, got %d", originalCount+1, len(profile.Tools))
	}

	// Register again with same name should override
	profile.RegisterTool(customTool)
	if len(profile.Tools) != originalCount+1 {
		t.Error("re-registering same tool name should override, not add")
	}
}

func TestDefaultSessionConfig(t *testing.T) {
	config := DefaultSessionConfig()
	if config.DefaultCommandTimeoutMs != 10000 {
		t.Errorf("expected 10000ms default timeout, got %d", config.DefaultCommandTimeoutMs)
	}
	if !config.EnableLoopDetection {
		t.Error("expected loop detection enabled by default")
	}
	if config.MaxSubagentDepth != 1 {
		t.Errorf("expected max subagent depth 1, got %d", config.MaxSubagentDepth)
	}
}

func TestBuildSystemPrompt(t *testing.T) {
	profile := DefaultAnthropicProfile("test-model")
	prompt := BuildSystemPrompt(profile, "/tmp/test", "Always use Go")

	if prompt == "" {
		t.Error("expected non-empty system prompt")
	}
	// Should contain environment context
	if !containsStr(prompt, "Platform:") {
		t.Error("expected platform info in system prompt")
	}
	// Should contain user instructions
	if !containsStr(prompt, "Always use Go") {
		t.Error("expected user instructions in system prompt")
	}
}

func containsStr(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && containsSubstr(s, sub))
}

func containsSubstr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
