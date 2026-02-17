package llm

import (
	"encoding/json"
	"testing"
)

func TestUsageAdd(t *testing.T) {
	a := Usage{InputTokens: 100, OutputTokens: 50, TotalTokens: 150, ReasoningTokens: 10}
	b := Usage{InputTokens: 200, OutputTokens: 75, TotalTokens: 275, ReasoningTokens: 20, CacheReadTokens: 50}

	result := a.Add(b)

	if result.InputTokens != 300 {
		t.Errorf("expected 300 input tokens, got %d", result.InputTokens)
	}
	if result.OutputTokens != 125 {
		t.Errorf("expected 125 output tokens, got %d", result.OutputTokens)
	}
	if result.TotalTokens != 425 {
		t.Errorf("expected 425 total tokens, got %d", result.TotalTokens)
	}
	if result.ReasoningTokens != 30 {
		t.Errorf("expected 30 reasoning tokens, got %d", result.ReasoningTokens)
	}
	if result.CacheReadTokens != 50 {
		t.Errorf("expected 50 cache read tokens, got %d", result.CacheReadTokens)
	}
}

func TestStreamAccumulator(t *testing.T) {
	acc := &StreamAccumulator{}

	acc.Process(StreamEvent{Type: StreamEventStart})
	acc.Process(StreamEvent{Type: StreamEventDelta, Delta: "Hello"})
	acc.Process(StreamEvent{Type: StreamEventDelta, Delta: ", world!"})
	acc.Process(StreamEvent{
		Type:         StreamEventEnd,
		FinishReason: FinishReasonStop,
		Usage:        &Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	})

	resp := acc.Response()
	if resp.Content != "Hello, world!" {
		t.Errorf("expected 'Hello, world!', got %q", resp.Content)
	}
	if resp.FinishReason != FinishReasonStop {
		t.Errorf("expected stop, got %s", resp.FinishReason)
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("expected 10 input tokens, got %d", resp.Usage.InputTokens)
	}
}

func TestStreamAccumulatorWithToolCalls(t *testing.T) {
	acc := &StreamAccumulator{}

	acc.Process(StreamEvent{
		Type: StreamEventToolCallStart,
		ToolCall: &ToolCall{
			ID:   "call-1",
			Name: "read_file",
		},
	})
	acc.Process(StreamEvent{
		Type:  StreamEventToolCallDelta,
		Delta: `{"path":"test.txt"}`,
	})
	acc.Process(StreamEvent{Type: StreamEventEnd, FinishReason: FinishReasonToolCalls})

	resp := acc.Response()
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].Name != "read_file" {
		t.Errorf("expected tool name 'read_file', got %q", resp.ToolCalls[0].Name)
	}
}

func TestStreamAccumulatorWithReasoning(t *testing.T) {
	acc := &StreamAccumulator{}

	acc.Process(StreamEvent{Type: StreamEventReasoningDelta, Delta: "Let me think..."})
	acc.Process(StreamEvent{Type: StreamEventReasoningDelta, Delta: " The answer is 42."})
	acc.Process(StreamEvent{Type: StreamEventDelta, Delta: "The answer is 42."})
	acc.Process(StreamEvent{Type: StreamEventEnd, FinishReason: FinishReasonStop})

	resp := acc.Response()
	if resp.Reasoning != "Let me think... The answer is 42." {
		t.Errorf("expected reasoning text, got %q", resp.Reasoning)
	}
	if resp.Content != "The answer is 42." {
		t.Errorf("expected content, got %q", resp.Content)
	}
}

func TestRoleConstants(t *testing.T) {
	roles := []Role{RoleSystem, RoleUser, RoleAssistant, RoleTool, RoleDeveloper}
	expected := []string{"system", "user", "assistant", "tool", "developer"}

	for i, role := range roles {
		if string(role) != expected[i] {
			t.Errorf("expected role %q, got %q", expected[i], role)
		}
	}
}

func TestFinishReasonConstants(t *testing.T) {
	reasons := []FinishReason{FinishReasonStop, FinishReasonLength, FinishReasonToolCalls, FinishReasonError}
	expected := []string{"stop", "length", "tool_calls", "error"}

	for i, reason := range reasons {
		if string(reason) != expected[i] {
			t.Errorf("expected %q, got %q", expected[i], reason)
		}
	}
}

func TestToolChoiceConstants(t *testing.T) {
	choices := []ToolChoice{ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired}
	expected := []string{"auto", "none", "required"}

	for i, choice := range choices {
		if string(choice) != expected[i] {
			t.Errorf("expected %q, got %q", expected[i], choice)
		}
	}
}

func TestRequestSerialization(t *testing.T) {
	req := &Request{
		Model:    "test-model",
		Provider: "test",
		Messages: []Message{
			{Role: RoleUser, Content: "Hello"},
		},
		MaxTokens: 100,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var decoded Request
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if decoded.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", decoded.Model)
	}
	if len(decoded.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(decoded.Messages))
	}
}

func TestResponseFormat(t *testing.T) {
	rf := &ResponseFormat{
		Type:       "json_schema",
		JSONSchema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
	}

	data, err := json.Marshal(rf)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var decoded ResponseFormat
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if decoded.Type != "json_schema" {
		t.Errorf("expected type json_schema, got %q", decoded.Type)
	}
}
