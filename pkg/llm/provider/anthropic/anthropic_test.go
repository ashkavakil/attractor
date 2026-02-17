package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ashka-vakil/attractor/pkg/llm"
)

// ---------------------------------------------------------------------------
// TestBuildRequest
// ---------------------------------------------------------------------------

func TestBuildRequest(t *testing.T) {
	adapter := NewAdapter(WithAPIKey("test-key"))

	t.Run("basic user message", func(t *testing.T) {
		req := &llm.Request{
			Model: "claude-sonnet-4-20250514",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hello"},
			},
		}
		mr := adapter.buildRequest(req)

		if mr.Model != "claude-sonnet-4-20250514" {
			t.Errorf("expected model claude-sonnet-4-20250514, got %s", mr.Model)
		}
		if mr.MaxTokens != 4096 {
			t.Errorf("expected default max_tokens 4096, got %d", mr.MaxTokens)
		}
		if len(mr.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(mr.Messages))
		}
		if mr.Messages[0].Role != "user" {
			t.Errorf("expected role user, got %s", mr.Messages[0].Role)
		}
		if mr.Messages[0].Content != "Hello" {
			t.Errorf("expected content Hello, got %v", mr.Messages[0].Content)
		}
	})

	t.Run("system prompt", func(t *testing.T) {
		req := &llm.Request{
			Model:        "claude-sonnet-4-20250514",
			SystemPrompt: "You are a helpful assistant.",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hi"},
			},
		}
		mr := adapter.buildRequest(req)

		if mr.System != "You are a helpful assistant." {
			t.Errorf("expected system prompt, got %s", mr.System)
		}
	})

	t.Run("system role message is skipped", func(t *testing.T) {
		req := &llm.Request{
			Model: "claude-sonnet-4-20250514",
			Messages: []llm.Message{
				{Role: llm.RoleSystem, Content: "System msg"},
				{Role: llm.RoleUser, Content: "Hi"},
			},
		}
		mr := adapter.buildRequest(req)

		if len(mr.Messages) != 1 {
			t.Fatalf("expected 1 message (system skipped), got %d", len(mr.Messages))
		}
		if mr.Messages[0].Role != "user" {
			t.Errorf("expected user role, got %s", mr.Messages[0].Role)
		}
	})

	t.Run("assistant with tool calls", func(t *testing.T) {
		req := &llm.Request{
			Model: "claude-sonnet-4-20250514",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Search for Go"},
				{
					Role:    llm.RoleAssistant,
					Content: "Let me search.",
					ToolCalls: []llm.ToolCall{
						{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"Go"}`)},
					},
				},
			},
		}
		mr := adapter.buildRequest(req)

		if len(mr.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(mr.Messages))
		}

		blocks, ok := mr.Messages[1].Content.([]contentBlock)
		if !ok {
			t.Fatalf("expected assistant content to be []contentBlock, got %T", mr.Messages[1].Content)
		}
		if len(blocks) != 2 {
			t.Fatalf("expected 2 content blocks (text + tool_use), got %d", len(blocks))
		}
		if blocks[0].Type != "text" || blocks[0].Text != "Let me search." {
			t.Errorf("unexpected text block: %+v", blocks[0])
		}
		if blocks[1].Type != "tool_use" || blocks[1].Name != "search" || blocks[1].ID != "tc1" {
			t.Errorf("unexpected tool_use block: %+v", blocks[1])
		}
	})

	t.Run("tool result message", func(t *testing.T) {
		req := &llm.Request{
			Model: "claude-sonnet-4-20250514",
			Messages: []llm.Message{
				{Role: llm.RoleTool, ToolCallID: "tc1", Content: "result data"},
			},
		}
		mr := adapter.buildRequest(req)

		if len(mr.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(mr.Messages))
		}
		if mr.Messages[0].Role != "user" {
			t.Errorf("expected tool result mapped to user role, got %s", mr.Messages[0].Role)
		}
		blocks, ok := mr.Messages[0].Content.([]contentBlock)
		if !ok {
			t.Fatalf("expected []contentBlock, got %T", mr.Messages[0].Content)
		}
		if blocks[0].Type != "tool_result" {
			t.Errorf("expected tool_result type, got %s", blocks[0].Type)
		}
		if blocks[0].ToolUseID != "tc1" {
			t.Errorf("expected tool_use_id tc1, got %s", blocks[0].ToolUseID)
		}
	})

	t.Run("tools", func(t *testing.T) {
		req := &llm.Request{
			Model: "claude-sonnet-4-20250514",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hi"},
			},
			Tools: []llm.Tool{
				{
					Name:        "get_weather",
					Description: "Get the weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
				},
			},
		}
		mr := adapter.buildRequest(req)

		if len(mr.Tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(mr.Tools))
		}
		if mr.Tools[0].Name != "get_weather" {
			t.Errorf("expected tool name get_weather, got %s", mr.Tools[0].Name)
		}
		if mr.Tools[0].Description != "Get the weather" {
			t.Errorf("expected tool description, got %s", mr.Tools[0].Description)
		}
	})

	t.Run("tool_choice auto", func(t *testing.T) {
		req := &llm.Request{
			Model:      "claude-sonnet-4-20250514",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Tools:      []llm.Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{}`)}},
			ToolChoice: llm.ToolChoiceAuto,
		}
		mr := adapter.buildRequest(req)

		tc, ok := mr.ToolChoice.(map[string]string)
		if !ok {
			t.Fatalf("expected map[string]string for auto, got %T", mr.ToolChoice)
		}
		if tc["type"] != "auto" {
			t.Errorf("expected type auto, got %s", tc["type"])
		}
	})

	t.Run("tool_choice none removes tools", func(t *testing.T) {
		req := &llm.Request{
			Model:      "claude-sonnet-4-20250514",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Tools:      []llm.Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{}`)}},
			ToolChoice: llm.ToolChoiceNone,
		}
		mr := adapter.buildRequest(req)

		if mr.Tools != nil {
			t.Errorf("expected tools to be nil for ToolChoiceNone, got %v", mr.Tools)
		}
	})

	t.Run("tool_choice required maps to any", func(t *testing.T) {
		req := &llm.Request{
			Model:      "claude-sonnet-4-20250514",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Tools:      []llm.Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{}`)}},
			ToolChoice: llm.ToolChoiceRequired,
		}
		mr := adapter.buildRequest(req)

		tc, ok := mr.ToolChoice.(map[string]string)
		if !ok {
			t.Fatalf("expected map[string]string for required, got %T", mr.ToolChoice)
		}
		if tc["type"] != "any" {
			t.Errorf("expected type any, got %s", tc["type"])
		}
	})

	t.Run("tool_choice function", func(t *testing.T) {
		req := &llm.Request{
			Model:      "claude-sonnet-4-20250514",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Tools:      []llm.Tool{{Name: "search", Description: "d", Parameters: json.RawMessage(`{}`)}},
			ToolChoice: llm.ToolChoiceFunction{Name: "search"},
		}
		mr := adapter.buildRequest(req)

		tc, ok := mr.ToolChoice.(map[string]interface{})
		if !ok {
			t.Fatalf("expected map[string]interface{} for function, got %T", mr.ToolChoice)
		}
		if tc["type"] != "tool" {
			t.Errorf("expected type tool, got %v", tc["type"])
		}
		if tc["name"] != "search" {
			t.Errorf("expected name search, got %v", tc["name"])
		}
	})

	t.Run("custom max_tokens", func(t *testing.T) {
		req := &llm.Request{
			Model:     "claude-sonnet-4-20250514",
			Messages:  []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			MaxTokens: 8192,
		}
		mr := adapter.buildRequest(req)

		if mr.MaxTokens != 8192 {
			t.Errorf("expected max_tokens 8192, got %d", mr.MaxTokens)
		}
	})

	t.Run("temperature and top_p", func(t *testing.T) {
		temp := 0.7
		topP := 0.9
		req := &llm.Request{
			Model:       "claude-sonnet-4-20250514",
			Messages:    []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Temperature: &temp,
			TopP:        &topP,
		}
		mr := adapter.buildRequest(req)

		if mr.Temperature == nil || *mr.Temperature != 0.7 {
			t.Errorf("expected temperature 0.7, got %v", mr.Temperature)
		}
		if mr.TopP == nil || *mr.TopP != 0.9 {
			t.Errorf("expected top_p 0.9, got %v", mr.TopP)
		}
	})
}

// ---------------------------------------------------------------------------
// TestComplete
// ---------------------------------------------------------------------------

func TestComplete(t *testing.T) {
	t.Run("text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/messages" {
				t.Errorf("unexpected path: %s", r.URL.Path)
			}
			if r.Header.Get("x-api-key") != "test-key" {
				t.Errorf("expected x-api-key test-key, got %s", r.Header.Get("x-api-key"))
			}
			if r.Header.Get("anthropic-version") != "2023-06-01" {
				t.Errorf("expected anthropic-version 2023-06-01, got %s", r.Header.Get("anthropic-version"))
			}

			// Verify the request body
			var reqBody messagesRequest
			if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
				t.Fatalf("failed to decode request: %v", err)
			}
			if reqBody.Stream {
				t.Error("expected stream=false for Complete")
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(messagesResponse{
				ID:    "msg_123",
				Type:  "message",
				Role:  "assistant",
				Model: "claude-sonnet-4-20250514",
				Content: []contentBlock{
					{Type: "text", Text: "Hello! How can I help?"},
				},
				StopReason: "end_turn",
				Usage:      anthropicUsage{InputTokens: 10, OutputTokens: 20},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		resp, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "claude-sonnet-4-20250514",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hello"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.ID != "msg_123" {
			t.Errorf("expected ID msg_123, got %s", resp.ID)
		}
		if resp.Model != "claude-sonnet-4-20250514" {
			t.Errorf("expected model claude-sonnet-4-20250514, got %s", resp.Model)
		}
		if resp.Content != "Hello! How can I help?" {
			t.Errorf("expected content 'Hello! How can I help?', got %s", resp.Content)
		}
		if resp.FinishReason != llm.FinishReasonStop {
			t.Errorf("expected finish_reason stop, got %s", resp.FinishReason)
		}
		if resp.Usage.InputTokens != 10 {
			t.Errorf("expected input_tokens 10, got %d", resp.Usage.InputTokens)
		}
		if resp.Usage.OutputTokens != 20 {
			t.Errorf("expected output_tokens 20, got %d", resp.Usage.OutputTokens)
		}
		if resp.Usage.TotalTokens != 30 {
			t.Errorf("expected total_tokens 30, got %d", resp.Usage.TotalTokens)
		}
	})

	t.Run("tool_use response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(messagesResponse{
				ID:    "msg_456",
				Model: "claude-sonnet-4-20250514",
				Content: []contentBlock{
					{Type: "text", Text: "Let me search that."},
					{Type: "tool_use", ID: "call_1", Name: "search", Input: json.RawMessage(`{"query":"Go lang"}`)},
				},
				StopReason: "tool_use",
				Usage:      anthropicUsage{InputTokens: 15, OutputTokens: 25},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		resp, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "claude-sonnet-4-20250514",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Search Go"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content != "Let me search that." {
			t.Errorf("expected text content, got %s", resp.Content)
		}
		if resp.FinishReason != llm.FinishReasonToolCalls {
			t.Errorf("expected finish_reason tool_calls, got %s", resp.FinishReason)
		}
		if len(resp.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
		}
		if resp.ToolCalls[0].ID != "call_1" {
			t.Errorf("expected tool call ID call_1, got %s", resp.ToolCalls[0].ID)
		}
		if resp.ToolCalls[0].Name != "search" {
			t.Errorf("expected tool call name search, got %s", resp.ToolCalls[0].Name)
		}
	})

	t.Run("thinking block response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(messagesResponse{
				ID:    "msg_789",
				Model: "claude-sonnet-4-20250514",
				Content: []contentBlock{
					{Type: "thinking", Thinking: "Let me think about this..."},
					{Type: "text", Text: "The answer is 42."},
				},
				StopReason: "end_turn",
				Usage:      anthropicUsage{InputTokens: 5, OutputTokens: 50},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		resp, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "claude-sonnet-4-20250514",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "What is the answer?"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Reasoning != "Let me think about this..." {
			t.Errorf("expected reasoning, got %s", resp.Reasoning)
		}
		if resp.Content != "The answer is 42." {
			t.Errorf("expected content, got %s", resp.Content)
		}
	})
}

// ---------------------------------------------------------------------------
// TestCompleteError
// ---------------------------------------------------------------------------

func TestCompleteError(t *testing.T) {
	tests := []struct {
		name           string
		statusCode     int
		body           string
		expectedType   llm.ErrorType
	}{
		{
			name:         "401 auth error",
			statusCode:   401,
			body:         `{"error":{"message":"Invalid API key"}}`,
			expectedType: llm.ErrorTypeAuth,
		},
		{
			name:         "403 auth error",
			statusCode:   403,
			body:         `{"error":{"message":"Forbidden"}}`,
			expectedType: llm.ErrorTypeAuth,
		},
		{
			name:         "429 rate limit",
			statusCode:   429,
			body:         `{"error":{"message":"Rate limit exceeded"}}`,
			expectedType: llm.ErrorTypeRateLimit,
		},
		{
			name:         "400 bad request",
			statusCode:   400,
			body:         `{"error":{"message":"Invalid request"}}`,
			expectedType: llm.ErrorTypeBadRequest,
		},
		{
			name:         "500 server error",
			statusCode:   500,
			body:         `{"error":{"message":"Internal server error"}}`,
			expectedType: llm.ErrorTypeServer,
		},
		{
			name:         "529 overloaded",
			statusCode:   529,
			body:         `{"error":{"message":"Overloaded"}}`,
			expectedType: llm.ErrorTypeServer,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				w.Write([]byte(tt.body))
			}))
			defer server.Close()

			adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
			_, err := adapter.Complete(context.Background(), &llm.Request{
				Model:    "claude-sonnet-4-20250514",
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			})

			if err == nil {
				t.Fatal("expected error, got nil")
			}
			llmErr, ok := err.(*llm.LLMError)
			if !ok {
				t.Fatalf("expected *llm.LLMError, got %T", err)
			}
			if llmErr.Type != tt.expectedType {
				t.Errorf("expected error type %s, got %s", tt.expectedType, llmErr.Type)
			}
			if llmErr.StatusCode != tt.statusCode {
				t.Errorf("expected status code %d, got %d", tt.statusCode, llmErr.StatusCode)
			}
			if llmErr.Provider != "anthropic" {
				t.Errorf("expected provider anthropic, got %s", llmErr.Provider)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestStream
// ---------------------------------------------------------------------------

func TestStream(t *testing.T) {
	t.Run("text streaming", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify request has stream=true
			var reqBody messagesRequest
			json.NewDecoder(r.Body).Decode(&reqBody)
			if !reqBody.Stream {
				t.Error("expected stream=true")
			}

			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)

			events := []string{
				`data: {"type":"message_start","message":{"id":"msg_stream1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"usage":{"input_tokens":25}}}`,
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
				`data: {"type":"content_block_stop","index":0}`,
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}`,
				`data: {"type":"message_stop"}`,
			}

			for _, event := range events {
				fmt.Fprintf(w, "%s\n\n", event)
				flusher.Flush()
			}
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "claude-sonnet-4-20250514",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var events []llm.StreamEvent
		for ev := range ch {
			events = append(events, ev)
		}

		// Expect: start, text delta "Hello", text delta " world", tool_call_end (from content_block_stop), end
		if len(events) < 4 {
			t.Fatalf("expected at least 4 events, got %d: %+v", len(events), events)
		}

		// First event should be start
		if events[0].Type != llm.StreamEventStart {
			t.Errorf("expected first event type start, got %s", events[0].Type)
		}

		// Collect text deltas
		var text string
		for _, ev := range events {
			if ev.Type == llm.StreamEventDelta {
				text += ev.Delta
			}
		}
		if text != "Hello world" {
			t.Errorf("expected accumulated text 'Hello world', got %s", text)
		}

		// Last event should be end with correct finish reason and usage
		lastEvent := events[len(events)-1]
		if lastEvent.Type != llm.StreamEventEnd {
			t.Errorf("expected last event type end, got %s", lastEvent.Type)
		}
		if lastEvent.FinishReason != llm.FinishReasonStop {
			t.Errorf("expected finish_reason stop, got %s", lastEvent.FinishReason)
		}
		if lastEvent.Usage == nil {
			t.Fatal("expected usage in end event")
		}
		if lastEvent.Usage.InputTokens != 25 {
			t.Errorf("expected input_tokens 25, got %d", lastEvent.Usage.InputTokens)
		}
		if lastEvent.Usage.OutputTokens != 15 {
			t.Errorf("expected output_tokens 15, got %d", lastEvent.Usage.OutputTokens)
		}
		if lastEvent.Usage.TotalTokens != 40 {
			t.Errorf("expected total_tokens 40, got %d", lastEvent.Usage.TotalTokens)
		}
	})

	t.Run("tool_use streaming", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)

			events := []string{
				`data: {"type":"message_start","message":{"id":"msg_stream2","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"usage":{"input_tokens":30}}}`,
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather"}}`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}`,
				`data: {"type":"content_block_stop","index":0}`,
				`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}`,
				`data: {"type":"message_stop"}`,
			}

			for _, event := range events {
				fmt.Fprintf(w, "%s\n\n", event)
				flusher.Flush()
			}
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "claude-sonnet-4-20250514",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Weather?"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var events []llm.StreamEvent
		for ev := range ch {
			events = append(events, ev)
		}

		// Check tool call start
		var foundToolStart bool
		var toolArgs string
		for _, ev := range events {
			if ev.Type == llm.StreamEventToolCallStart {
				foundToolStart = true
				if ev.ToolCall == nil {
					t.Fatal("expected tool call in tool_call_start event")
				}
				if ev.ToolCall.ID != "toolu_01" {
					t.Errorf("expected tool call ID toolu_01, got %s", ev.ToolCall.ID)
				}
				if ev.ToolCall.Name != "get_weather" {
					t.Errorf("expected tool call name get_weather, got %s", ev.ToolCall.Name)
				}
			}
			if ev.Type == llm.StreamEventToolCallDelta {
				toolArgs += ev.Delta
			}
		}

		if !foundToolStart {
			t.Error("expected tool_call_start event")
		}
		if toolArgs != `{"city":"SF"}` {
			t.Errorf("expected tool args {\"city\":\"SF\"}, got %s", toolArgs)
		}

		// Check end event with tool_use stop reason
		lastEvent := events[len(events)-1]
		if lastEvent.Type != llm.StreamEventEnd {
			t.Errorf("expected last event type end, got %s", lastEvent.Type)
		}
		if lastEvent.FinishReason != llm.FinishReasonToolCalls {
			t.Errorf("expected finish_reason tool_calls, got %s", lastEvent.FinishReason)
		}
	})

	t.Run("thinking streaming", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)

			events := []string{
				`data: {"type":"message_start","message":{"id":"msg_stream3","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"usage":{"input_tokens":10}}}`,
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Hmm let me think"}}`,
				`data: {"type":"content_block_stop","index":0}`,
				`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
				`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}`,
				`data: {"type":"content_block_stop","index":1}`,
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":30}}`,
				`data: {"type":"message_stop"}`,
			}

			for _, event := range events {
				fmt.Fprintf(w, "%s\n\n", event)
				flusher.Flush()
			}
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "claude-sonnet-4-20250514",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Think about this"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var reasoning string
		var text string
		for ev := range ch {
			if ev.Type == llm.StreamEventReasoningDelta {
				reasoning += ev.Delta
			}
			if ev.Type == llm.StreamEventDelta {
				text += ev.Delta
			}
		}

		if reasoning != "Hmm let me think" {
			t.Errorf("expected reasoning 'Hmm let me think', got %s", reasoning)
		}
		if text != "Answer" {
			t.Errorf("expected text 'Answer', got %s", text)
		}
	})
}

// ---------------------------------------------------------------------------
// TestConvertStopReason
// ---------------------------------------------------------------------------

func TestConvertStopReason(t *testing.T) {
	tests := []struct {
		input    string
		expected llm.FinishReason
	}{
		{"end_turn", llm.FinishReasonStop},
		{"max_tokens", llm.FinishReasonLength},
		{"tool_use", llm.FinishReasonToolCalls},
		{"unknown_reason", llm.FinishReasonStop},
		{"", llm.FinishReasonStop},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("input=%q", tt.input), func(t *testing.T) {
			result := convertStopReason(tt.input)
			if result != tt.expected {
				t.Errorf("convertStopReason(%q) = %s, want %s", tt.input, result, tt.expected)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestName
// ---------------------------------------------------------------------------

func TestName(t *testing.T) {
	adapter := NewAdapter(WithAPIKey("test-key"))
	if adapter.Name() != "anthropic" {
		t.Errorf("expected name anthropic, got %s", adapter.Name())
	}
}

// ---------------------------------------------------------------------------
// TestClose
// ---------------------------------------------------------------------------

func TestClose(t *testing.T) {
	adapter := NewAdapter(WithAPIKey("test-key"))
	if err := adapter.Close(); err != nil {
		t.Errorf("expected nil error from Close, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// TestStreamError
// ---------------------------------------------------------------------------

func TestStreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"Rate limited"}}`))
	}))
	defer server.Close()

	adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := adapter.Stream(context.Background(), &llm.Request{
		Model:    "claude-sonnet-4-20250514",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
	})

	if err == nil {
		t.Fatal("expected error, got nil")
	}
	llmErr, ok := err.(*llm.LLMError)
	if !ok {
		t.Fatalf("expected *llm.LLMError, got %T", err)
	}
	if llmErr.Type != llm.ErrorTypeRateLimit {
		t.Errorf("expected rate_limit error, got %s", llmErr.Type)
	}
}

// ---------------------------------------------------------------------------
// TestDoRequestSendsHeaders
// ---------------------------------------------------------------------------

func TestDoRequestSendsHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}
		if r.Header.Get("x-api-key") != "my-key" {
			t.Errorf("expected x-api-key my-key, got %s", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("expected anthropic-version, got %s", r.Header.Get("anthropic-version"))
		}

		// Read the request body to verify it is valid JSON
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("failed to read body: %v", err)
		}
		if !json.Valid(body) {
			t.Error("request body is not valid JSON")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(messagesResponse{
			ID:         "msg_hdr",
			Model:      "claude-sonnet-4-20250514",
			Content:    []contentBlock{{Type: "text", Text: "ok"}},
			StopReason: "end_turn",
			Usage:      anthropicUsage{InputTokens: 1, OutputTokens: 1},
		})
	}))
	defer server.Close()

	adapter := NewAdapter(WithAPIKey("my-key"), WithBaseURL(server.URL))
	_, err := adapter.Complete(context.Background(), &llm.Request{
		Model:    "claude-sonnet-4-20250514",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
