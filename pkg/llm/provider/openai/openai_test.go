package openai

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
			Model: "gpt-4o",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hello"},
			},
		}
		cr := adapter.buildRequest(req)

		if cr.Model != "gpt-4o" {
			t.Errorf("expected model gpt-4o, got %s", cr.Model)
		}
		if len(cr.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(cr.Messages))
		}
		if cr.Messages[0].Role != "user" {
			t.Errorf("expected role user, got %s", cr.Messages[0].Role)
		}
		content, ok := cr.Messages[0].Content.(string)
		if !ok {
			t.Fatalf("expected string content, got %T", cr.Messages[0].Content)
		}
		if content != "Hello" {
			t.Errorf("expected content Hello, got %s", content)
		}
	})

	t.Run("system prompt as first message", func(t *testing.T) {
		req := &llm.Request{
			Model:        "gpt-4o",
			SystemPrompt: "You are helpful.",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hi"},
			},
		}
		cr := adapter.buildRequest(req)

		if len(cr.Messages) != 2 {
			t.Fatalf("expected 2 messages (system + user), got %d", len(cr.Messages))
		}
		if cr.Messages[0].Role != "system" {
			t.Errorf("expected first message role system, got %s", cr.Messages[0].Role)
		}
		sysContent, ok := cr.Messages[0].Content.(string)
		if !ok {
			t.Fatalf("expected string content for system, got %T", cr.Messages[0].Content)
		}
		if sysContent != "You are helpful." {
			t.Errorf("expected system content, got %s", sysContent)
		}
		if cr.Messages[1].Role != "user" {
			t.Errorf("expected second message role user, got %s", cr.Messages[1].Role)
		}
	})

	t.Run("assistant with tool calls", func(t *testing.T) {
		req := &llm.Request{
			Model: "gpt-4o",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Search"},
				{
					Role:    llm.RoleAssistant,
					Content: "Searching...",
					ToolCalls: []llm.ToolCall{
						{ID: "call_1", Name: "search", Arguments: json.RawMessage(`{"q":"test"}`)},
					},
				},
			},
		}
		cr := adapter.buildRequest(req)

		if len(cr.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(cr.Messages))
		}

		assistantMsg := cr.Messages[1]
		if len(assistantMsg.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(assistantMsg.ToolCalls))
		}
		if assistantMsg.ToolCalls[0].ID != "call_1" {
			t.Errorf("expected tool call ID call_1, got %s", assistantMsg.ToolCalls[0].ID)
		}
		if assistantMsg.ToolCalls[0].Type != "function" {
			t.Errorf("expected tool call type function, got %s", assistantMsg.ToolCalls[0].Type)
		}
		if assistantMsg.ToolCalls[0].Function.Name != "search" {
			t.Errorf("expected function name search, got %s", assistantMsg.ToolCalls[0].Function.Name)
		}
		if assistantMsg.ToolCalls[0].Function.Arguments != `{"q":"test"}` {
			t.Errorf("expected function arguments, got %s", assistantMsg.ToolCalls[0].Function.Arguments)
		}
	})

	t.Run("tool result message", func(t *testing.T) {
		req := &llm.Request{
			Model: "gpt-4o",
			Messages: []llm.Message{
				{Role: llm.RoleTool, ToolCallID: "call_1", Content: "result"},
			},
		}
		cr := adapter.buildRequest(req)

		if len(cr.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(cr.Messages))
		}
		if cr.Messages[0].Role != "tool" {
			t.Errorf("expected role tool, got %s", cr.Messages[0].Role)
		}
		if cr.Messages[0].ToolCallID != "call_1" {
			t.Errorf("expected tool_call_id call_1, got %s", cr.Messages[0].ToolCallID)
		}
	})

	t.Run("tools", func(t *testing.T) {
		req := &llm.Request{
			Model: "gpt-4o",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hi"},
			},
			Tools: []llm.Tool{
				{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  json.RawMessage(`{"type":"object"}`),
				},
			},
		}
		cr := adapter.buildRequest(req)

		if len(cr.Tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(cr.Tools))
		}
		if cr.Tools[0].Type != "function" {
			t.Errorf("expected tool type function, got %s", cr.Tools[0].Type)
		}
		if cr.Tools[0].Function.Name != "get_weather" {
			t.Errorf("expected tool name get_weather, got %s", cr.Tools[0].Function.Name)
		}
		if cr.Tools[0].Function.Description != "Get weather" {
			t.Errorf("expected tool description, got %s", cr.Tools[0].Function.Description)
		}
	})

	t.Run("tool_choice auto", func(t *testing.T) {
		req := &llm.Request{
			Model:      "gpt-4o",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Tools:      []llm.Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{}`)}},
			ToolChoice: llm.ToolChoiceAuto,
		}
		cr := adapter.buildRequest(req)

		tc, ok := cr.ToolChoice.(string)
		if !ok {
			t.Fatalf("expected string tool_choice for auto, got %T", cr.ToolChoice)
		}
		if tc != "auto" {
			t.Errorf("expected auto, got %s", tc)
		}
	})

	t.Run("tool_choice none", func(t *testing.T) {
		req := &llm.Request{
			Model:      "gpt-4o",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Tools:      []llm.Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{}`)}},
			ToolChoice: llm.ToolChoiceNone,
		}
		cr := adapter.buildRequest(req)

		tc, ok := cr.ToolChoice.(string)
		if !ok {
			t.Fatalf("expected string tool_choice for none, got %T", cr.ToolChoice)
		}
		if tc != "none" {
			t.Errorf("expected none, got %s", tc)
		}
	})

	t.Run("tool_choice required", func(t *testing.T) {
		req := &llm.Request{
			Model:      "gpt-4o",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Tools:      []llm.Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{}`)}},
			ToolChoice: llm.ToolChoiceRequired,
		}
		cr := adapter.buildRequest(req)

		tc, ok := cr.ToolChoice.(string)
		if !ok {
			t.Fatalf("expected string tool_choice for required, got %T", cr.ToolChoice)
		}
		if tc != "required" {
			t.Errorf("expected required, got %s", tc)
		}
	})

	t.Run("tool_choice function", func(t *testing.T) {
		req := &llm.Request{
			Model:      "gpt-4o",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Tools:      []llm.Tool{{Name: "search", Description: "d", Parameters: json.RawMessage(`{}`)}},
			ToolChoice: llm.ToolChoiceFunction{Name: "search"},
		}
		cr := adapter.buildRequest(req)

		tc, ok := cr.ToolChoice.(map[string]interface{})
		if !ok {
			t.Fatalf("expected map[string]interface{} for function choice, got %T", cr.ToolChoice)
		}
		if tc["type"] != "function" {
			t.Errorf("expected type function, got %v", tc["type"])
		}
		fnMap, ok := tc["function"].(map[string]string)
		if !ok {
			t.Fatalf("expected function to be map[string]string, got %T", tc["function"])
		}
		if fnMap["name"] != "search" {
			t.Errorf("expected function name search, got %s", fnMap["name"])
		}
	})

	t.Run("reasoning_effort", func(t *testing.T) {
		req := &llm.Request{
			Model:           "o1",
			Messages:        []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			ReasoningEffort: "high",
		}
		cr := adapter.buildRequest(req)

		if cr.ReasoningEffort != "high" {
			t.Errorf("expected reasoning_effort high, got %s", cr.ReasoningEffort)
		}
	})

	t.Run("response_format", func(t *testing.T) {
		req := &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			ResponseFormat: &llm.ResponseFormat{
				Type: "json_object",
			},
		}
		cr := adapter.buildRequest(req)

		if cr.ResponseFormat == nil {
			t.Fatal("expected response_format to be set")
		}
		if cr.ResponseFormat.Type != "json_object" {
			t.Errorf("expected response_format type json_object, got %s", cr.ResponseFormat.Type)
		}
	})

	t.Run("response_format with json_schema", func(t *testing.T) {
		schema := json.RawMessage(`{"name":"test","schema":{"type":"object"}}`)
		req := &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			ResponseFormat: &llm.ResponseFormat{
				Type:       "json_schema",
				JSONSchema: schema,
			},
		}
		cr := adapter.buildRequest(req)

		if cr.ResponseFormat == nil {
			t.Fatal("expected response_format to be set")
		}
		if cr.ResponseFormat.Type != "json_schema" {
			t.Errorf("expected response_format type json_schema, got %s", cr.ResponseFormat.Type)
		}
		if string(cr.ResponseFormat.JSONSchema) != string(schema) {
			t.Errorf("expected json_schema to match, got %s", string(cr.ResponseFormat.JSONSchema))
		}
	})

	t.Run("temperature and top_p and stop", func(t *testing.T) {
		temp := 0.5
		topP := 0.8
		req := &llm.Request{
			Model:         "gpt-4o",
			Messages:      []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			Temperature:   &temp,
			TopP:          &topP,
			StopSequences: []string{"END"},
		}
		cr := adapter.buildRequest(req)

		if cr.Temperature == nil || *cr.Temperature != 0.5 {
			t.Errorf("expected temperature 0.5, got %v", cr.Temperature)
		}
		if cr.TopP == nil || *cr.TopP != 0.8 {
			t.Errorf("expected top_p 0.8, got %v", cr.TopP)
		}
		if len(cr.Stop) != 1 || cr.Stop[0] != "END" {
			t.Errorf("expected stop [END], got %v", cr.Stop)
		}
	})

	t.Run("max_tokens", func(t *testing.T) {
		req := &llm.Request{
			Model:     "gpt-4o",
			Messages:  []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			MaxTokens: 2048,
		}
		cr := adapter.buildRequest(req)

		if cr.MaxTokens != 2048 {
			t.Errorf("expected max_tokens 2048, got %d", cr.MaxTokens)
		}
	})
}

// ---------------------------------------------------------------------------
// TestComplete
// ---------------------------------------------------------------------------

func TestComplete(t *testing.T) {
	t.Run("text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/chat/completions" {
				t.Errorf("unexpected path: %s", r.URL.Path)
			}
			if r.Header.Get("Authorization") != "Bearer test-key" {
				t.Errorf("expected Authorization Bearer test-key, got %s", r.Header.Get("Authorization"))
			}

			var reqBody chatRequest
			json.NewDecoder(r.Body).Decode(&reqBody)
			if reqBody.Stream {
				t.Error("expected stream=false for Complete")
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(chatResponse{
				ID:    "chatcmpl-123",
				Model: "gpt-4o",
				Choices: []chatChoice{
					{
						Index: 0,
						Message: chatMessage{
							Role:    "assistant",
							Content: "Hello! How can I help?",
						},
						FinishReason: "stop",
					},
				},
				Usage:   chatUsage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
				Created: 1700000000,
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		resp, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hello"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.ID != "chatcmpl-123" {
			t.Errorf("expected ID chatcmpl-123, got %s", resp.ID)
		}
		if resp.Model != "gpt-4o" {
			t.Errorf("expected model gpt-4o, got %s", resp.Model)
		}
		if resp.Content != "Hello! How can I help?" {
			t.Errorf("expected content, got %s", resp.Content)
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

	t.Run("tool_calls response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(chatResponse{
				ID:    "chatcmpl-456",
				Model: "gpt-4o",
				Choices: []chatChoice{
					{
						Index: 0,
						Message: chatMessage{
							Role:    "assistant",
							Content: nil,
							ToolCalls: []chatToolCall{
								{
									ID:   "call_abc",
									Type: "function",
									Function: chatFunctionCall{
										Name:      "get_weather",
										Arguments: `{"city":"NYC"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
				Usage: chatUsage{PromptTokens: 15, CompletionTokens: 25, TotalTokens: 40},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		resp, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Weather?"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.FinishReason != llm.FinishReasonToolCalls {
			t.Errorf("expected finish_reason tool_calls, got %s", resp.FinishReason)
		}
		if len(resp.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
		}
		if resp.ToolCalls[0].ID != "call_abc" {
			t.Errorf("expected tool call ID call_abc, got %s", resp.ToolCalls[0].ID)
		}
		if resp.ToolCalls[0].Name != "get_weather" {
			t.Errorf("expected tool call name get_weather, got %s", resp.ToolCalls[0].Name)
		}
		if string(resp.ToolCalls[0].Arguments) != `{"city":"NYC"}` {
			t.Errorf("expected tool call arguments, got %s", string(resp.ToolCalls[0].Arguments))
		}
	})

	t.Run("finish_reason length", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(chatResponse{
				ID:    "chatcmpl-789",
				Model: "gpt-4o",
				Choices: []chatChoice{
					{
						Index: 0,
						Message: chatMessage{
							Role:    "assistant",
							Content: "Truncated output...",
						},
						FinishReason: "length",
					},
				},
				Usage: chatUsage{PromptTokens: 100, CompletionTokens: 4096, TotalTokens: 4196},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		resp, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Write long text"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.FinishReason != llm.FinishReasonLength {
			t.Errorf("expected finish_reason length, got %s", resp.FinishReason)
		}
	})
}

// ---------------------------------------------------------------------------
// TestCompleteError
// ---------------------------------------------------------------------------

func TestCompleteError(t *testing.T) {
	tests := []struct {
		name         string
		statusCode   int
		body         string
		expectedType llm.ErrorType
	}{
		{
			name:         "401 auth error",
			statusCode:   401,
			body:         `{"error":{"message":"Invalid API key","type":"invalid_request_error"}}`,
			expectedType: llm.ErrorTypeAuth,
		},
		{
			name:         "403 forbidden",
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
			body:         `{"error":{"message":"Invalid model"}}`,
			expectedType: llm.ErrorTypeBadRequest,
		},
		{
			name:         "422 unprocessable",
			statusCode:   422,
			body:         `{"error":{"message":"Unprocessable entity"}}`,
			expectedType: llm.ErrorTypeBadRequest,
		},
		{
			name:         "500 server error",
			statusCode:   500,
			body:         `{"error":{"message":"Internal server error"}}`,
			expectedType: llm.ErrorTypeServer,
		},
		{
			name:         "502 bad gateway",
			statusCode:   502,
			body:         `{"error":{"message":"Bad gateway"}}`,
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
				Model:    "gpt-4o",
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
			if llmErr.Provider != "openai" {
				t.Errorf("expected provider openai, got %s", llmErr.Provider)
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
			var reqBody chatRequest
			json.NewDecoder(r.Body).Decode(&reqBody)
			if !reqBody.Stream {
				t.Error("expected stream=true")
			}
			if reqBody.StreamOptions == nil || !reqBody.StreamOptions.IncludeUsage {
				t.Error("expected stream_options.include_usage=true")
			}

			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)

			chunks := []string{
				`data: {"id":"chatcmpl-stream1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: {"id":"chatcmpl-stream1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: {"id":"chatcmpl-stream1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: {"id":"chatcmpl-stream1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: {"id":"chatcmpl-stream1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				fmt.Fprintf(w, "%s\n\n", chunk)
				flusher.Flush()
			}
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var events []llm.StreamEvent
		for ev := range ch {
			events = append(events, ev)
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

		// Last event should be end with finish reason and usage
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
		if lastEvent.Usage.InputTokens != 10 {
			t.Errorf("expected input_tokens 10, got %d", lastEvent.Usage.InputTokens)
		}
		if lastEvent.Usage.OutputTokens != 5 {
			t.Errorf("expected output_tokens 5, got %d", lastEvent.Usage.OutputTokens)
		}
		if lastEvent.Usage.TotalTokens != 15 {
			t.Errorf("expected total_tokens 15, got %d", lastEvent.Usage.TotalTokens)
		}
	})

	t.Run("tool call streaming", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)

			chunks := []string{
				`data: {"id":"chatcmpl-stream2","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"id":"call_xyz","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: {"id":"chatcmpl-stream2","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: {"id":"chatcmpl-stream2","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"\"NYC\"}"}}]},"finish_reason":null}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: {"id":"chatcmpl-stream2","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: {"id":"chatcmpl-stream2","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				fmt.Fprintf(w, "%s\n\n", chunk)
				flusher.Flush()
			}
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Weather?"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var events []llm.StreamEvent
		for ev := range ch {
			events = append(events, ev)
		}

		// Verify tool call start
		var foundToolStart bool
		var toolArgs string
		for _, ev := range events {
			if ev.Type == llm.StreamEventToolCallStart {
				foundToolStart = true
				if ev.ToolCall == nil {
					t.Fatal("expected tool call in tool_call_start event")
				}
				if ev.ToolCall.ID != "call_xyz" {
					t.Errorf("expected tool call ID call_xyz, got %s", ev.ToolCall.ID)
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
		if toolArgs != `{"city":"NYC"}` {
			t.Errorf("expected tool args {\"city\":\"NYC\"}, got %s", toolArgs)
		}

		// Verify end event
		lastEvent := events[len(events)-1]
		if lastEvent.Type != llm.StreamEventEnd {
			t.Errorf("expected last event type end, got %s", lastEvent.Type)
		}
		if lastEvent.FinishReason != llm.FinishReasonToolCalls {
			t.Errorf("expected finish_reason tool_calls, got %s", lastEvent.FinishReason)
		}
		if lastEvent.Usage == nil {
			t.Fatal("expected usage in end event")
		}
		if lastEvent.Usage.InputTokens != 20 {
			t.Errorf("expected input_tokens 20, got %d", lastEvent.Usage.InputTokens)
		}
		if lastEvent.Usage.OutputTokens != 10 {
			t.Errorf("expected output_tokens 10, got %d", lastEvent.Usage.OutputTokens)
		}
	})

	t.Run("DONE sentinel handled", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)

			chunks := []string{
				`data: {"id":"chatcmpl-done","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: {"id":"chatcmpl-done","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
				`data: [DONE]`,
			}

			for _, chunk := range chunks {
				fmt.Fprintf(w, "%s\n\n", chunk)
				flusher.Flush()
			}
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var events []llm.StreamEvent
		for ev := range ch {
			events = append(events, ev)
		}

		// Verify the stream ends properly and channel closes
		if len(events) == 0 {
			t.Fatal("expected at least one event")
		}
		lastEvent := events[len(events)-1]
		if lastEvent.Type != llm.StreamEventEnd {
			t.Errorf("expected last event type end, got %s", lastEvent.Type)
		}
	})

	t.Run("stream_options include_usage is set", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			var reqBody chatRequest
			json.Unmarshal(body, &reqBody)

			if !reqBody.Stream {
				t.Error("expected stream=true")
			}
			if reqBody.StreamOptions == nil {
				t.Fatal("expected stream_options to be set")
			}
			if !reqBody.StreamOptions.IncludeUsage {
				t.Error("expected stream_options.include_usage=true")
			}

			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)
			fmt.Fprintf(w, "data: {\"id\":\"x\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}\n\n")
			fmt.Fprintf(w, "data: {\"id\":\"x\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}\n\n")
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Drain channel
		for range ch {
		}
	})
}

// ---------------------------------------------------------------------------
// TestStreamError
// ---------------------------------------------------------------------------

func TestStreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":{"message":"Server error"}}`))
	}))
	defer server.Close()

	adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := adapter.Stream(context.Background(), &llm.Request{
		Model:    "gpt-4o",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
	})

	if err == nil {
		t.Fatal("expected error, got nil")
	}
	llmErr, ok := err.(*llm.LLMError)
	if !ok {
		t.Fatalf("expected *llm.LLMError, got %T", err)
	}
	if llmErr.Type != llm.ErrorTypeServer {
		t.Errorf("expected server error, got %s", llmErr.Type)
	}
}

// ---------------------------------------------------------------------------
// TestName
// ---------------------------------------------------------------------------

func TestName(t *testing.T) {
	adapter := NewAdapter(WithAPIKey("test-key"))
	if adapter.Name() != "openai" {
		t.Errorf("expected name openai, got %s", adapter.Name())
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
// TestDoRequestSendsHeaders
// ---------------------------------------------------------------------------

func TestDoRequestSendsHeaders(t *testing.T) {
	t.Run("basic headers", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "POST" {
				t.Errorf("expected POST, got %s", r.Method)
			}
			if r.Header.Get("Content-Type") != "application/json" {
				t.Errorf("expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
			}
			if r.Header.Get("Authorization") != "Bearer my-key" {
				t.Errorf("expected Authorization Bearer my-key, got %s", r.Header.Get("Authorization"))
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(chatResponse{
				ID:    "chatcmpl-hdr",
				Model: "gpt-4o",
				Choices: []chatChoice{
					{Message: chatMessage{Content: "ok"}, FinishReason: "stop"},
				},
				Usage: chatUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("my-key"), WithBaseURL(server.URL))
		_, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("org ID header", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("OpenAI-Organization") != "org-123" {
				t.Errorf("expected OpenAI-Organization org-123, got %s", r.Header.Get("OpenAI-Organization"))
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(chatResponse{
				ID:    "chatcmpl-org",
				Model: "gpt-4o",
				Choices: []chatChoice{
					{Message: chatMessage{Content: "ok"}, FinishReason: "stop"},
				},
				Usage: chatUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("key"), WithBaseURL(server.URL), WithOrgID("org-123"))
		_, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "gpt-4o",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}
