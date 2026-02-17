package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
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
			Model: "gemini-2.0-flash",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hello"},
			},
		}
		gr := adapter.buildRequest(req)

		if len(gr.Contents) != 1 {
			t.Fatalf("expected 1 content, got %d", len(gr.Contents))
		}
		if gr.Contents[0].Role != "user" {
			t.Errorf("expected role user, got %s", gr.Contents[0].Role)
		}
		if len(gr.Contents[0].Parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(gr.Contents[0].Parts))
		}
		if gr.Contents[0].Parts[0].Text != "Hello" {
			t.Errorf("expected text Hello, got %s", gr.Contents[0].Parts[0].Text)
		}
	})

	t.Run("system instruction", func(t *testing.T) {
		req := &llm.Request{
			Model:        "gemini-2.0-flash",
			SystemPrompt: "You are a helpful assistant.",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hi"},
			},
		}
		gr := adapter.buildRequest(req)

		if gr.SystemInstruction == nil {
			t.Fatal("expected system instruction to be set")
		}
		if len(gr.SystemInstruction.Parts) != 1 {
			t.Fatalf("expected 1 part in system instruction, got %d", len(gr.SystemInstruction.Parts))
		}
		if gr.SystemInstruction.Parts[0].Text != "You are a helpful assistant." {
			t.Errorf("expected system text, got %s", gr.SystemInstruction.Parts[0].Text)
		}
	})

	t.Run("system role message is skipped", func(t *testing.T) {
		req := &llm.Request{
			Model: "gemini-2.0-flash",
			Messages: []llm.Message{
				{Role: llm.RoleSystem, Content: "System msg"},
				{Role: llm.RoleUser, Content: "Hi"},
			},
		}
		gr := adapter.buildRequest(req)

		if len(gr.Contents) != 1 {
			t.Fatalf("expected 1 content (system skipped), got %d", len(gr.Contents))
		}
		if gr.Contents[0].Role != "user" {
			t.Errorf("expected user role, got %s", gr.Contents[0].Role)
		}
	})

	t.Run("assistant message maps to model role", func(t *testing.T) {
		req := &llm.Request{
			Model: "gemini-2.0-flash",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hi"},
				{Role: llm.RoleAssistant, Content: "Hello!"},
			},
		}
		gr := adapter.buildRequest(req)

		if len(gr.Contents) != 2 {
			t.Fatalf("expected 2 contents, got %d", len(gr.Contents))
		}
		if gr.Contents[1].Role != "model" {
			t.Errorf("expected model role for assistant, got %s", gr.Contents[1].Role)
		}
		if gr.Contents[1].Parts[0].Text != "Hello!" {
			t.Errorf("expected text Hello!, got %s", gr.Contents[1].Parts[0].Text)
		}
	})

	t.Run("assistant with tool calls", func(t *testing.T) {
		req := &llm.Request{
			Model: "gemini-2.0-flash",
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
		gr := adapter.buildRequest(req)

		if len(gr.Contents) != 2 {
			t.Fatalf("expected 2 contents, got %d", len(gr.Contents))
		}

		modelContent := gr.Contents[1]
		if modelContent.Role != "model" {
			t.Errorf("expected role model, got %s", modelContent.Role)
		}
		if len(modelContent.Parts) != 2 {
			t.Fatalf("expected 2 parts (text + function call), got %d", len(modelContent.Parts))
		}
		if modelContent.Parts[0].Text != "Searching..." {
			t.Errorf("expected text part, got %s", modelContent.Parts[0].Text)
		}
		if modelContent.Parts[1].FunctionCall == nil {
			t.Fatal("expected function call part")
		}
		if modelContent.Parts[1].FunctionCall.Name != "search" {
			t.Errorf("expected function call name search, got %s", modelContent.Parts[1].FunctionCall.Name)
		}
		if modelContent.Parts[1].FunctionCall.Args["q"] != "test" {
			t.Errorf("expected args q=test, got %v", modelContent.Parts[1].FunctionCall.Args)
		}
	})

	t.Run("tool result message", func(t *testing.T) {
		req := &llm.Request{
			Model: "gemini-2.0-flash",
			Messages: []llm.Message{
				{Role: llm.RoleTool, Name: "search", Content: "search results"},
			},
		}
		gr := adapter.buildRequest(req)

		if len(gr.Contents) != 1 {
			t.Fatalf("expected 1 content, got %d", len(gr.Contents))
		}
		if gr.Contents[0].Role != "user" {
			t.Errorf("expected user role for tool result, got %s", gr.Contents[0].Role)
		}
		if gr.Contents[0].Parts[0].FunctionResponse == nil {
			t.Fatal("expected function response part")
		}
		if gr.Contents[0].Parts[0].FunctionResponse.Name != "search" {
			t.Errorf("expected function response name search, got %s", gr.Contents[0].Parts[0].FunctionResponse.Name)
		}
		respMap := gr.Contents[0].Parts[0].FunctionResponse.Response
		if respMap["result"] != "search results" {
			t.Errorf("expected result 'search results', got %v", respMap["result"])
		}
	})

	t.Run("tools", func(t *testing.T) {
		req := &llm.Request{
			Model: "gemini-2.0-flash",
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hi"},
			},
			Tools: []llm.Tool{
				{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
				},
				{
					Name:        "search",
					Description: "Search the web",
					Parameters:  json.RawMessage(`{"type":"object"}`),
				},
			},
		}
		gr := adapter.buildRequest(req)

		if len(gr.Tools) != 1 {
			t.Fatalf("expected 1 tool group, got %d", len(gr.Tools))
		}
		if len(gr.Tools[0].FunctionDeclarations) != 2 {
			t.Fatalf("expected 2 function declarations, got %d", len(gr.Tools[0].FunctionDeclarations))
		}
		if gr.Tools[0].FunctionDeclarations[0].Name != "get_weather" {
			t.Errorf("expected first tool name get_weather, got %s", gr.Tools[0].FunctionDeclarations[0].Name)
		}
		if gr.Tools[0].FunctionDeclarations[1].Name != "search" {
			t.Errorf("expected second tool name search, got %s", gr.Tools[0].FunctionDeclarations[1].Name)
		}
	})

	t.Run("generation config", func(t *testing.T) {
		temp := 0.7
		topP := 0.9
		req := &llm.Request{
			Model:         "gemini-2.0-flash",
			Messages:      []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
			MaxTokens:     2048,
			Temperature:   &temp,
			TopP:          &topP,
			StopSequences: []string{"END", "STOP"},
		}
		gr := adapter.buildRequest(req)

		if gr.GenerationConfig == nil {
			t.Fatal("expected generation config to be set")
		}
		if gr.GenerationConfig.MaxOutputTokens != 2048 {
			t.Errorf("expected maxOutputTokens 2048, got %d", gr.GenerationConfig.MaxOutputTokens)
		}
		if gr.GenerationConfig.Temperature == nil || *gr.GenerationConfig.Temperature != 0.7 {
			t.Errorf("expected temperature 0.7, got %v", gr.GenerationConfig.Temperature)
		}
		if gr.GenerationConfig.TopP == nil || *gr.GenerationConfig.TopP != 0.9 {
			t.Errorf("expected topP 0.9, got %v", gr.GenerationConfig.TopP)
		}
		if len(gr.GenerationConfig.StopSequences) != 2 {
			t.Fatalf("expected 2 stop sequences, got %d", len(gr.GenerationConfig.StopSequences))
		}
	})

	t.Run("no tools when empty", func(t *testing.T) {
		req := &llm.Request{
			Model:    "gemini-2.0-flash",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		}
		gr := adapter.buildRequest(req)

		if len(gr.Tools) != 0 {
			t.Errorf("expected no tools, got %d", len(gr.Tools))
		}
	})

	t.Run("no system instruction when empty", func(t *testing.T) {
		req := &llm.Request{
			Model:    "gemini-2.0-flash",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		}
		gr := adapter.buildRequest(req)

		if gr.SystemInstruction != nil {
			t.Errorf("expected nil system instruction, got %+v", gr.SystemInstruction)
		}
	})
}

// ---------------------------------------------------------------------------
// TestComplete
// ---------------------------------------------------------------------------

func TestComplete(t *testing.T) {
	t.Run("text response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify URL path contains model and generateContent
			if !strings.Contains(r.URL.Path, "/models/gemini-2.0-flash:generateContent") {
				t.Errorf("unexpected path: %s", r.URL.Path)
			}
			// Verify API key in query
			if r.URL.Query().Get("key") != "test-key" {
				t.Errorf("expected key=test-key, got %s", r.URL.Query().Get("key"))
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(generateResponse{
				Candidates: []candidate{
					{
						Content: content{
							Role: "model",
							Parts: []part{
								{Text: "Hello! How can I help?"},
							},
						},
						FinishReason: "STOP",
					},
				},
				UsageMetadata: usageMetadata{
					PromptTokenCount:     10,
					CandidatesTokenCount: 20,
					TotalTokenCount:      30,
				},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		resp, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "gemini-2.0-flash",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hello"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
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
		// ID should start with gemini-
		if !strings.HasPrefix(resp.ID, "gemini-") {
			t.Errorf("expected ID to start with gemini-, got %s", resp.ID)
		}
	})

	t.Run("function call response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(generateResponse{
				Candidates: []candidate{
					{
						Content: content{
							Role: "model",
							Parts: []part{
								{
									FunctionCall: &functionCall{
										Name: "get_weather",
										Args: map[string]interface{}{"city": "NYC"},
									},
								},
							},
						},
						FinishReason: "STOP",
					},
				},
				UsageMetadata: usageMetadata{
					PromptTokenCount:     15,
					CandidatesTokenCount: 10,
					TotalTokenCount:      25,
				},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		resp, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "gemini-2.0-flash",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Weather?"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// When there are tool calls, the finish reason is overridden to tool_calls
		if resp.FinishReason != llm.FinishReasonToolCalls {
			t.Errorf("expected finish_reason tool_calls, got %s", resp.FinishReason)
		}
		if len(resp.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
		}
		if resp.ToolCalls[0].Name != "get_weather" {
			t.Errorf("expected tool call name get_weather, got %s", resp.ToolCalls[0].Name)
		}
		// Verify arguments
		var args map[string]interface{}
		if err := json.Unmarshal(resp.ToolCalls[0].Arguments, &args); err != nil {
			t.Fatalf("failed to unmarshal arguments: %v", err)
		}
		if args["city"] != "NYC" {
			t.Errorf("expected city NYC, got %v", args["city"])
		}
		// Tool call ID should start with call_get_weather_
		if !strings.HasPrefix(resp.ToolCalls[0].ID, "call_get_weather_") {
			t.Errorf("expected tool call ID to start with call_get_weather_, got %s", resp.ToolCalls[0].ID)
		}
	})

	t.Run("text and function call combined", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(generateResponse{
				Candidates: []candidate{
					{
						Content: content{
							Role: "model",
							Parts: []part{
								{Text: "Let me check the weather."},
								{
									FunctionCall: &functionCall{
										Name: "get_weather",
										Args: map[string]interface{}{"city": "SF"},
									},
								},
							},
						},
						FinishReason: "STOP",
					},
				},
				UsageMetadata: usageMetadata{
					PromptTokenCount:     20,
					CandidatesTokenCount: 15,
					TotalTokenCount:      35,
				},
			})
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		resp, err := adapter.Complete(context.Background(), &llm.Request{
			Model:    "gemini-2.0-flash",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Weather in SF?"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content != "Let me check the weather." {
			t.Errorf("expected text content, got %s", resp.Content)
		}
		if len(resp.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
		}
		if resp.FinishReason != llm.FinishReasonToolCalls {
			t.Errorf("expected finish_reason tool_calls (override when tool calls present), got %s", resp.FinishReason)
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
			body:         `{"error":{"message":"API key not valid"}}`,
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
			body:         `{"error":{"message":"Resource exhausted"}}`,
			expectedType: llm.ErrorTypeRateLimit,
		},
		{
			name:         "400 bad request",
			statusCode:   400,
			body:         `{"error":{"message":"Invalid model"}}`,
			expectedType: llm.ErrorTypeBadRequest,
		},
		{
			name:         "500 server error",
			statusCode:   500,
			body:         `{"error":{"message":"Internal error"}}`,
			expectedType: llm.ErrorTypeServer,
		},
		{
			name:         "503 service unavailable",
			statusCode:   503,
			body:         `{"error":{"message":"Service unavailable"}}`,
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
				Model:    "gemini-2.0-flash",
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
			if llmErr.Provider != "gemini" {
				t.Errorf("expected provider gemini, got %s", llmErr.Provider)
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
			// Verify URL contains streamGenerateContent
			if !strings.Contains(r.URL.Path, "/models/gemini-2.0-flash:streamGenerateContent") {
				t.Errorf("unexpected path: %s", r.URL.Path)
			}
			if r.URL.Query().Get("alt") != "sse" {
				t.Errorf("expected alt=sse, got %s", r.URL.Query().Get("alt"))
			}
			if r.URL.Query().Get("key") != "test-key" {
				t.Errorf("expected key=test-key, got %s", r.URL.Query().Get("key"))
			}

			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)

			events := []string{
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]},"finishReason":""}],"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0,"totalTokenCount":0}}`,
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":""}],"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0,"totalTokenCount":0}}`,
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`,
			}

			for _, event := range events {
				fmt.Fprintf(w, "%s\n\n", event)
				flusher.Flush()
			}
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "gemini-2.0-flash",
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
		if text != "Hello world!" {
			t.Errorf("expected accumulated text 'Hello world!', got %s", text)
		}

		// Find end event
		var foundEnd bool
		for _, ev := range events {
			if ev.Type == llm.StreamEventEnd {
				foundEnd = true
				if ev.FinishReason != llm.FinishReasonStop {
					t.Errorf("expected finish_reason stop, got %s", ev.FinishReason)
				}
				if ev.Usage == nil {
					t.Error("expected usage in end event")
				} else {
					if ev.Usage.InputTokens != 10 {
						t.Errorf("expected input_tokens 10, got %d", ev.Usage.InputTokens)
					}
					if ev.Usage.OutputTokens != 5 {
						t.Errorf("expected output_tokens 5, got %d", ev.Usage.OutputTokens)
					}
					if ev.Usage.TotalTokens != 15 {
						t.Errorf("expected total_tokens 15, got %d", ev.Usage.TotalTokens)
					}
				}
			}
		}
		if !foundEnd {
			t.Error("expected end event")
		}
	})

	t.Run("function call streaming", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)

			events := []string{
				`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"NYC"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":10,"totalTokenCount":25}}`,
			}

			for _, event := range events {
				fmt.Fprintf(w, "%s\n\n", event)
				flusher.Flush()
			}
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "gemini-2.0-flash",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Weather?"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var events []llm.StreamEvent
		for ev := range ch {
			events = append(events, ev)
		}

		// Should have tool_call_start event
		var foundToolStart bool
		for _, ev := range events {
			if ev.Type == llm.StreamEventToolCallStart {
				foundToolStart = true
				if ev.ToolCall == nil {
					t.Fatal("expected tool call in event")
				}
				if ev.ToolCall.Name != "get_weather" {
					t.Errorf("expected tool call name get_weather, got %s", ev.ToolCall.Name)
				}
				// Verify args are marshalled
				var args map[string]interface{}
				if err := json.Unmarshal(ev.ToolCall.Arguments, &args); err != nil {
					t.Fatalf("failed to unmarshal tool call args: %v", err)
				}
				if args["city"] != "NYC" {
					t.Errorf("expected city NYC, got %v", args["city"])
				}
				// ID should start with call_get_weather_
				if !strings.HasPrefix(ev.ToolCall.ID, "call_get_weather_") {
					t.Errorf("expected tool call ID to start with call_get_weather_, got %s", ev.ToolCall.ID)
				}
			}
		}
		if !foundToolStart {
			t.Error("expected tool_call_start event")
		}

		// End event should have tool_calls finish reason (overridden due to function calls)
		var foundEnd bool
		for _, ev := range events {
			if ev.Type == llm.StreamEventEnd {
				foundEnd = true
				if ev.FinishReason != llm.FinishReasonToolCalls {
					t.Errorf("expected finish_reason tool_calls, got %s", ev.FinishReason)
				}
				if ev.Usage == nil {
					t.Error("expected usage in end event")
				} else {
					if ev.Usage.InputTokens != 15 {
						t.Errorf("expected input_tokens 15, got %d", ev.Usage.InputTokens)
					}
				}
			}
		}
		if !foundEnd {
			t.Error("expected end event")
		}
	})

	t.Run("usage-only chunk without candidates", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)

			events := []string{
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]},"finishReason":""}],"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0,"totalTokenCount":0}}`,
				// Usage-only chunk with no candidates
				`data: {"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`,
			}

			for _, event := range events {
				fmt.Fprintf(w, "%s\n\n", event)
				flusher.Flush()
			}
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		ch, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "gemini-2.0-flash",
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
		})

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var events []llm.StreamEvent
		for ev := range ch {
			events = append(events, ev)
		}

		// Should have a delta for "Hello" and an end event from the usage-only chunk
		var foundDelta, foundEnd bool
		for _, ev := range events {
			if ev.Type == llm.StreamEventDelta && ev.Delta == "Hello" {
				foundDelta = true
			}
			if ev.Type == llm.StreamEventEnd {
				foundEnd = true
				if ev.Usage == nil {
					t.Error("expected usage in end event from usage-only chunk")
				} else if ev.Usage.TotalTokens != 15 {
					t.Errorf("expected total_tokens 15, got %d", ev.Usage.TotalTokens)
				}
				if ev.FinishReason != llm.FinishReasonStop {
					t.Errorf("expected finish_reason stop, got %s", ev.FinishReason)
				}
			}
		}
		if !foundDelta {
			t.Error("expected delta event with 'Hello'")
		}
		if !foundEnd {
			t.Error("expected end event from usage-only chunk")
		}
	})

	t.Run("stream error on HTTP error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error":{"message":"Quota exceeded"}}`))
		}))
		defer server.Close()

		adapter := NewAdapter(WithAPIKey("test-key"), WithBaseURL(server.URL))
		_, err := adapter.Stream(context.Background(), &llm.Request{
			Model:    "gemini-2.0-flash",
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
	})
}

// ---------------------------------------------------------------------------
// TestConvertGeminiFinishReason
// ---------------------------------------------------------------------------

func TestConvertGeminiFinishReason(t *testing.T) {
	tests := []struct {
		input    string
		expected llm.FinishReason
	}{
		{"STOP", llm.FinishReasonStop},
		{"MAX_TOKENS", llm.FinishReasonLength},
		{"SAFETY", llm.FinishReasonStop},
		{"RECITATION", llm.FinishReasonStop},
		{"OTHER", llm.FinishReasonStop},
		{"UNKNOWN", llm.FinishReasonStop},
		{"", llm.FinishReasonStop},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("input=%q", tt.input), func(t *testing.T) {
			result := convertGeminiFinishReason(tt.input)
			if result != tt.expected {
				t.Errorf("convertGeminiFinishReason(%q) = %s, want %s", tt.input, result, tt.expected)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestName
// ---------------------------------------------------------------------------

func TestName(t *testing.T) {
	adapter := NewAdapter(WithAPIKey("test-key"))
	if adapter.Name() != "gemini" {
		t.Errorf("expected name gemini, got %s", adapter.Name())
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
// TestCompleteURLConstruction
// ---------------------------------------------------------------------------

func TestCompleteURLConstruction(t *testing.T) {
	var capturedPath string
	var capturedQuery string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedQuery = r.URL.RawQuery

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(generateResponse{
			Candidates: []candidate{
				{
					Content:      content{Role: "model", Parts: []part{{Text: "ok"}}},
					FinishReason: "STOP",
				},
			},
			UsageMetadata: usageMetadata{PromptTokenCount: 1, CandidatesTokenCount: 1, TotalTokenCount: 2},
		})
	}))
	defer server.Close()

	adapter := NewAdapter(WithAPIKey("my-api-key"), WithBaseURL(server.URL))
	_, err := adapter.Complete(context.Background(), &llm.Request{
		Model:    "gemini-2.0-flash-exp",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedPath := "/models/gemini-2.0-flash-exp:generateContent"
	if capturedPath != expectedPath {
		t.Errorf("expected path %s, got %s", expectedPath, capturedPath)
	}
	if !strings.Contains(capturedQuery, "key=my-api-key") {
		t.Errorf("expected query to contain key=my-api-key, got %s", capturedQuery)
	}
}

// ---------------------------------------------------------------------------
// TestStreamURLConstruction
// ---------------------------------------------------------------------------

func TestStreamURLConstruction(t *testing.T) {
	var capturedPath string
	var capturedQuery string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedQuery = r.URL.RawQuery

		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	adapter := NewAdapter(WithAPIKey("my-api-key"), WithBaseURL(server.URL))
	ch, err := adapter.Stream(context.Background(), &llm.Request{
		Model:    "gemini-2.0-flash-exp",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Drain channel
	for range ch {
	}

	expectedPath := "/models/gemini-2.0-flash-exp:streamGenerateContent"
	if capturedPath != expectedPath {
		t.Errorf("expected path %s, got %s", expectedPath, capturedPath)
	}
	if !strings.Contains(capturedQuery, "alt=sse") {
		t.Errorf("expected query to contain alt=sse, got %s", capturedQuery)
	}
	if !strings.Contains(capturedQuery, "key=my-api-key") {
		t.Errorf("expected query to contain key=my-api-key, got %s", capturedQuery)
	}
}
