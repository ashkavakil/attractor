// Package testutil provides reusable test utilities for the Attractor project.
package testutil

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"time"

	"github.com/ashka-vakil/attractor/pkg/llm"
)

// MockAdapter implements llm.ProviderAdapter with configurable responses.
type MockAdapter struct {
	mu              sync.Mutex
	name            string
	CompleteFunc    func(ctx context.Context, req *llm.Request) (*llm.Response, error)
	StreamFunc      func(ctx context.Context, req *llm.Request) (<-chan llm.StreamEvent, error)
	CompleteCalls   []*llm.Request
	StreamCalls     []*llm.Request
}

// NewMockAdapter creates a MockAdapter with sensible defaults.
func NewMockAdapter(name string) *MockAdapter {
	return &MockAdapter{
		name: name,
		CompleteFunc: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
			return &llm.Response{
				ID:           "mock-response-1",
				Model:        "mock-model",
				Content:      "Hello from mock",
				FinishReason: llm.FinishReasonStop,
				Usage:        llm.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
				CreatedAt:    time.Now(),
			}, nil
		},
	}
}

func (m *MockAdapter) Name() string { return m.name }
func (m *MockAdapter) Close() error { return nil }

func (m *MockAdapter) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	m.mu.Lock()
	m.CompleteCalls = append(m.CompleteCalls, req)
	m.mu.Unlock()
	return m.CompleteFunc(ctx, req)
}

func (m *MockAdapter) Stream(ctx context.Context, req *llm.Request) (<-chan llm.StreamEvent, error) {
	m.mu.Lock()
	m.StreamCalls = append(m.StreamCalls, req)
	m.mu.Unlock()
	if m.StreamFunc != nil {
		return m.StreamFunc(ctx, req)
	}
	// Default: convert Complete response to stream events
	ch := make(chan llm.StreamEvent, 4)
	go func() {
		defer close(ch)
		resp, err := m.CompleteFunc(ctx, req)
		if err != nil {
			ch <- llm.StreamEvent{Type: llm.StreamEventError, Error: err}
			return
		}
		ch <- llm.StreamEvent{Type: llm.StreamEventStart}
		if resp.Content != "" {
			ch <- llm.StreamEvent{Type: llm.StreamEventDelta, Delta: resp.Content}
		}
		ch <- llm.StreamEvent{
			Type:         llm.StreamEventEnd,
			FinishReason: resp.FinishReason,
			Usage:        &resp.Usage,
		}
	}()
	return ch, nil
}

// MockResponse creates a simple text response for testing.
func MockResponse(content string) *llm.Response {
	return &llm.Response{
		ID:           fmt.Sprintf("mock-%d", time.Now().UnixNano()),
		Model:        "mock-model",
		Content:      content,
		FinishReason: llm.FinishReasonStop,
		Usage:        llm.Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30},
		CreatedAt:    time.Now(),
	}
}

// MockToolCallResponse creates a response with tool calls for testing.
func MockToolCallResponse(toolCalls []llm.ToolCall) *llm.Response {
	return &llm.Response{
		ID:           fmt.Sprintf("mock-%d", time.Now().UnixNano()),
		Model:        "mock-model",
		ToolCalls:    toolCalls,
		FinishReason: llm.FinishReasonToolCalls,
		Usage:        llm.Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30},
		CreatedAt:    time.Now(),
	}
}

// NewMockClient creates an llm.Client with a MockAdapter.
func NewMockClient(adapter *MockAdapter) *llm.Client {
	return llm.NewClient(
		llm.WithProvider(adapter.Name(), adapter),
		llm.WithDefaultProvider(adapter.Name()),
	)
}

// SSEServer creates an httptest.Server that serves SSE responses.
// The handler function receives the request body and returns SSE lines to send.
type SSEHandler func(reqBody []byte) []string

// NewSSEServer creates a test server for SSE streaming.
func NewSSEServer(handler SSEHandler) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := make([]byte, r.ContentLength)
		r.Body.Read(body)
		r.Body.Close()

		lines := handler(body)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		for _, line := range lines {
			fmt.Fprintf(w, "%s\n", line)
		}
	}))
}

// NewJSONServer creates a test server that returns a JSON response.
func NewJSONServer(statusCode int, response interface{}) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		json.NewEncoder(w).Encode(response)
	}))
}

// CollectStreamEvents drains a stream event channel and returns all events.
func CollectStreamEvents(ch <-chan llm.StreamEvent) []llm.StreamEvent {
	var events []llm.StreamEvent
	for event := range ch {
		events = append(events, event)
	}
	return events
}
