package llm

import (
	"context"
	"testing"
	"time"
)

// mockAdapter is a test adapter.
type mockAdapter struct {
	name     string
	response *Response
	err      error
}

func (m *mockAdapter) Name() string { return m.name }
func (m *mockAdapter) Close() error { return nil }
func (m *mockAdapter) Complete(ctx context.Context, req *Request) (*Response, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.response, nil
}
func (m *mockAdapter) Stream(ctx context.Context, req *Request) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent, 2)
	go func() {
		defer close(ch)
		ch <- StreamEvent{Type: StreamEventDelta, Delta: m.response.Content}
		ch <- StreamEvent{Type: StreamEventEnd, FinishReason: FinishReasonStop}
	}()
	return ch, nil
}

func TestClientComplete(t *testing.T) {
	adapter := &mockAdapter{
		name: "test",
		response: &Response{
			ID:           "resp-1",
			Content:      "Hello, world!",
			FinishReason: FinishReasonStop,
			CreatedAt:    time.Now(),
		},
	}

	client := NewClient(WithProvider("test", adapter), WithDefaultProvider("test"))
	resp, err := client.Complete(context.Background(), &Request{
		Model: "test-model",
		Messages: []Message{
			{Role: RoleUser, Content: "Hi"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "Hello, world!" {
		t.Errorf("expected %q, got %q", "Hello, world!", resp.Content)
	}
}

func TestClientNoProvider(t *testing.T) {
	client := NewClient()
	_, err := client.Complete(context.Background(), &Request{
		Messages: []Message{{Role: RoleUser, Content: "Hi"}},
	})
	if err == nil {
		t.Error("expected error when no provider configured")
	}
}

func TestClientProviderResolution(t *testing.T) {
	adapter1 := &mockAdapter{name: "a", response: &Response{Content: "from-a"}}
	adapter2 := &mockAdapter{name: "b", response: &Response{Content: "from-b"}}

	client := NewClient(
		WithProvider("a", adapter1),
		WithProvider("b", adapter2),
		WithDefaultProvider("a"),
	)

	// Default provider
	resp, _ := client.Complete(context.Background(), &Request{
		Messages: []Message{{Role: RoleUser, Content: "Hi"}},
	})
	if resp.Content != "from-a" {
		t.Errorf("expected from-a, got %q", resp.Content)
	}

	// Explicit provider
	resp, _ = client.Complete(context.Background(), &Request{
		Provider: "b",
		Messages: []Message{{Role: RoleUser, Content: "Hi"}},
	})
	if resp.Content != "from-b" {
		t.Errorf("expected from-b, got %q", resp.Content)
	}
}

func TestClientMiddleware(t *testing.T) {
	adapter := &mockAdapter{
		name:     "test",
		response: &Response{Content: "original"},
	}

	called := false
	mw := func(ctx context.Context, req *Request, next MiddlewareNext) (*Response, error) {
		called = true
		resp, err := next(ctx, req)
		if err != nil {
			return nil, err
		}
		resp.Content = "modified"
		return resp, nil
	}

	client := NewClient(
		WithProvider("test", adapter),
		WithDefaultProvider("test"),
		WithMiddleware(mw),
	)

	resp, err := client.Complete(context.Background(), &Request{
		Messages: []Message{{Role: RoleUser, Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("middleware was not called")
	}
	if resp.Content != "modified" {
		t.Errorf("expected modified, got %q", resp.Content)
	}
}

func TestClientStream(t *testing.T) {
	adapter := &mockAdapter{
		name:     "test",
		response: &Response{Content: "streamed"},
	}

	client := NewClient(WithProvider("test", adapter), WithDefaultProvider("test"))
	ch, err := client.Stream(context.Background(), &Request{
		Messages: []Message{{Role: RoleUser, Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var events []StreamEvent
	for event := range ch {
		events = append(events, event)
	}

	if len(events) < 2 {
		t.Fatalf("expected at least 2 events, got %d", len(events))
	}
	if events[0].Delta != "streamed" {
		t.Errorf("expected delta 'streamed', got %q", events[0].Delta)
	}
}

func TestClientClose(t *testing.T) {
	adapter := &mockAdapter{name: "test", response: &Response{}}
	client := NewClient(WithProvider("test", adapter))
	if err := client.Close(); err != nil {
		t.Errorf("unexpected close error: %v", err)
	}
}

func TestSingleProviderDefaultSelection(t *testing.T) {
	adapter := &mockAdapter{name: "only", response: &Response{Content: "ok"}}
	client := NewClient(WithProvider("only", adapter))

	// Should auto-select the only provider as default
	resp, err := client.Complete(context.Background(), &Request{
		Messages: []Message{{Role: RoleUser, Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("expected ok, got %q", resp.Content)
	}
}
