// Package integration_test contains live API tests that require real API keys.
// Run with: go test ./test/ -run TestLive -v -count=1
// These tests are skipped when no API key is set.
package integration_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ashka-vakil/attractor/pkg/agent"
	"github.com/ashka-vakil/attractor/pkg/agent/env"
	"github.com/ashka-vakil/attractor/pkg/llm"
	"github.com/ashka-vakil/attractor/pkg/llm/provider/openai"
)

func skipIfNoOpenAI(t *testing.T) {
	t.Helper()
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set, skipping live test")
	}
}

// TestLiveOpenAIComplete sends a real request to the OpenAI API and verifies
// we get a coherent response back through the unified LLM client.
func TestLiveOpenAIComplete(t *testing.T) {
	skipIfNoOpenAI(t)

	adapter := openai.NewAdapter()
	client := llm.NewClient(
		llm.WithProvider("openai", adapter),
		llm.WithDefaultProvider("openai"),
	)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := client.Complete(ctx, &llm.Request{
		Model: "gpt-4.1-nano",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "Reply with exactly the word 'pong' and nothing else."},
		},
		MaxTokens: 10,
	})
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}

	// Verify response structure
	if resp.ID == "" {
		t.Error("expected non-empty response ID")
	}
	if resp.Content == "" {
		t.Error("expected non-empty content")
	}
	if !strings.Contains(strings.ToLower(resp.Content), "pong") {
		t.Errorf("expected 'pong' in response, got: %q", resp.Content)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Errorf("expected finish_reason stop, got %s", resp.FinishReason)
	}
	if resp.Usage.InputTokens == 0 {
		t.Error("expected non-zero input tokens")
	}
	if resp.Usage.OutputTokens == 0 {
		t.Error("expected non-zero output tokens")
	}

	t.Logf("Response: %q (input=%d, output=%d tokens)",
		resp.Content, resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

// TestLiveOpenAIStream sends a real streaming request and verifies deltas
// accumulate into a coherent response.
func TestLiveOpenAIStream(t *testing.T) {
	skipIfNoOpenAI(t)

	adapter := openai.NewAdapter()
	client := llm.NewClient(
		llm.WithProvider("openai", adapter),
		llm.WithDefaultProvider("openai"),
	)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	ch, err := client.Stream(ctx, &llm.Request{
		Model: "gpt-4.1-nano",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "Count from 1 to 5, separated by commas."},
		},
		MaxTokens: 30,
	})
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	var accumulated string
	var gotEnd bool
	var endEvent llm.StreamEvent
	eventCount := 0

	for event := range ch {
		eventCount++
		switch event.Type {
		case llm.StreamEventDelta:
			accumulated += event.Delta
		case llm.StreamEventEnd:
			gotEnd = true
			endEvent = event
		case llm.StreamEventError:
			t.Fatalf("stream error: %v", event.Error)
		}
	}

	if accumulated == "" {
		t.Error("expected non-empty accumulated text from stream")
	}
	if !gotEnd {
		t.Error("expected end event")
	}
	if endEvent.FinishReason != llm.FinishReasonStop {
		t.Errorf("expected finish_reason stop, got %s", endEvent.FinishReason)
	}
	if endEvent.Usage != nil && endEvent.Usage.TotalTokens == 0 {
		t.Error("expected non-zero total tokens in usage")
	}

	// Verify the response contains numbers
	for _, n := range []string{"1", "2", "3", "4", "5"} {
		if !strings.Contains(accumulated, n) {
			t.Errorf("expected %q in streamed output: %q", n, accumulated)
		}
	}

	t.Logf("Streamed %d events, accumulated: %q", eventCount, accumulated)
}

// TestLiveOpenAIToolCall sends a request that triggers a tool call and verifies
// the tool call is properly structured.
func TestLiveOpenAIToolCall(t *testing.T) {
	skipIfNoOpenAI(t)

	adapter := openai.NewAdapter()
	client := llm.NewClient(
		llm.WithProvider("openai", adapter),
		llm.WithDefaultProvider("openai"),
	)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := client.Complete(ctx, &llm.Request{
		Model: "gpt-4.1-nano",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "What is the weather in San Francisco? Use the get_weather tool."},
		},
		Tools: []llm.Tool{
			{
				Name:        "get_weather",
				Description: "Get the current weather for a city",
				Parameters:  []byte(`{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}`),
			},
		},
		ToolChoice: llm.ToolChoiceRequired,
		MaxTokens:  100,
	})
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}

	if len(resp.ToolCalls) == 0 {
		t.Fatalf("expected tool calls, got none. Content: %q", resp.Content)
	}

	tc := resp.ToolCalls[0]
	if tc.ID == "" {
		t.Error("expected non-empty tool call ID")
	}
	if tc.Name != "get_weather" {
		t.Errorf("expected tool name 'get_weather', got %q", tc.Name)
	}
	if !strings.Contains(string(tc.Arguments), "city") {
		t.Errorf("expected 'city' in arguments, got: %s", string(tc.Arguments))
	}
	if resp.FinishReason != llm.FinishReasonToolCalls {
		t.Errorf("expected finish_reason tool_calls, got %s", resp.FinishReason)
	}

	t.Logf("Tool call: %s(%s)", tc.Name, string(tc.Arguments))
}

// TestLiveAgentSession runs a real agent session that reads a file using the
// LLM to decide which tool to call.
func TestLiveAgentSession(t *testing.T) {
	skipIfNoOpenAI(t)

	adapter := openai.NewAdapter()
	client := llm.NewClient(
		llm.WithProvider("openai", adapter),
		llm.WithDefaultProvider("openai"),
	)
	defer client.Close()

	// Set up a temp directory with a file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "message.txt")
	os.WriteFile(testFile, []byte("The secret code is 42."), 0o644)

	profile := agent.DefaultOpenAIProfile("gpt-4.1-nano")
	localEnv := env.NewLocalEnvironment(tmpDir)

	config := agent.DefaultSessionConfig()
	config.MaxTurns = 10
	config.MaxToolRoundsPerInput = 5

	session := agent.NewSession(client, profile, localEnv, config)
	defer session.Close()

	// Collect events
	var events []agent.Event
	var mu sync.Mutex
	session.EventEmitter.On(func(e agent.Event) {
		mu.Lock()
		events = append(events, e)
		mu.Unlock()
	})

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	err := session.Submit(ctx, "Read the file message.txt and tell me what the secret code is. Use the read_file tool.")
	if err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	// Verify session completed
	if session.State != agent.StateIdle {
		t.Errorf("expected session state idle, got %s", session.State)
	}

	// Verify history has multiple turns (user, at least one tool round, final response)
	if len(session.History) < 3 {
		t.Errorf("expected at least 3 turns, got %d", len(session.History))
	}

	// Find the final assistant response
	var finalContent string
	for i := len(session.History) - 1; i >= 0; i-- {
		if at, ok := session.History[i].(*agent.AssistantTurn); ok {
			finalContent = at.Content
			break
		}
	}

	if finalContent == "" {
		t.Error("expected non-empty final assistant response")
	}
	if !strings.Contains(finalContent, "42") {
		t.Errorf("expected '42' in final response, got: %q", finalContent)
	}

	// Verify events were emitted
	mu.Lock()
	defer mu.Unlock()

	eventTypes := make(map[agent.EventType]bool)
	for _, e := range events {
		eventTypes[e.Type] = true
	}

	if !eventTypes[agent.EventSessionStarted] {
		t.Error("expected session_started event")
	}
	if !eventTypes[agent.EventToolCallStarted] {
		t.Error("expected tool_call_started event (agent should have called read_file)")
	}

	t.Logf("Session completed with %d turns and %d events", len(session.History), len(events))
	t.Logf("Final response: %q", finalContent)
}
