package integration_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
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
	"github.com/ashka-vakil/attractor/pkg/pipeline"
	"github.com/ashka-vakil/attractor/pkg/pipeline/handler"
	"github.com/ashka-vakil/attractor/pkg/pipeline/transform"
)

// registryAdapter bridges handler.Registry to pipeline.HandlerResolver.
type registryAdapter struct {
	registry *handler.Registry
}

func (r *registryAdapter) Resolve(node *pipeline.Node) pipeline.Handler {
	h := r.registry.Resolve(node)
	if h == nil {
		return nil
	}
	return h
}

// ---------- Test: End-to-End Agent Loop ----------

func TestEndToEndAgentLoop(t *testing.T) {
	// Track which request we're on (1st = tool_call, 2nd = text response).
	var mu sync.Mutex
	requestCount := 0

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		r.Body.Close()
		_ = body // available for debugging

		mu.Lock()
		requestCount++
		current := requestCount
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")

		if current == 1 {
			// First response: tool_call for read_file
			resp := map[string]interface{}{
				"id":      "chatcmpl-1",
				"model":   "gpt-4.1",
				"created": time.Now().Unix(),
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"message": map[string]interface{}{
							"role":    "assistant",
							"content": nil,
							"tool_calls": []map[string]interface{}{
								{
									"id":   "call_001",
									"type": "function",
									"function": map[string]interface{}{
										"name":      "read_file",
										"arguments": `{"path":"hello.txt"}`,
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
				"usage": map[string]interface{}{
					"prompt_tokens":     50,
					"completion_tokens": 20,
					"total_tokens":      70,
				},
			}
			json.NewEncoder(w).Encode(resp)
		} else {
			// Second response: text with finish_reason stop
			resp := map[string]interface{}{
				"id":      "chatcmpl-2",
				"model":   "gpt-4.1",
				"created": time.Now().Unix(),
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"message": map[string]interface{}{
							"role":    "assistant",
							"content": "The file says hello",
						},
						"finish_reason": "stop",
					},
				},
				"usage": map[string]interface{}{
					"prompt_tokens":     80,
					"completion_tokens": 10,
					"total_tokens":      90,
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer mockServer.Close()

	// Create OpenAI adapter pointing at the test server.
	adapter := openai.NewAdapter(
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(mockServer.URL),
	)

	// Create LLM client with the adapter.
	client := llm.NewClient(
		llm.WithProvider("openai", adapter),
		llm.WithDefaultProvider("openai"),
	)
	defer client.Close()

	// Create a temp directory and write hello.txt.
	tmpDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmpDir, "hello.txt"), []byte("hello world"), 0o644); err != nil {
		t.Fatalf("failed to write hello.txt: %v", err)
	}

	// Create an agent.ProviderProfile.
	profile := agent.DefaultOpenAIProfile("gpt-4.1")

	// Create a real local environment pointing at the temp dir.
	localEnv := env.NewLocalEnvironment(tmpDir)

	// Create session config.
	config := agent.DefaultSessionConfig()
	config.MaxToolRoundsPerInput = 10

	// Create agent session.
	session := agent.NewSession(client, profile, localEnv, config)
	defer session.Close()

	// Collect events.
	var events []agent.Event
	var eventsMu sync.Mutex
	session.EventEmitter.On(func(e agent.Event) {
		eventsMu.Lock()
		events = append(events, e)
		eventsMu.Unlock()
	})

	// Submit a user query.
	ctx := context.Background()
	err := session.Submit(ctx, "What does hello.txt say?")
	if err != nil {
		t.Fatalf("session.Submit failed: %v", err)
	}

	// Verify: session is idle.
	if session.State != agent.StateIdle {
		t.Errorf("expected session state idle, got %s", session.State)
	}

	// Verify: history has at least 4 turns (user, assistant+tool_call, tool_results, assistant+text).
	if len(session.History) < 4 {
		t.Errorf("expected at least 4 turns in history, got %d", len(session.History))
		for i, turn := range session.History {
			t.Logf("  turn %d: %T", i, turn)
		}
	}

	// Verify: the final assistant response contains "The file says hello".
	foundFinalResponse := false
	for i := len(session.History) - 1; i >= 0; i-- {
		if at, ok := session.History[i].(*agent.AssistantTurn); ok {
			if strings.Contains(at.Content, "The file says hello") {
				foundFinalResponse = true
			}
			break
		}
	}
	if !foundFinalResponse {
		t.Error("final assistant response does not contain 'The file says hello'")
	}

	// Verify events.
	eventsMu.Lock()
	defer eventsMu.Unlock()

	eventTypes := make(map[agent.EventType]bool)
	for _, e := range events {
		eventTypes[e.Type] = true
	}

	expectedEvents := []agent.EventType{
		agent.EventSessionStarted,
		agent.EventToolCallStarted,
		agent.EventToolCallCompleted,
		agent.EventTurnCompleted,
	}
	for _, expected := range expectedEvents {
		if !eventTypes[expected] {
			t.Errorf("expected event %s was not emitted", expected)
		}
	}
}

// ---------- Test: End-to-End Pipeline HTTP API ----------

func TestEndToEndPipelineHTTPAPI(t *testing.T) {
	// Create a handler registry.
	registry := handler.NewRegistry(nil, &handler.AutoApproveInterviewer{})
	resolver := &registryAdapter{registry: registry}

	// Create a pipeline server.
	server := pipeline.NewServer(resolver)
	ts := httptest.NewServer(server.Handler())
	defer ts.Close()

	// POST /pipelines with a simple DOT source.
	dotSource := `digraph simple {
		goal = "Simple test"
		start [shape=Mdiamond, label="Start"]
		work  [shape=box, label="Work", prompt="Do work"]
		done  [shape=Msquare, label="Done"]
		start -> work
		work -> done
	}`

	body := fmt.Sprintf(`{"dot_source": %s}`, jsonString(dotSource))
	resp, err := http.Post(ts.URL+"/pipelines", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /pipelines failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 201, got %d: %s", resp.StatusCode, string(respBody))
	}

	var createResp struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&createResp); err != nil {
		t.Fatalf("failed to decode create response: %v", err)
	}
	if createResp.ID == "" {
		t.Fatal("expected non-empty pipeline ID")
	}

	// Poll GET /pipelines/{id} until completed or failed (with timeout).
	deadline := time.Now().Add(10 * time.Second)
	var status string
	for time.Now().Before(deadline) {
		getResp, err := http.Get(ts.URL + "/pipelines/" + createResp.ID)
		if err != nil {
			t.Fatalf("GET /pipelines/%s failed: %v", createResp.ID, err)
		}

		var pipelineResp struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		}
		json.NewDecoder(getResp.Body).Decode(&pipelineResp)
		getResp.Body.Close()

		status = pipelineResp.Status
		if status == "completed" || status == "failed" {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if status != "completed" {
		t.Errorf("expected pipeline status 'completed', got %q", status)
	}
}

// ---------- Test: End-to-End Pipeline Runner ----------

func TestEndToEndPipelineRunner(t *testing.T) {
	// Read the example pipeline DOT.
	exampleDOT, err := os.ReadFile(filepath.Join("..", "examples", "hello_world.dot"))
	if err != nil {
		t.Fatalf("failed to read example DOT file: %v", err)
	}

	// Create a handler registry.
	registry := handler.NewRegistry(nil, &handler.AutoApproveInterviewer{})
	resolver := &registryAdapter{registry: registry}

	// Create a Runner.
	runner := pipeline.NewRunner(resolver)

	// Register default transforms.
	runner.RegisterTransform(transform.VariableExpansion())
	runner.RegisterTransform(transform.StylesheetApplication())

	// Run the pipeline.
	result, err := runner.RunFromSource(string(exampleDOT))
	if err != nil {
		t.Fatalf("RunFromSource failed: %v", err)
	}

	// Verify result status.
	if result.Status != pipeline.StatusSuccess {
		t.Errorf("expected status %s, got %s", pipeline.StatusSuccess, result.Status)
	}

	// Verify all expected stages completed.
	expectedStages := []string{"start", "plan", "implement", "review"}
	for _, stage := range expectedStages {
		found := false
		for _, completed := range result.CompletedNodes {
			if completed == stage {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected stage %q to be completed, completed stages: %v", stage, result.CompletedNodes)
		}
	}
}

// ---------- Test: End-to-End Streaming with Mock Server ----------

func TestEndToEndStreamingWithMockServer(t *testing.T) {
	// Start an httptest.Server that returns SSE chunks in OpenAI format.
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		r.Body.Close()

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		// Chunk 1: "Hello"
		fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]interface{}{
			"id":      "chatcmpl-stream-1",
			"model":   "gpt-4.1",
			"created": time.Now().Unix(),
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"content": "Hello",
					},
					"finish_reason": nil,
				},
			},
		}))
		flusher.Flush()

		// Chunk 2: " "
		fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]interface{}{
			"id":      "chatcmpl-stream-1",
			"model":   "gpt-4.1",
			"created": time.Now().Unix(),
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"content": " ",
					},
					"finish_reason": nil,
				},
			},
		}))
		flusher.Flush()

		// Chunk 3: "World"
		fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]interface{}{
			"id":      "chatcmpl-stream-1",
			"model":   "gpt-4.1",
			"created": time.Now().Unix(),
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"content": "World",
					},
					"finish_reason": nil,
				},
			},
		}))
		flusher.Flush()

		// Chunk 4: finish_reason=stop
		fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]interface{}{
			"id":      "chatcmpl-stream-1",
			"model":   "gpt-4.1",
			"created": time.Now().Unix(),
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"delta":         map[string]interface{}{},
					"finish_reason": "stop",
				},
			},
		}))
		flusher.Flush()

		// Final chunk: [DONE] with usage
		fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]interface{}{
			"id":      "chatcmpl-stream-1",
			"model":   "gpt-4.1",
			"created": time.Now().Unix(),
			"choices": []map[string]interface{}{},
			"usage": map[string]interface{}{
				"prompt_tokens":     30,
				"completion_tokens": 5,
				"total_tokens":      35,
			},
		}))
		flusher.Flush()

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer mockServer.Close()

	// Create OpenAI adapter pointing at the test server.
	adapter := openai.NewAdapter(
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(mockServer.URL),
	)

	// Call Stream.
	ctx := context.Background()
	req := &llm.Request{
		Model: "gpt-4.1",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "Say hello world"},
		},
	}

	ch, err := adapter.Stream(ctx, req)
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}

	// Collect all StreamEvents.
	var events []llm.StreamEvent
	for event := range ch {
		events = append(events, event)
	}

	// Verify: got StreamEventDelta events that concatenate to "Hello World".
	var concatenated string
	for _, e := range events {
		if e.Type == llm.StreamEventDelta {
			concatenated += e.Delta
		}
	}
	if concatenated != "Hello World" {
		t.Errorf("expected concatenated deltas to be 'Hello World', got %q", concatenated)
	}

	// Verify: got StreamEventEnd with FinishReasonStop and usage.
	foundEnd := false
	for _, e := range events {
		if e.Type == llm.StreamEventEnd {
			foundEnd = true
			if e.FinishReason != llm.FinishReasonStop {
				t.Errorf("expected finish reason 'stop', got %q", e.FinishReason)
			}
			if e.Usage == nil {
				t.Error("expected usage in end event, got nil")
			} else if e.Usage.TotalTokens != 35 {
				t.Errorf("expected total tokens 35, got %d", e.Usage.TotalTokens)
			}
		}
	}
	if !foundEnd {
		t.Error("did not receive StreamEventEnd")
	}
}

// ---------- Helpers ----------

func mustJSON(v interface{}) string {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(data)
}

func jsonString(s string) string {
	data, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return string(data)
}
