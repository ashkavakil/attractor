// Package anthropic implements the Anthropic provider adapter for the unified LLM client.
package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ashka-vakil/attractor/pkg/llm"
)

// Adapter implements llm.ProviderAdapter for Anthropic.
type Adapter struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	headers    map[string]string
}

// Option configures the Anthropic adapter.
type Option func(*Adapter)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(a *Adapter) { a.apiKey = key }
}

// WithBaseURL sets the base URL.
func WithBaseURL(url string) Option {
	return func(a *Adapter) { a.baseURL = url }
}

// NewAdapter creates a new Anthropic adapter.
func NewAdapter(opts ...Option) *Adapter {
	a := &Adapter{
		baseURL: "https://api.anthropic.com",
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
	for _, opt := range opts {
		opt(a)
	}
	if a.apiKey == "" {
		a.apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if url := os.Getenv("ANTHROPIC_BASE_URL"); url != "" && a.baseURL == "https://api.anthropic.com" {
		a.baseURL = url
	}
	return a
}

func (a *Adapter) Name() string { return "anthropic" }
func (a *Adapter) Close() error { return nil }

// Anthropic-specific request/response types
type messagesRequest struct {
	Model         string           `json:"model"`
	Messages      []messageParam   `json:"messages"`
	System        string           `json:"system,omitempty"`
	MaxTokens     int              `json:"max_tokens"`
	Temperature   *float64         `json:"temperature,omitempty"`
	TopP          *float64         `json:"top_p,omitempty"`
	StopSequences []string         `json:"stop_sequences,omitempty"`
	Tools         []toolParam      `json:"tools,omitempty"`
	ToolChoice    interface{}      `json:"tool_choice,omitempty"`
	Stream        bool             `json:"stream,omitempty"`
	Thinking      *thinkingParam   `json:"thinking,omitempty"`
}

type thinkingParam struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type messageParam struct {
	Role    string        `json:"role"`
	Content interface{}   `json:"content"` // string or []contentBlock
}

type contentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   interface{}     `json:"content,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
}

type toolParam struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type messagesResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []contentBlock `json:"content"`
	Model        string         `json:"model"`
	StopReason   string         `json:"stop_reason"`
	Usage        anthropicUsage `json:"usage"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func (a *Adapter) buildRequest(req *llm.Request) messagesRequest {
	msgs := make([]messageParam, 0, len(req.Messages))
	for _, m := range req.Messages {
		switch m.Role {
		case llm.RoleSystem:
			continue // handled via system field
		case llm.RoleUser:
			msgs = append(msgs, messageParam{Role: "user", Content: m.Content})
		case llm.RoleAssistant:
			if len(m.ToolCalls) > 0 {
				blocks := []contentBlock{}
				if m.Content != "" {
					blocks = append(blocks, contentBlock{Type: "text", Text: m.Content})
				}
				for _, tc := range m.ToolCalls {
					blocks = append(blocks, contentBlock{
						Type:  "tool_use",
						ID:    tc.ID,
						Name:  tc.Name,
						Input: tc.Arguments,
					})
				}
				msgs = append(msgs, messageParam{Role: "assistant", Content: blocks})
			} else {
				msgs = append(msgs, messageParam{Role: "assistant", Content: m.Content})
			}
		case llm.RoleTool:
			msgs = append(msgs, messageParam{
				Role: "user",
				Content: []contentBlock{
					{
						Type:      "tool_result",
						ToolUseID: m.ToolCallID,
						Content:   m.Content,
					},
				},
			})
		}
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	mr := messagesRequest{
		Model:         req.Model,
		Messages:      msgs,
		MaxTokens:     maxTokens,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: req.StopSequences,
	}

	if req.SystemPrompt != "" {
		mr.System = req.SystemPrompt
	}

	for _, t := range req.Tools {
		mr.Tools = append(mr.Tools, toolParam{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Parameters,
		})
	}

	if req.ToolChoice != nil {
		switch tc := req.ToolChoice.(type) {
		case llm.ToolChoice:
			switch tc {
			case llm.ToolChoiceAuto:
				mr.ToolChoice = map[string]string{"type": "auto"}
			case llm.ToolChoiceNone:
				// Anthropic doesn't have "none", omit tools
				mr.Tools = nil
			case llm.ToolChoiceRequired:
				mr.ToolChoice = map[string]string{"type": "any"}
			}
		case llm.ToolChoiceFunction:
			mr.ToolChoice = map[string]interface{}{
				"type": "tool",
				"name": tc.Name,
			}
		}
	}

	return mr
}

func (a *Adapter) doRequest(ctx context.Context, body interface{}, stream bool) (*http.Response, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+"/v1/messages", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", a.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	for k, v := range a.headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, &llm.LLMError{
			Type:     llm.ErrorTypeNetwork,
			Message:  err.Error(),
			Provider: "anthropic",
			Cause:    err,
		}
	}

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, llm.ClassifyHTTPError(resp.StatusCode, string(respBody), "anthropic")
	}

	return resp, nil
}

func (a *Adapter) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	mr := a.buildRequest(req)
	mr.Stream = false

	resp, err := a.doRequest(ctx, mr, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var msgResp messagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&msgResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return a.convertResponse(&msgResp), nil
}

func (a *Adapter) convertResponse(mr *messagesResponse) *llm.Response {
	resp := &llm.Response{
		ID:    mr.ID,
		Model: mr.Model,
		Usage: llm.Usage{
			InputTokens:  mr.Usage.InputTokens,
			OutputTokens: mr.Usage.OutputTokens,
			TotalTokens:  mr.Usage.InputTokens + mr.Usage.OutputTokens,
		},
		CreatedAt: time.Now(),
	}

	switch mr.StopReason {
	case "end_turn":
		resp.FinishReason = llm.FinishReasonStop
	case "max_tokens":
		resp.FinishReason = llm.FinishReasonLength
	case "tool_use":
		resp.FinishReason = llm.FinishReasonToolCalls
	}

	var textParts []string
	for _, block := range mr.Content {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "tool_use":
			resp.ToolCalls = append(resp.ToolCalls, llm.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: block.Input,
			})
		case "thinking":
			resp.Reasoning = block.Thinking
		}
	}
	resp.Content = strings.Join(textParts, "")

	return resp
}

func (a *Adapter) Stream(ctx context.Context, req *llm.Request) (<-chan llm.StreamEvent, error) {
	mr := a.buildRequest(req)
	mr.Stream = true

	resp, err := a.doRequest(ctx, mr, true)
	if err != nil {
		return nil, err
	}

	ch := make(chan llm.StreamEvent, 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")

			var event struct {
				Type         string         `json:"type"`
				Delta        *contentBlock  `json:"delta,omitempty"`
				ContentBlock *contentBlock  `json:"content_block,omitempty"`
				Message      *messagesResponse `json:"message,omitempty"`
				Usage        *anthropicUsage   `json:"usage,omitempty"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			switch event.Type {
			case "message_start":
				ch <- llm.StreamEvent{Type: llm.StreamEventStart}
			case "content_block_start":
				if event.ContentBlock != nil && event.ContentBlock.Type == "tool_use" {
					ch <- llm.StreamEvent{
						Type: llm.StreamEventToolCallStart,
						ToolCall: &llm.ToolCall{
							ID:   event.ContentBlock.ID,
							Name: event.ContentBlock.Name,
						},
					}
				}
			case "content_block_delta":
				if event.Delta != nil {
					switch event.Delta.Type {
					case "text_delta":
						ch <- llm.StreamEvent{
							Type:  llm.StreamEventDelta,
							Delta: event.Delta.Text,
						}
					case "input_json_delta":
						ch <- llm.StreamEvent{
							Type:  llm.StreamEventToolCallDelta,
							Delta: event.Delta.Text,
						}
					case "thinking_delta":
						ch <- llm.StreamEvent{
							Type:  llm.StreamEventReasoningDelta,
							Delta: event.Delta.Thinking,
						}
					}
				}
			case "content_block_stop":
				// No specific action needed
			case "message_delta":
				if event.Delta != nil {
					// The delta here contains stop_reason
				}
			case "message_stop":
				ch <- llm.StreamEvent{
					Type:         llm.StreamEventEnd,
					FinishReason: llm.FinishReasonStop,
				}
			}
		}
	}()

	return ch, nil
}
