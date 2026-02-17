// Package openai implements the OpenAI provider adapter for the unified LLM client.
package openai

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

func init() {
	llm.RegisterProviderFactory("openai", "OPENAI_API_KEY", func() llm.ProviderAdapter {
		return NewAdapter()
	})
}

// Adapter implements llm.ProviderAdapter for OpenAI.
type Adapter struct {
	apiKey     string
	baseURL    string
	orgID      string
	projectID  string
	httpClient *http.Client
	headers    map[string]string
}

// Option configures the OpenAI adapter.
type Option func(*Adapter)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(a *Adapter) { a.apiKey = key }
}

// WithBaseURL sets the base URL.
func WithBaseURL(url string) Option {
	return func(a *Adapter) { a.baseURL = url }
}

// WithOrgID sets the organization ID.
func WithOrgID(orgID string) Option {
	return func(a *Adapter) { a.orgID = orgID }
}

// WithHeaders sets custom headers.
func WithHeaders(h map[string]string) Option {
	return func(a *Adapter) { a.headers = h }
}

// NewAdapter creates a new OpenAI adapter.
func NewAdapter(opts ...Option) *Adapter {
	a := &Adapter{
		baseURL: "https://api.openai.com/v1",
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
	for _, opt := range opts {
		opt(a)
	}
	if a.apiKey == "" {
		a.apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if url := os.Getenv("OPENAI_BASE_URL"); url != "" && a.baseURL == "https://api.openai.com/v1" {
		a.baseURL = url
	}
	if orgID := os.Getenv("OPENAI_ORG_ID"); orgID != "" && a.orgID == "" {
		a.orgID = orgID
	}
	return a
}

func (a *Adapter) Name() string { return "openai" }

func (a *Adapter) Close() error { return nil }

// OpenAI request/response types
type chatRequest struct {
	Model            string            `json:"model"`
	Messages         []chatMessage     `json:"messages"`
	Tools            []chatTool        `json:"tools,omitempty"`
	ToolChoice       interface{}       `json:"tool_choice,omitempty"`
	MaxTokens        int               `json:"max_tokens,omitempty"`
	Temperature      *float64          `json:"temperature,omitempty"`
	TopP             *float64          `json:"top_p,omitempty"`
	Stop             []string          `json:"stop,omitempty"`
	Stream           bool              `json:"stream,omitempty"`
	StreamOptions    *streamOptions    `json:"stream_options,omitempty"`
	ResponseFormat   *responseFormat   `json:"response_format,omitempty"`
	ReasoningEffort  string            `json:"reasoning_effort,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    interface{}    `json:"content,omitempty"` // string or []contentPart
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

type chatTool struct {
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type responseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

type chatResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Choices []chatChoice   `json:"choices"`
	Usage   chatUsage      `json:"usage"`
	Created int64          `json:"created"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
	Delta        chatMessage `json:"delta"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (a *Adapter) buildRequest(req *llm.Request) chatRequest {
	msgs := make([]chatMessage, 0, len(req.Messages)+1)
	if req.SystemPrompt != "" {
		msgs = append(msgs, chatMessage{
			Role:    "system",
			Content: req.SystemPrompt,
		})
	}
	for _, m := range req.Messages {
		cm := chatMessage{
			Role:       string(m.Role),
			ToolCallID: m.ToolCallID,
			Name:       m.Name,
		}
		if len(m.ToolCalls) > 0 {
			for _, tc := range m.ToolCalls {
				cm.ToolCalls = append(cm.ToolCalls, chatToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: chatFunctionCall{
						Name:      tc.Name,
						Arguments: string(tc.Arguments),
					},
				})
			}
		}
		if m.Content != "" {
			cm.Content = m.Content
		}
		msgs = append(msgs, cm)
	}

	cr := chatRequest{
		Model:       req.Model,
		Messages:    msgs,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.StopSequences,
	}

	if req.ReasoningEffort != "" {
		cr.ReasoningEffort = req.ReasoningEffort
	}

	for _, t := range req.Tools {
		cr.Tools = append(cr.Tools, chatTool{
			Type: "function",
			Function: chatFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}

	if req.ToolChoice != nil {
		switch tc := req.ToolChoice.(type) {
		case llm.ToolChoice:
			cr.ToolChoice = string(tc)
		case llm.ToolChoiceFunction:
			cr.ToolChoice = map[string]interface{}{
				"type":     "function",
				"function": map[string]string{"name": tc.Name},
			}
		}
	}

	if req.ResponseFormat != nil {
		cr.ResponseFormat = &responseFormat{
			Type:       req.ResponseFormat.Type,
			JSONSchema: req.ResponseFormat.JSONSchema,
		}
	}

	return cr
}

func (a *Adapter) doRequest(ctx context.Context, body interface{}, stream bool) (*http.Response, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+"/chat/completions", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+a.apiKey)
	if a.orgID != "" {
		httpReq.Header.Set("OpenAI-Organization", a.orgID)
	}
	if a.projectID != "" {
		httpReq.Header.Set("OpenAI-Project", a.projectID)
	}
	for k, v := range a.headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, &llm.LLMError{
			Type:     llm.ErrorTypeNetwork,
			Message:  err.Error(),
			Provider: "openai",
			Cause:    err,
		}
	}

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, llm.ClassifyHTTPError(resp.StatusCode, string(body), "openai")
	}

	return resp, nil
}

func (a *Adapter) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	cr := a.buildRequest(req)
	cr.Stream = false

	resp, err := a.doRequest(ctx, cr, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var chatResp chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return a.convertResponse(&chatResp), nil
}

func (a *Adapter) convertResponse(cr *chatResponse) *llm.Response {
	resp := &llm.Response{
		ID:    cr.ID,
		Model: cr.Model,
		Usage: llm.Usage{
			InputTokens:  cr.Usage.PromptTokens,
			OutputTokens: cr.Usage.CompletionTokens,
			TotalTokens:  cr.Usage.TotalTokens,
		},
		CreatedAt: time.Unix(cr.Created, 0),
	}

	if len(cr.Choices) > 0 {
		choice := cr.Choices[0]
		if content, ok := choice.Message.Content.(string); ok {
			resp.Content = content
		}

		switch choice.FinishReason {
		case "stop":
			resp.FinishReason = llm.FinishReasonStop
		case "length":
			resp.FinishReason = llm.FinishReasonLength
		case "tool_calls":
			resp.FinishReason = llm.FinishReasonToolCalls
		}

		for _, tc := range choice.Message.ToolCalls {
			resp.ToolCalls = append(resp.ToolCalls, llm.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			})
		}
	}

	return resp
}

func (a *Adapter) Stream(ctx context.Context, req *llm.Request) (<-chan llm.StreamEvent, error) {
	cr := a.buildRequest(req)
	cr.Stream = true
	cr.StreamOptions = &streamOptions{IncludeUsage: true}

	resp, err := a.doRequest(ctx, cr, true)
	if err != nil {
		return nil, err
	}

	ch := make(chan llm.StreamEvent, 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		var finalUsage *llm.Usage
		var finishReason llm.FinishReason

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024) // 1MB buffer
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				endEvent := llm.StreamEvent{
					Type:         llm.StreamEventEnd,
					FinishReason: finishReason,
				}
				if finalUsage != nil {
					endEvent.Usage = finalUsage
				}
				ch <- endEvent
				break
			}

			var chunk chatResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				ch <- llm.StreamEvent{Type: llm.StreamEventError, Error: err}
				return
			}

			// Capture usage from the final chunk (sent when stream_options.include_usage is true)
			if chunk.Usage.TotalTokens > 0 {
				finalUsage = &llm.Usage{
					InputTokens:  chunk.Usage.PromptTokens,
					OutputTokens: chunk.Usage.CompletionTokens,
					TotalTokens:  chunk.Usage.TotalTokens,
				}
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			delta := choice.Delta

			if content, ok := delta.Content.(string); ok && content != "" {
				ch <- llm.StreamEvent{
					Type:  llm.StreamEventDelta,
					Delta: content,
				}
			}

			for _, tc := range delta.ToolCalls {
				if tc.Function.Name != "" {
					ch <- llm.StreamEvent{
						Type: llm.StreamEventToolCallStart,
						ToolCall: &llm.ToolCall{
							ID:   tc.ID,
							Name: tc.Function.Name,
						},
					}
				}
				if tc.Function.Arguments != "" {
					ch <- llm.StreamEvent{
						Type:  llm.StreamEventToolCallDelta,
						Delta: tc.Function.Arguments,
					}
				}
			}

			if choice.FinishReason != "" {
				switch choice.FinishReason {
				case "stop":
					finishReason = llm.FinishReasonStop
				case "length":
					finishReason = llm.FinishReasonLength
				case "tool_calls":
					finishReason = llm.FinishReasonToolCalls
				default:
					finishReason = llm.FinishReasonStop
				}
			}
		}
	}()

	return ch, nil
}
