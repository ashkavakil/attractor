// Package llm provides a unified interface for interacting with multiple LLM providers.
package llm

import (
	"context"
	"encoding/json"
	"time"
)

// Role represents the role of a message participant.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
	RoleDeveloper Role = "developer"
)

// FinishReason describes why the model stopped generating.
type FinishReason string

const (
	FinishReasonStop      FinishReason = "stop"
	FinishReasonLength    FinishReason = "length"
	FinishReasonToolCalls FinishReason = "tool_calls"
	FinishReasonError     FinishReason = "error"
)

// ContentPartType identifies the type of content in a message.
type ContentPartType string

const (
	ContentPartText  ContentPartType = "text"
	ContentPartImage ContentPartType = "image"
)

// ContentPart is a single piece of content within a message.
type ContentPart struct {
	Type     ContentPartType `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL string          `json:"image_url,omitempty"`
	MimeType string          `json:"mime_type,omitempty"`
	Data     []byte          `json:"data,omitempty"`
}

// Message is a single message in a conversation.
type Message struct {
	Role       Role          `json:"role"`
	Content    string        `json:"content,omitempty"`
	Parts      []ContentPart `json:"parts,omitempty"`
	ToolCalls  []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	Name       string        `json:"name,omitempty"`
}

// Tool defines a tool the model can call.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolCall is a request from the model to execute a tool.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolResult is the result of executing a tool call.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
	IsError    bool   `json:"is_error,omitempty"`
}

// ToolChoice controls how the model selects tools.
type ToolChoice string

const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceRequired ToolChoice = "required"
)

// ToolChoiceFunction forces a specific function to be called.
type ToolChoiceFunction struct {
	Name string `json:"name"`
}

// Usage tracks token consumption for a request.
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	TotalTokens      int `json:"total_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`
	CachedTokens     int `json:"cached_tokens,omitempty"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

// Request is a provider-agnostic LLM request.
type Request struct {
	Model           string              `json:"model"`
	Provider        string              `json:"provider,omitempty"`
	Messages        []Message           `json:"messages"`
	Tools           []Tool              `json:"tools,omitempty"`
	ToolChoice      interface{}         `json:"tool_choice,omitempty"` // ToolChoice or ToolChoiceFunction
	MaxTokens       int                 `json:"max_tokens,omitempty"`
	Temperature     *float64            `json:"temperature,omitempty"`
	TopP            *float64            `json:"top_p,omitempty"`
	StopSequences   []string            `json:"stop_sequences,omitempty"`
	SystemPrompt    string              `json:"system_prompt,omitempty"`
	ReasoningEffort string              `json:"reasoning_effort,omitempty"`
	ResponseFormat  *ResponseFormat     `json:"response_format,omitempty"`
	ProviderOptions map[string]interface{} `json:"provider_options,omitempty"`
}

// ResponseFormat controls the output format from the model.
type ResponseFormat struct {
	Type       string          `json:"type"` // "text", "json_object", "json_schema"
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

// Response is a provider-agnostic LLM response.
type Response struct {
	ID           string       `json:"id"`
	Model        string       `json:"model"`
	Content      string       `json:"content"`
	Parts        []ContentPart `json:"parts,omitempty"`
	ToolCalls    []ToolCall   `json:"tool_calls,omitempty"`
	FinishReason FinishReason `json:"finish_reason"`
	Usage        Usage        `json:"usage"`
	Reasoning    string       `json:"reasoning,omitempty"`
	CreatedAt    time.Time      `json:"created_at"`
	Raw          interface{}    `json:"-"`                          // Preserved raw provider response
	Warnings     []Warning      `json:"warnings,omitempty"`        // Non-fatal issues
	RateLimit    *RateLimitInfo `json:"rate_limit,omitempty"`      // Rate limit metadata
}

// Warning is a non-fatal issue from a response.
type Warning struct {
	Message string `json:"message"`
	Code    string `json:"code,omitempty"`
}

// RateLimitInfo contains rate limit metadata from provider headers.
type RateLimitInfo struct {
	RequestsRemaining int        `json:"requests_remaining,omitempty"`
	RequestsLimit     int        `json:"requests_limit,omitempty"`
	TokensRemaining   int        `json:"tokens_remaining,omitempty"`
	TokensLimit       int        `json:"tokens_limit,omitempty"`
	ResetAt           *time.Time `json:"reset_at,omitempty"`
}

// StepResult tracks a single step in a multi-step tool loop.
type StepResult struct {
	Text         string       `json:"text"`
	Reasoning    string       `json:"reasoning,omitempty"`
	ToolCalls    []ToolCall   `json:"tool_calls,omitempty"`
	ToolResults  []ToolResult `json:"tool_results,omitempty"`
	FinishReason FinishReason `json:"finish_reason"`
	Usage        Usage        `json:"usage"`
	Response     *Response    `json:"response"`
	Warnings     []Warning    `json:"warnings,omitempty"`
}

// GenerateResult is the complete result from a high-level generate call.
type GenerateResult struct {
	Text         string       `json:"text"`
	Reasoning    string       `json:"reasoning,omitempty"`
	ToolCalls    []ToolCall   `json:"tool_calls,omitempty"`
	ToolResults  []ToolResult `json:"tool_results,omitempty"`
	FinishReason FinishReason `json:"finish_reason"`
	Usage        Usage        `json:"usage"`       // Usage from final step
	TotalUsage   Usage        `json:"total_usage"` // Aggregated across all steps
	Steps        []StepResult `json:"steps"`
	Response     *Response    `json:"response"`
}

// Add adds two Usage values together.
func (u Usage) Add(other Usage) Usage {
	return Usage{
		InputTokens:      u.InputTokens + other.InputTokens,
		OutputTokens:     u.OutputTokens + other.OutputTokens,
		TotalTokens:      u.TotalTokens + other.TotalTokens,
		ReasoningTokens:  u.ReasoningTokens + other.ReasoningTokens,
		CachedTokens:     u.CachedTokens + other.CachedTokens,
		CacheReadTokens:  u.CacheReadTokens + other.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens + other.CacheWriteTokens,
	}
}

// StreamAccumulator accumulates stream events into a complete Response.
type StreamAccumulator struct {
	content      string
	reasoning    string
	toolCalls    []ToolCall
	finishReason FinishReason
	usage        *Usage
	model        string
	id           string
}

// Process handles a single stream event.
func (a *StreamAccumulator) Process(event StreamEvent) {
	switch event.Type {
	case StreamEventDelta:
		a.content += event.Delta
	case StreamEventReasoningDelta:
		a.reasoning += event.Delta
	case StreamEventToolCallStart:
		if event.ToolCall != nil {
			a.toolCalls = append(a.toolCalls, *event.ToolCall)
		}
	case StreamEventToolCallDelta:
		if len(a.toolCalls) > 0 && event.Delta != "" {
			last := &a.toolCalls[len(a.toolCalls)-1]
			last.Arguments = append(last.Arguments, []byte(event.Delta)...)
		}
	case StreamEventEnd:
		a.finishReason = event.FinishReason
		if event.Usage != nil {
			a.usage = event.Usage
		}
	}
}

// Response returns the accumulated response.
func (a *StreamAccumulator) Response() *Response {
	resp := &Response{
		ID:           a.id,
		Model:        a.model,
		Content:      a.content,
		Reasoning:    a.reasoning,
		ToolCalls:    a.toolCalls,
		FinishReason: a.finishReason,
		CreatedAt:    time.Now(),
	}
	if a.usage != nil {
		resp.Usage = *a.usage
	}
	return resp
}

// StreamEventType identifies the type of stream event.
type StreamEventType string

const (
	StreamEventStart         StreamEventType = "start"
	StreamEventDelta         StreamEventType = "delta"
	StreamEventToolCallStart StreamEventType = "tool_call_start"
	StreamEventToolCallDelta StreamEventType = "tool_call_delta"
	StreamEventToolCallEnd   StreamEventType = "tool_call_end"
	StreamEventReasoningDelta StreamEventType = "reasoning_delta"
	StreamEventEnd           StreamEventType = "end"
	StreamEventError         StreamEventType = "error"
)

// StreamEvent is a single event in a streaming response.
type StreamEvent struct {
	Type         StreamEventType `json:"type"`
	Delta        string          `json:"delta,omitempty"`
	ToolCall     *ToolCall       `json:"tool_call,omitempty"`
	FinishReason FinishReason    `json:"finish_reason,omitempty"`
	Usage        *Usage          `json:"usage,omitempty"`
	Response     *Response       `json:"response,omitempty"`
	Error        error           `json:"-"`
}

// ProviderAdapter is the interface every LLM provider must implement.
type ProviderAdapter interface {
	// Name returns the provider identifier.
	Name() string

	// Complete sends a request and blocks until the response is ready.
	Complete(ctx context.Context, req *Request) (*Response, error)

	// Stream sends a request and returns a channel of stream events.
	Stream(ctx context.Context, req *Request) (<-chan StreamEvent, error)

	// Close releases resources.
	Close() error
}

// Middleware wraps provider calls for cross-cutting concerns.
type Middleware func(ctx context.Context, req *Request, next MiddlewareNext) (*Response, error)

// MiddlewareNext is the function to call the next middleware or provider.
type MiddlewareNext func(ctx context.Context, req *Request) (*Response, error)

// StreamMiddleware wraps streaming provider calls.
type StreamMiddleware func(ctx context.Context, req *Request, next StreamMiddlewareNext) (<-chan StreamEvent, error)

// StreamMiddlewareNext is the function to call the next middleware or provider for streaming.
type StreamMiddlewareNext func(ctx context.Context, req *Request) (<-chan StreamEvent, error)
