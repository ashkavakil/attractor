package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

// defaultClient is the module-level default client.
var (
	defaultClient     *Client
	defaultClientOnce sync.Once
	defaultClientMu   sync.Mutex
)

// SetDefaultClient sets the module-level default client.
func SetDefaultClient(c *Client) {
	defaultClientMu.Lock()
	defer defaultClientMu.Unlock()
	defaultClient = c
}

// GetDefaultClient returns the module-level default client, initializing from env if needed.
func GetDefaultClient() *Client {
	defaultClientMu.Lock()
	defer defaultClientMu.Unlock()
	if defaultClient == nil {
		defaultClientOnce.Do(func() {
			defaultClient = FromEnv()
		})
	}
	return defaultClient
}

// GenerateOption configures a Generate call.
type GenerateOption func(*Request)

// WithModel sets the model for the request.
func WithModel(model string) GenerateOption {
	return func(r *Request) {
		r.Model = model
	}
}

// WithProviderOption sets the provider for the request.
func WithProviderOption(provider string) GenerateOption {
	return func(r *Request) {
		r.Provider = provider
	}
}

// WithTools sets tools for the request.
func WithTools(tools ...Tool) GenerateOption {
	return func(r *Request) {
		r.Tools = tools
	}
}

// WithMaxTokens sets the max tokens for the request.
func WithMaxTokens(n int) GenerateOption {
	return func(r *Request) {
		r.MaxTokens = n
	}
}

// WithTemperature sets the temperature for the request.
func WithTemperature(t float64) GenerateOption {
	return func(r *Request) {
		r.Temperature = &t
	}
}

// WithSystemPrompt sets the system prompt for the request.
func WithSystemPrompt(prompt string) GenerateOption {
	return func(r *Request) {
		r.SystemPrompt = prompt
	}
}

// WithReasoningEffort sets the reasoning effort.
func WithReasoningEffort(effort string) GenerateOption {
	return func(r *Request) {
		r.ReasoningEffort = effort
	}
}

// Generate is a high-level function that sends a prompt to the default client.
// It handles tool call loops automatically.
func Generate(ctx context.Context, prompt string, opts ...GenerateOption) (*Response, error) {
	client := GetDefaultClient()
	return GenerateWithClient(ctx, client, prompt, opts...)
}

// GenerateWithClient is like Generate but uses a specific client.
func GenerateWithClient(ctx context.Context, client *Client, prompt string, opts ...GenerateOption) (*Response, error) {
	req := &Request{
		Messages: []Message{
			{Role: RoleUser, Content: prompt},
		},
	}
	for _, opt := range opts {
		opt(req)
	}

	// Tool call loop
	maxRounds := 50
	for round := 0; round < maxRounds; round++ {
		resp, err := client.Complete(ctx, req)
		if err != nil {
			return nil, err
		}

		if len(resp.ToolCalls) == 0 {
			return resp, nil
		}

		// Append assistant turn with tool calls.
		req.Messages = append(req.Messages, Message{
			Role:      RoleAssistant,
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		})

		// We can't execute tools here; return to caller.
		// The high-level API returns the response with tool calls
		// and lets the caller handle execution.
		return resp, nil
	}

	return nil, fmt.Errorf("tool call loop exceeded %d rounds", maxRounds)
}

// GenerateObject generates a structured JSON response and unmarshals it.
func GenerateObject(ctx context.Context, prompt string, result interface{}, opts ...GenerateOption) error {
	opts = append(opts, func(r *Request) {
		r.ResponseFormat = &ResponseFormat{Type: "json_object"}
	})
	resp, err := Generate(ctx, prompt, opts...)
	if err != nil {
		return err
	}
	return json.Unmarshal([]byte(resp.Content), result)
}

// StreamGenerate is a high-level streaming function.
func StreamGenerate(ctx context.Context, prompt string, opts ...GenerateOption) (<-chan StreamEvent, error) {
	client := GetDefaultClient()
	return StreamGenerateWithClient(ctx, client, prompt, opts...)
}

// StreamGenerateWithClient is like StreamGenerate but uses a specific client.
func StreamGenerateWithClient(ctx context.Context, client *Client, prompt string, opts ...GenerateOption) (<-chan StreamEvent, error) {
	req := &Request{
		Messages: []Message{
			{Role: RoleUser, Content: prompt},
		},
	}
	for _, opt := range opts {
		opt(req)
	}
	return client.Stream(ctx, req)
}
