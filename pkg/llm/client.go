package llm

import (
	"context"
	"fmt"
	"os"
	"sync"
)

// Client is the core orchestration layer that routes requests to provider adapters.
type Client struct {
	mu              sync.RWMutex
	providers       map[string]ProviderAdapter
	defaultProvider string
	middleware      []Middleware
	streamMW       []StreamMiddleware
}

// ClientOption configures a Client.
type ClientOption func(*Client)

// WithProvider registers a provider adapter.
func WithProvider(name string, adapter ProviderAdapter) ClientOption {
	return func(c *Client) {
		c.providers[name] = adapter
	}
}

// WithDefaultProvider sets the default provider name.
func WithDefaultProvider(name string) ClientOption {
	return func(c *Client) {
		c.defaultProvider = name
	}
}

// WithMiddleware adds request/response middleware.
func WithMiddleware(mw ...Middleware) ClientOption {
	return func(c *Client) {
		c.middleware = append(c.middleware, mw...)
	}
}

// WithStreamMiddleware adds streaming middleware.
func WithStreamMiddleware(mw ...StreamMiddleware) ClientOption {
	return func(c *Client) {
		c.streamMW = append(c.streamMW, mw...)
	}
}

// NewClient creates a new Client with the given options.
func NewClient(opts ...ClientOption) *Client {
	c := &Client{
		providers: make(map[string]ProviderAdapter),
	}
	for _, opt := range opts {
		opt(c)
	}
	if c.defaultProvider == "" && len(c.providers) == 1 {
		for name := range c.providers {
			c.defaultProvider = name
		}
	}
	return c
}

// FromEnv creates a Client from environment variables.
// Registers providers whose API keys are present.
func FromEnv() *Client {
	c := &Client{
		providers: make(map[string]ProviderAdapter),
	}
	// Provider registration is handled by the provider packages.
	// This is a placeholder; actual providers register themselves.
	_ = os.Getenv("OPENAI_API_KEY")
	_ = os.Getenv("ANTHROPIC_API_KEY")
	_ = os.Getenv("GEMINI_API_KEY")
	return c
}

// RegisterProvider adds a provider adapter to the client.
func (c *Client) RegisterProvider(name string, adapter ProviderAdapter) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.providers[name] = adapter
	if c.defaultProvider == "" {
		c.defaultProvider = name
	}
}

// resolveProvider determines which provider to use for a request.
func (c *Client) resolveProvider(req *Request) (ProviderAdapter, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	provider := req.Provider
	if provider == "" {
		provider = c.defaultProvider
	}
	if provider == "" {
		return nil, fmt.Errorf("no provider specified and no default provider set")
	}
	adapter, ok := c.providers[provider]
	if !ok {
		return nil, fmt.Errorf("provider %q not registered", provider)
	}
	return adapter, nil
}

// Complete sends a blocking request to the resolved provider, applying middleware.
func (c *Client) Complete(ctx context.Context, req *Request) (*Response, error) {
	adapter, err := c.resolveProvider(req)
	if err != nil {
		return nil, err
	}

	// Build the middleware chain.
	final := func(ctx context.Context, r *Request) (*Response, error) {
		return adapter.Complete(ctx, r)
	}

	chain := final
	for i := len(c.middleware) - 1; i >= 0; i-- {
		mw := c.middleware[i]
		next := chain
		chain = func(ctx context.Context, r *Request) (*Response, error) {
			return mw(ctx, r, next)
		}
	}

	return chain(ctx, req)
}

// Stream sends a streaming request to the resolved provider, applying middleware.
func (c *Client) Stream(ctx context.Context, req *Request) (<-chan StreamEvent, error) {
	adapter, err := c.resolveProvider(req)
	if err != nil {
		return nil, err
	}

	final := func(ctx context.Context, r *Request) (<-chan StreamEvent, error) {
		return adapter.Stream(ctx, r)
	}

	chain := final
	for i := len(c.streamMW) - 1; i >= 0; i-- {
		mw := c.streamMW[i]
		next := chain
		chain = func(ctx context.Context, r *Request) (<-chan StreamEvent, error) {
			return mw(ctx, r, next)
		}
	}

	return chain(ctx, req)
}

// Close releases all provider resources.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	var firstErr error
	for _, adapter := range c.providers {
		if err := adapter.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
