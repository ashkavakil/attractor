// Package gemini implements the Google Gemini provider adapter for the unified LLM client.
package gemini

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
	llm.RegisterProviderFactory("gemini", "GEMINI_API_KEY,GOOGLE_API_KEY", func() llm.ProviderAdapter {
		return NewAdapter()
	})
}

// Adapter implements llm.ProviderAdapter for Google Gemini.
type Adapter struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// Option configures the Gemini adapter.
type Option func(*Adapter)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(a *Adapter) { a.apiKey = key }
}

// WithBaseURL sets the base URL.
func WithBaseURL(url string) Option {
	return func(a *Adapter) { a.baseURL = url }
}

// NewAdapter creates a new Gemini adapter.
func NewAdapter(opts ...Option) *Adapter {
	a := &Adapter{
		baseURL: "https://generativelanguage.googleapis.com/v1beta",
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
	for _, opt := range opts {
		opt(a)
	}
	if a.apiKey == "" {
		a.apiKey = os.Getenv("GEMINI_API_KEY")
		if a.apiKey == "" {
			a.apiKey = os.Getenv("GOOGLE_API_KEY")
		}
	}
	if url := os.Getenv("GEMINI_BASE_URL"); url != "" && a.baseURL == "https://generativelanguage.googleapis.com/v1beta" {
		a.baseURL = url
	}
	return a
}

func (a *Adapter) Name() string { return "gemini" }
func (a *Adapter) Close() error { return nil }

// Gemini-specific types
type generateRequest struct {
	Contents         []content          `json:"contents"`
	Tools            []geminiTool       `json:"tools,omitempty"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	GenerationConfig *generationConfig  `json:"generationConfig,omitempty"`
}

type content struct {
	Role  string `json:"role"`
	Parts []part `json:"parts"`
}

type part struct {
	Text             string            `json:"text,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
}

type functionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

type functionResponse struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}

type geminiTool struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
}

type functionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type generationConfig struct {
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type generateResponse struct {
	Candidates    []candidate    `json:"candidates"`
	UsageMetadata usageMetadata  `json:"usageMetadata"`
}

type candidate struct {
	Content      content `json:"content"`
	FinishReason string  `json:"finishReason"`
}

type usageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

func convertGeminiFinishReason(reason string) llm.FinishReason {
	switch reason {
	case "STOP":
		return llm.FinishReasonStop
	case "MAX_TOKENS":
		return llm.FinishReasonLength
	case "SAFETY", "RECITATION", "OTHER":
		return llm.FinishReasonStop
	default:
		return llm.FinishReasonStop
	}
}

func (a *Adapter) buildRequest(req *llm.Request) generateRequest {
	gr := generateRequest{}

	if req.SystemPrompt != "" {
		gr.SystemInstruction = &content{
			Parts: []part{{Text: req.SystemPrompt}},
		}
	}

	for _, m := range req.Messages {
		switch m.Role {
		case llm.RoleSystem:
			continue
		case llm.RoleUser:
			gr.Contents = append(gr.Contents, content{
				Role:  "user",
				Parts: []part{{Text: m.Content}},
			})
		case llm.RoleAssistant:
			c := content{Role: "model"}
			if m.Content != "" {
				c.Parts = append(c.Parts, part{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				var args map[string]interface{}
				json.Unmarshal(tc.Arguments, &args)
				c.Parts = append(c.Parts, part{
					FunctionCall: &functionCall{
						Name: tc.Name,
						Args: args,
					},
				})
			}
			gr.Contents = append(gr.Contents, c)
		case llm.RoleTool:
			gr.Contents = append(gr.Contents, content{
				Role: "user",
				Parts: []part{{
					FunctionResponse: &functionResponse{
						Name: m.Name,
						Response: map[string]interface{}{
							"result": m.Content,
						},
					},
				}},
			})
		}
	}

	if len(req.Tools) > 0 {
		decls := make([]functionDeclaration, len(req.Tools))
		for i, t := range req.Tools {
			decls[i] = functionDeclaration{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			}
		}
		gr.Tools = []geminiTool{{FunctionDeclarations: decls}}
	}

	gc := &generationConfig{}
	if req.MaxTokens > 0 {
		gc.MaxOutputTokens = req.MaxTokens
	}
	gc.Temperature = req.Temperature
	gc.TopP = req.TopP
	gc.StopSequences = req.StopSequences
	gr.GenerationConfig = gc

	return gr
}

func (a *Adapter) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	gr := a.buildRequest(req)

	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", a.baseURL, req.Model, a.apiKey)
	data, err := json.Marshal(gr)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, &llm.LLMError{
			Type:     llm.ErrorTypeNetwork,
			Message:  err.Error(),
			Provider: "gemini",
			Cause:    err,
		}
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, llm.ClassifyHTTPError(resp.StatusCode, string(body), "gemini")
	}

	var genResp generateResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return a.convertResponse(&genResp), nil
}

func (a *Adapter) convertResponse(gr *generateResponse) *llm.Response {
	resp := &llm.Response{
		ID:    fmt.Sprintf("gemini-%d", time.Now().UnixNano()),
		Usage: llm.Usage{
			InputTokens:  gr.UsageMetadata.PromptTokenCount,
			OutputTokens: gr.UsageMetadata.CandidatesTokenCount,
			TotalTokens:  gr.UsageMetadata.TotalTokenCount,
		},
		CreatedAt: time.Now(),
	}

	if len(gr.Candidates) > 0 {
		cand := gr.Candidates[0]

		resp.FinishReason = convertGeminiFinishReason(cand.FinishReason)

		var textParts []string
		for _, p := range cand.Content.Parts {
			if p.Text != "" {
				textParts = append(textParts, p.Text)
			}
			if p.FunctionCall != nil {
				args, _ := json.Marshal(p.FunctionCall.Args)
				resp.ToolCalls = append(resp.ToolCalls, llm.ToolCall{
					ID:        fmt.Sprintf("call_%s_%d", p.FunctionCall.Name, time.Now().UnixNano()),
					Name:      p.FunctionCall.Name,
					Arguments: args,
				})
			}
		}
		resp.Content = strings.Join(textParts, "")

		if len(resp.ToolCalls) > 0 {
			resp.FinishReason = llm.FinishReasonToolCalls
		}
	}

	return resp
}

func (a *Adapter) Stream(ctx context.Context, req *llm.Request) (<-chan llm.StreamEvent, error) {
	gr := a.buildRequest(req)

	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s", a.baseURL, req.Model, a.apiKey)
	data, err := json.Marshal(gr)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, &llm.LLMError{
			Type:     llm.ErrorTypeNetwork,
			Message:  err.Error(),
			Provider: "gemini",
			Cause:    err,
		}
	}

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, llm.ClassifyHTTPError(resp.StatusCode, string(body), "gemini")
	}

	ch := make(chan llm.StreamEvent, 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024) // 1MB buffer
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")

			var chunk generateResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			if len(chunk.Candidates) == 0 {
				// May contain usage metadata without candidates
				if chunk.UsageMetadata.TotalTokenCount > 0 {
					ch <- llm.StreamEvent{
						Type: llm.StreamEventEnd,
						Usage: &llm.Usage{
							InputTokens:  chunk.UsageMetadata.PromptTokenCount,
							OutputTokens: chunk.UsageMetadata.CandidatesTokenCount,
							TotalTokens:  chunk.UsageMetadata.TotalTokenCount,
						},
						FinishReason: llm.FinishReasonStop,
					}
				}
				continue
			}

			cand := chunk.Candidates[0]
			for _, p := range cand.Content.Parts {
				if p.Text != "" {
					ch <- llm.StreamEvent{
						Type:  llm.StreamEventDelta,
						Delta: p.Text,
					}
				}
				if p.FunctionCall != nil {
					args, _ := json.Marshal(p.FunctionCall.Args)
					ch <- llm.StreamEvent{
						Type: llm.StreamEventToolCallStart,
						ToolCall: &llm.ToolCall{
							ID:        fmt.Sprintf("call_%s_%d", p.FunctionCall.Name, time.Now().UnixNano()),
							Name:      p.FunctionCall.Name,
							Arguments: args,
						},
					}
				}
			}

			if cand.FinishReason != "" {
				fr := convertGeminiFinishReason(cand.FinishReason)
				hasToolCalls := false
				for _, p := range cand.Content.Parts {
					if p.FunctionCall != nil {
						hasToolCalls = true
						break
					}
				}
				if hasToolCalls {
					fr = llm.FinishReasonToolCalls
				}
				endEvent := llm.StreamEvent{
					Type:         llm.StreamEventEnd,
					FinishReason: fr,
				}
				if chunk.UsageMetadata.TotalTokenCount > 0 {
					endEvent.Usage = &llm.Usage{
						InputTokens:  chunk.UsageMetadata.PromptTokenCount,
						OutputTokens: chunk.UsageMetadata.CandidatesTokenCount,
						TotalTokens:  chunk.UsageMetadata.TotalTokenCount,
					}
				}
				ch <- endEvent
			}
		}
		if err := scanner.Err(); err != nil {
			ch <- llm.StreamEvent{Type: llm.StreamEventError, Error: err}
		}
	}()

	return ch, nil
}
