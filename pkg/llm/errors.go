package llm

import (
	"fmt"
	"time"
)

// ErrorType categorizes LLM errors for retry decisions.
type ErrorType string

const (
	ErrorTypeAuth       ErrorType = "auth"
	ErrorTypeRateLimit  ErrorType = "rate_limit"
	ErrorTypeServer     ErrorType = "server"
	ErrorTypeNetwork    ErrorType = "network"
	ErrorTypeBadRequest ErrorType = "bad_request"
	ErrorTypeTimeout    ErrorType = "timeout"
	ErrorTypeUnknown    ErrorType = "unknown"
)

// LLMError is a structured error from an LLM provider.
type LLMError struct {
	Type       ErrorType `json:"type"`
	Message    string    `json:"message"`
	StatusCode int       `json:"status_code,omitempty"`
	Provider   string    `json:"provider,omitempty"`
	RetryAfter time.Duration `json:"retry_after,omitempty"`
	Cause      error     `json:"-"`
}

func (e *LLMError) Error() string {
	if e.StatusCode > 0 {
		return fmt.Sprintf("[%s] %s (HTTP %d): %s", e.Provider, e.Type, e.StatusCode, e.Message)
	}
	return fmt.Sprintf("[%s] %s: %s", e.Provider, e.Type, e.Message)
}

func (e *LLMError) Unwrap() error {
	return e.Cause
}

// IsRetryable returns true if the error is likely transient.
func (e *LLMError) IsRetryable() bool {
	switch e.Type {
	case ErrorTypeRateLimit, ErrorTypeServer, ErrorTypeNetwork, ErrorTypeTimeout:
		return true
	default:
		return false
	}
}

// ClassifyHTTPError maps HTTP status codes to error types.
func ClassifyHTTPError(statusCode int, body string, provider string) *LLMError {
	var errType ErrorType
	switch {
	case statusCode == 401 || statusCode == 403:
		errType = ErrorTypeAuth
	case statusCode == 429:
		errType = ErrorTypeRateLimit
	case statusCode == 400 || statusCode == 422:
		errType = ErrorTypeBadRequest
	case statusCode >= 500:
		errType = ErrorTypeServer
	default:
		errType = ErrorTypeUnknown
	}
	return &LLMError{
		Type:       errType,
		Message:    body,
		StatusCode: statusCode,
		Provider:   provider,
	}
}
