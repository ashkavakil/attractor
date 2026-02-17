package llm

import (
	"context"
	"math"
	"math/rand"
	"time"
)

// RetryConfig configures retry behavior.
type RetryConfig struct {
	MaxAttempts    int           `json:"max_attempts"`
	InitialDelay  time.Duration `json:"initial_delay"`
	BackoffFactor float64       `json:"backoff_factor"`
	MaxDelay      time.Duration `json:"max_delay"`
	Jitter        bool          `json:"jitter"`
	ShouldRetry   func(error) bool
}

// DefaultRetryConfig returns the standard retry configuration.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:    5,
		InitialDelay:  200 * time.Millisecond,
		BackoffFactor: 2.0,
		MaxDelay:      60 * time.Second,
		Jitter:        true,
		ShouldRetry:   DefaultShouldRetry,
	}
}

// NoRetryConfig returns a config that does not retry.
func NoRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts: 1,
	}
}

// AggressiveRetryConfig returns a config for unreliable operations.
func AggressiveRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:    5,
		InitialDelay:  500 * time.Millisecond,
		BackoffFactor: 2.0,
		MaxDelay:      60 * time.Second,
		Jitter:        true,
		ShouldRetry:   DefaultShouldRetry,
	}
}

// LinearRetryConfig returns a config with fixed delays.
func LinearRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:    3,
		InitialDelay:  500 * time.Millisecond,
		BackoffFactor: 1.0,
		MaxDelay:      60 * time.Second,
		Jitter:        false,
		ShouldRetry:   DefaultShouldRetry,
	}
}

// PatientRetryConfig returns a config for long-running operations.
func PatientRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:    3,
		InitialDelay:  2 * time.Second,
		BackoffFactor: 3.0,
		MaxDelay:      60 * time.Second,
		Jitter:        true,
		ShouldRetry:   DefaultShouldRetry,
	}
}

// DefaultShouldRetry returns true for retryable errors.
func DefaultShouldRetry(err error) bool {
	if llmErr, ok := err.(*LLMError); ok {
		return llmErr.IsRetryable()
	}
	return false
}

// DelayForAttempt calculates the delay for a given attempt number (1-indexed).
func (rc RetryConfig) DelayForAttempt(attempt int) time.Duration {
	delay := float64(rc.InitialDelay) * math.Pow(rc.BackoffFactor, float64(attempt-1))
	if delay > float64(rc.MaxDelay) {
		delay = float64(rc.MaxDelay)
	}
	if rc.Jitter {
		jitter := 0.5 + rand.Float64()
		delay *= jitter
	}
	return time.Duration(delay)
}

// Retry executes fn with retry logic according to the config.
func Retry(ctx context.Context, config RetryConfig, fn func(ctx context.Context) (*Response, error)) (*Response, error) {
	shouldRetry := config.ShouldRetry
	if shouldRetry == nil {
		shouldRetry = DefaultShouldRetry
	}

	var lastErr error
	for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
		resp, err := fn(ctx)
		if err == nil {
			return resp, nil
		}
		lastErr = err

		if !shouldRetry(err) || attempt >= config.MaxAttempts {
			return nil, lastErr
		}

		// Use Retry-After header if available and within max delay
		delay := config.DelayForAttempt(attempt)
		if llmErr, ok := err.(*LLMError); ok && llmErr.RetryAfter > 0 {
			retryAfter := llmErr.RetryAfter
			if retryAfter <= config.MaxDelay {
				delay = retryAfter
			}
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}
	return nil, lastErr
}
