// Package agent implements the coding agent loop.
package agent

import (
	"time"

	"github.com/ashka-vakil/attractor/pkg/llm"
)

// SessionState represents the lifecycle state of a session.
type SessionState string

const (
	StateIdle          SessionState = "idle"
	StateProcessing    SessionState = "processing"
	StateAwaitingInput SessionState = "awaiting_input"
	StateClosed        SessionState = "closed"
)

// SessionConfig configures the agent session.
type SessionConfig struct {
	MaxTurns                int               `json:"max_turns"`
	MaxToolRoundsPerInput   int               `json:"max_tool_rounds_per_input"`
	DefaultCommandTimeoutMs int               `json:"default_command_timeout_ms"`
	MaxCommandTimeoutMs     int               `json:"max_command_timeout_ms"`
	ReasoningEffort         string            `json:"reasoning_effort,omitempty"`
	ToolOutputLimits        map[string]int    `json:"tool_output_limits,omitempty"`
	EnableLoopDetection     bool              `json:"enable_loop_detection"`
	LoopDetectionWindow     int               `json:"loop_detection_window"`
	MaxSubagentDepth        int               `json:"max_subagent_depth"`
}

// DefaultSessionConfig returns the default session configuration.
func DefaultSessionConfig() SessionConfig {
	return SessionConfig{
		MaxTurns:                0,
		MaxToolRoundsPerInput:   0,
		DefaultCommandTimeoutMs: 10000,
		MaxCommandTimeoutMs:     600000,
		EnableLoopDetection:     true,
		LoopDetectionWindow:     10,
		MaxSubagentDepth:        1,
	}
}

// Turn represents a single entry in the conversation history.
type Turn interface {
	turnType() string
}

// UserTurn is a user message.
type UserTurn struct {
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

func (t *UserTurn) turnType() string { return "user" }

// SteeringTurn is a mid-task steering message injected by the host.
type SteeringTurn struct {
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

func (t *SteeringTurn) turnType() string { return "steering" }

// AssistantTurn is an assistant response.
type AssistantTurn struct {
	Content    string         `json:"content"`
	ToolCalls  []llm.ToolCall `json:"tool_calls,omitempty"`
	Reasoning  string         `json:"reasoning,omitempty"`
	Usage      llm.Usage      `json:"usage"`
	ResponseID string         `json:"response_id,omitempty"`
	Timestamp  time.Time      `json:"timestamp"`
}

func (t *AssistantTurn) turnType() string { return "assistant" }

// ToolResultsTurn contains results from tool executions.
type ToolResultsTurn struct {
	Results   []llm.ToolResult `json:"results"`
	Timestamp time.Time        `json:"timestamp"`
}

func (t *ToolResultsTurn) turnType() string { return "tool_results" }

// EventType identifies the type of agent event.
type EventType string

const (
	EventSessionStarted    EventType = "session_started"
	EventTurnStarted       EventType = "turn_started"
	EventTurnCompleted     EventType = "turn_completed"
	EventToolCallStarted   EventType = "tool_call_started"
	EventToolCallCompleted EventType = "tool_call_completed"
	EventTextDelta         EventType = "text_delta"
	EventReasoningDelta    EventType = "reasoning_delta"
	EventError             EventType = "error"
	EventSessionClosed     EventType = "session_closed"
	EventLoopDetected      EventType = "loop_detected"
	EventSteeringApplied   EventType = "steering_applied"
)

// Event is a single agent event.
type Event struct {
	Type      EventType              `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

// EventEmitter delivers events to the host application.
type EventEmitter struct {
	listeners []func(Event)
}

// NewEventEmitter creates a new event emitter.
func NewEventEmitter() *EventEmitter {
	return &EventEmitter{}
}

// On registers an event listener.
func (e *EventEmitter) On(listener func(Event)) {
	e.listeners = append(e.listeners, listener)
}

// Emit sends an event to all listeners.
func (e *EventEmitter) Emit(event Event) {
	for _, l := range e.listeners {
		l(event)
	}
}
