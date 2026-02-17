// Package events defines the typed event system for the Attractor pipeline engine.
package events

import (
	"sync"
	"time"
)

// EventType identifies the type of pipeline event.
type EventType string

const (
	// Pipeline lifecycle events
	EventPipelineStarted   EventType = "pipeline_started"
	EventPipelineCompleted EventType = "pipeline_completed"
	EventPipelineFailed    EventType = "pipeline_failed"

	// Stage lifecycle events
	EventStageStarted   EventType = "stage_started"
	EventStageCompleted EventType = "stage_completed"
	EventStageFailed    EventType = "stage_failed"
	EventStageRetrying  EventType = "stage_retrying"

	// Parallel execution events
	EventParallelStarted         EventType = "parallel_started"
	EventParallelBranchStarted   EventType = "parallel_branch_started"
	EventParallelBranchCompleted EventType = "parallel_branch_completed"
	EventParallelCompleted       EventType = "parallel_completed"

	// Human interaction events
	EventInterviewStarted   EventType = "interview_started"
	EventInterviewCompleted EventType = "interview_completed"
	EventInterviewTimeout   EventType = "interview_timeout"

	// Checkpoint events
	EventCheckpointSaved EventType = "checkpoint_saved"
)

// Event is a single pipeline event.
type Event struct {
	Type      EventType              `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

// NewEvent creates a new event with the current timestamp.
func NewEvent(eventType EventType, data map[string]interface{}) Event {
	return Event{
		Type:      eventType,
		Timestamp: time.Now(),
		Data:      data,
	}
}

// Emitter provides event emission and subscription.
type Emitter struct {
	mu        sync.RWMutex
	listeners []func(Event)
}

// NewEmitter creates a new event emitter.
func NewEmitter() *Emitter {
	return &Emitter{}
}

// On registers a listener for all events.
func (e *Emitter) On(listener func(Event)) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.listeners = append(e.listeners, listener)
}

// Emit sends an event to all listeners.
func (e *Emitter) Emit(event Event) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	for _, listener := range e.listeners {
		listener(event)
	}
}

// EmitPipelineStarted emits a pipeline started event.
func (e *Emitter) EmitPipelineStarted(name, id string) {
	e.Emit(NewEvent(EventPipelineStarted, map[string]interface{}{
		"name": name,
		"id":   id,
	}))
}

// EmitPipelineCompleted emits a pipeline completed event.
func (e *Emitter) EmitPipelineCompleted(duration time.Duration, artifactCount int) {
	e.Emit(NewEvent(EventPipelineCompleted, map[string]interface{}{
		"duration":       duration.String(),
		"artifact_count": artifactCount,
	}))
}

// EmitPipelineFailed emits a pipeline failed event.
func (e *Emitter) EmitPipelineFailed(errMsg string, duration time.Duration) {
	e.Emit(NewEvent(EventPipelineFailed, map[string]interface{}{
		"error":    errMsg,
		"duration": duration.String(),
	}))
}

// EmitStageStarted emits a stage started event.
func (e *Emitter) EmitStageStarted(name string, index int) {
	e.Emit(NewEvent(EventStageStarted, map[string]interface{}{
		"name":  name,
		"index": index,
	}))
}

// EmitStageCompleted emits a stage completed event.
func (e *Emitter) EmitStageCompleted(name string, index int, duration time.Duration) {
	e.Emit(NewEvent(EventStageCompleted, map[string]interface{}{
		"name":     name,
		"index":    index,
		"duration": duration.String(),
	}))
}

// EmitStageFailed emits a stage failed event.
func (e *Emitter) EmitStageFailed(name string, index int, errMsg string, willRetry bool) {
	e.Emit(NewEvent(EventStageFailed, map[string]interface{}{
		"name":       name,
		"index":      index,
		"error":      errMsg,
		"will_retry": willRetry,
	}))
}

// EmitStageRetrying emits a stage retrying event.
func (e *Emitter) EmitStageRetrying(name string, index, attempt int, delay time.Duration) {
	e.Emit(NewEvent(EventStageRetrying, map[string]interface{}{
		"name":    name,
		"index":   index,
		"attempt": attempt,
		"delay":   delay.String(),
	}))
}

// EmitCheckpointSaved emits a checkpoint saved event.
func (e *Emitter) EmitCheckpointSaved(nodeID string) {
	e.Emit(NewEvent(EventCheckpointSaved, map[string]interface{}{
		"node_id": nodeID,
	}))
}
