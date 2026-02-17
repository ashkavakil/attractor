package events

import (
	"testing"
	"time"
)

func TestNewEvent(t *testing.T) {
	evt := NewEvent(EventPipelineStarted, map[string]interface{}{
		"name": "test",
	})
	if evt.Type != EventPipelineStarted {
		t.Errorf("expected pipeline_started, got %s", evt.Type)
	}
	if evt.Data["name"] != "test" {
		t.Errorf("expected name=test, got %v", evt.Data["name"])
	}
	if evt.Timestamp.IsZero() {
		t.Error("expected non-zero timestamp")
	}
}

func TestEmitter(t *testing.T) {
	emitter := NewEmitter()
	var received []Event

	emitter.On(func(e Event) {
		received = append(received, e)
	})

	emitter.Emit(NewEvent(EventStageStarted, nil))
	emitter.Emit(NewEvent(EventStageCompleted, nil))

	if len(received) != 2 {
		t.Errorf("expected 2 events, got %d", len(received))
	}
}

func TestEmitPipelineStarted(t *testing.T) {
	emitter := NewEmitter()
	var received Event

	emitter.On(func(e Event) { received = e })
	emitter.EmitPipelineStarted("test", "id-1")

	if received.Type != EventPipelineStarted {
		t.Errorf("expected pipeline_started, got %s", received.Type)
	}
	if received.Data["name"] != "test" {
		t.Errorf("expected name=test")
	}
}

func TestEmitPipelineCompleted(t *testing.T) {
	emitter := NewEmitter()
	var received Event

	emitter.On(func(e Event) { received = e })
	emitter.EmitPipelineCompleted(5*time.Second, 3)

	if received.Type != EventPipelineCompleted {
		t.Errorf("expected pipeline_completed, got %s", received.Type)
	}
}

func TestEmitStageFailed(t *testing.T) {
	emitter := NewEmitter()
	var received Event

	emitter.On(func(e Event) { received = e })
	emitter.EmitStageFailed("test_stage", 0, "something broke", true)

	if received.Type != EventStageFailed {
		t.Errorf("expected stage_failed, got %s", received.Type)
	}
	if received.Data["will_retry"] != true {
		t.Error("expected will_retry=true")
	}
}

func TestMultipleListeners(t *testing.T) {
	emitter := NewEmitter()
	count1, count2 := 0, 0

	emitter.On(func(e Event) { count1++ })
	emitter.On(func(e Event) { count2++ })

	emitter.Emit(NewEvent(EventCheckpointSaved, nil))

	if count1 != 1 || count2 != 1 {
		t.Errorf("expected both listeners called, got %d and %d", count1, count2)
	}
}
