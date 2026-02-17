package pipeline

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/ashka-vakil/attractor/pkg/pipeline/events"
)

// Server exposes the pipeline engine as an HTTP service.
type Server struct {
	mu        sync.RWMutex
	resolver  HandlerResolver
	pipelines map[string]*pipelineRun
	emitter   *events.Emitter
}

type pipelineRun struct {
	ID        string      `json:"id"`
	Status    string      `json:"status"`
	Graph     *Graph      `json:"graph"`
	Result    *RunResult  `json:"result,omitempty"`
	Events    []events.Event `json:"events"`
	Questions []pendingQuestion `json:"questions,omitempty"`
	StartTime time.Time   `json:"start_time"`
	mu        sync.Mutex
}

type pendingQuestion struct {
	ID       string          `json:"id"`
	Question json.RawMessage `json:"question"`
}

// NewServer creates a new HTTP pipeline server.
func NewServer(resolver HandlerResolver) *Server {
	return &Server{
		resolver:  resolver,
		pipelines: make(map[string]*pipelineRun),
		emitter:   events.NewEmitter(),
	}
}

// Handler returns the HTTP handler for the server.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /pipelines", s.handleCreatePipeline)
	mux.HandleFunc("GET /pipelines/{id}", s.handleGetPipeline)
	mux.HandleFunc("GET /pipelines/{id}/events", s.handleGetEvents)
	mux.HandleFunc("POST /pipelines/{id}/cancel", s.handleCancelPipeline)
	mux.HandleFunc("GET /pipelines/{id}/context", s.handleGetContext)
	mux.HandleFunc("GET /pipelines/{id}/checkpoint", s.handleGetCheckpoint)
	mux.HandleFunc("GET /pipelines/{id}/questions", s.handleGetQuestions)
	mux.HandleFunc("POST /pipelines/{id}/questions/{qid}/answer", s.handleAnswerQuestion)
	return mux
}

func (s *Server) handleCreatePipeline(w http.ResponseWriter, r *http.Request) {
	var req struct {
		DOTSource string `json:"dot_source"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	graph, err := Parse(req.DOTSource)
	if err != nil {
		http.Error(w, fmt.Sprintf("parse error: %v", err), http.StatusBadRequest)
		return
	}

	if _, err := ValidateOrRaise(graph); err != nil {
		http.Error(w, fmt.Sprintf("validation error: %v", err), http.StatusBadRequest)
		return
	}

	id := fmt.Sprintf("pipeline-%d", time.Now().UnixNano())
	run := &pipelineRun{
		ID:        id,
		Status:    "running",
		Graph:     graph,
		StartTime: time.Now(),
	}

	s.mu.Lock()
	s.pipelines[id] = run
	s.mu.Unlock()

	// Run pipeline in background
	go func() {
		emitter := events.NewEmitter()
		emitter.On(func(e events.Event) {
			run.mu.Lock()
			run.Events = append(run.Events, e)
			run.mu.Unlock()
		})

		engine := NewEngine(EngineConfig{}, s.resolver, emitter)
		result, err := engine.Run(graph)

		run.mu.Lock()
		if err != nil {
			run.Status = "failed"
		} else {
			run.Result = result
			if result.Status == StatusSuccess {
				run.Status = "completed"
			} else {
				run.Status = "failed"
			}
		}
		run.mu.Unlock()
	}()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"id": id})
}

func (s *Server) handleGetPipeline(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.mu.RLock()
	run, ok := s.pipelines[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "pipeline not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":     run.ID,
		"status": run.Status,
		"result": run.Result,
	})
}

func (s *Server) handleGetEvents(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.mu.RLock()
	run, ok := s.pipelines[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "pipeline not found", http.StatusNotFound)
		return
	}

	// SSE response
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	run.mu.Lock()
	evts := make([]events.Event, len(run.Events))
	copy(evts, run.Events)
	run.mu.Unlock()

	for _, evt := range evts {
		data, _ := json.Marshal(evt)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func (s *Server) handleCancelPipeline(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.mu.RLock()
	run, ok := s.pipelines[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "pipeline not found", http.StatusNotFound)
		return
	}
	run.mu.Lock()
	run.Status = "cancelled"
	run.mu.Unlock()
	w.WriteHeader(http.StatusOK)
}

func (s *Server) handleGetContext(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.mu.RLock()
	run, ok := s.pipelines[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "pipeline not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if run.Result != nil {
		json.NewEncoder(w).Encode(run.Result.NodeOutcomes)
	} else {
		json.NewEncoder(w).Encode(map[string]interface{}{})
	}
}

func (s *Server) handleGetCheckpoint(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.mu.RLock()
	_, ok := s.pipelines[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "pipeline not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *Server) handleGetQuestions(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.mu.RLock()
	run, ok := s.pipelines[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "pipeline not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(run.Questions)
}

func (s *Server) handleAnswerQuestion(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.mu.RLock()
	_, ok := s.pipelines[id]
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "pipeline not found", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusOK)
}

// ListenAndServe starts the HTTP server.
func (s *Server) ListenAndServe(addr string) error {
	return http.ListenAndServe(addr, s.Handler())
}
