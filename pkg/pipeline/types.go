// Package pipeline implements the Attractor DOT-based pipeline engine.
package pipeline

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// StageStatus represents the outcome status of a node handler.
type StageStatus string

const (
	StatusSuccess        StageStatus = "success"
	StatusPartialSuccess StageStatus = "partial_success"
	StatusRetry          StageStatus = "retry"
	StatusFail           StageStatus = "fail"
	StatusSkipped        StageStatus = "skipped"
)

// Outcome is the result of executing a node handler.
type Outcome struct {
	Status           StageStatus        `json:"outcome"`
	PreferredLabel   string             `json:"preferred_next_label,omitempty"`
	SuggestedNextIDs []string           `json:"suggested_next_ids,omitempty"`
	ContextUpdates   map[string]interface{} `json:"context_updates,omitempty"`
	Notes            string             `json:"notes,omitempty"`
	FailureReason    string             `json:"failure_reason,omitempty"`
}

// Node represents a node in the pipeline graph.
type Node struct {
	ID                  string            `json:"id"`
	Label               string            `json:"label,omitempty"`
	Shape               string            `json:"shape,omitempty"`
	Type                string            `json:"type,omitempty"`
	Prompt              string            `json:"prompt,omitempty"`
	MaxRetries          int               `json:"max_retries,omitempty"`
	GoalGate            bool              `json:"goal_gate,omitempty"`
	RetryTarget         string            `json:"retry_target,omitempty"`
	FallbackRetryTarget string            `json:"fallback_retry_target,omitempty"`
	Fidelity            string            `json:"fidelity,omitempty"`
	ThreadID            string            `json:"thread_id,omitempty"`
	Class               string            `json:"class,omitempty"`
	Timeout             time.Duration     `json:"timeout,omitempty"`
	LLMModel            string            `json:"llm_model,omitempty"`
	LLMProvider         string            `json:"llm_provider,omitempty"`
	ReasoningEffort     string            `json:"reasoning_effort,omitempty"`
	AutoStatus          bool              `json:"auto_status,omitempty"`
	AllowPartial        bool              `json:"allow_partial,omitempty"`
	Attrs               map[string]string `json:"attrs,omitempty"`
}

// Edge represents a directed edge in the pipeline graph.
type Edge struct {
	From        string `json:"from"`
	To          string `json:"to"`
	Label       string `json:"label,omitempty"`
	Condition   string `json:"condition,omitempty"`
	Weight      int    `json:"weight,omitempty"`
	Fidelity    string `json:"fidelity,omitempty"`
	ThreadID    string `json:"thread_id,omitempty"`
	LoopRestart bool   `json:"loop_restart,omitempty"`
}

// Graph is the complete pipeline graph.
type Graph struct {
	Name                 string            `json:"name"`
	Goal                 string            `json:"goal,omitempty"`
	Label                string            `json:"label,omitempty"`
	ModelStylesheet      string            `json:"model_stylesheet,omitempty"`
	DefaultMaxRetry      int               `json:"default_max_retry,omitempty"`
	DefaultFidelity      string            `json:"default_fidelity,omitempty"`
	RetryTarget          string            `json:"retry_target,omitempty"`
	FallbackRetryTarget  string            `json:"fallback_retry_target,omitempty"`
	Nodes                map[string]*Node  `json:"nodes"`
	Edges                []*Edge           `json:"edges"`
	Attrs                map[string]string `json:"attrs,omitempty"`
}

// OutgoingEdges returns all edges originating from the given node ID.
func (g *Graph) OutgoingEdges(nodeID string) []*Edge {
	var edges []*Edge
	for _, e := range g.Edges {
		if e.From == nodeID {
			edges = append(edges, e)
		}
	}
	return edges
}

// IncomingEdges returns all edges targeting the given node ID.
func (g *Graph) IncomingEdges(nodeID string) []*Edge {
	var edges []*Edge
	for _, e := range g.Edges {
		if e.To == nodeID {
			edges = append(edges, e)
		}
	}
	return edges
}

// Context is a thread-safe key-value store for pipeline state.
type Context struct {
	mu     sync.RWMutex
	values map[string]interface{}
	logs   []string
}

// NewContext creates a new empty context.
func NewContext() *Context {
	return &Context{
		values: make(map[string]interface{}),
	}
}

// Set stores a value in the context.
func (c *Context) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.values[key] = value
}

// Get retrieves a value from the context.
func (c *Context) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.values[key]
	return v, ok
}

// GetString retrieves a string value from the context.
func (c *Context) GetString(key string) string {
	v, ok := c.Get(key)
	if !ok {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return s
}

// AppendLog adds a log entry.
func (c *Context) AppendLog(entry string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.logs = append(c.logs, entry)
}

// Snapshot returns a serializable copy of all values.
func (c *Context) Snapshot() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()
	result := make(map[string]interface{}, len(c.values))
	for k, v := range c.values {
		result[k] = v
	}
	return result
}

// Clone creates a deep copy of the context.
func (c *Context) Clone() *Context {
	c.mu.RLock()
	defer c.mu.RUnlock()
	nc := NewContext()
	for k, v := range c.values {
		nc.values[k] = v
	}
	nc.logs = make([]string, len(c.logs))
	copy(nc.logs, c.logs)
	return nc
}

// ApplyUpdates merges a map of updates into the context.
func (c *Context) ApplyUpdates(updates map[string]interface{}) {
	if updates == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, v := range updates {
		c.values[k] = v
	}
}

// Logs returns a copy of the log entries.
func (c *Context) Logs() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	result := make([]string, len(c.logs))
	copy(result, c.logs)
	return result
}

// Checkpoint is a serializable snapshot of execution state.
type Checkpoint struct {
	Timestamp      time.Time              `json:"timestamp"`
	CurrentNode    string                 `json:"current_node"`
	CompletedNodes []string               `json:"completed_nodes"`
	NodeRetries    map[string]int         `json:"node_retries"`
	ContextValues  map[string]interface{} `json:"context"`
	Logs           []string               `json:"logs"`
}

// Save writes the checkpoint to a JSON file.
func (cp *Checkpoint) Save(path string) error {
	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return err
	}
	return writeFile(path, data)
}

// LoadCheckpoint reads a checkpoint from a JSON file.
func LoadCheckpoint(path string) (*Checkpoint, error) {
	data, err := readFile(path)
	if err != nil {
		return nil, err
	}
	var cp Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return nil, err
	}
	return &cp, nil
}

// ArtifactInfo describes a stored artifact.
type ArtifactInfo struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	SizeBytes    int       `json:"size_bytes"`
	StoredAt     time.Time `json:"stored_at"`
	IsFileBacked bool      `json:"is_file_backed"`
}

// ArtifactStore provides named, typed storage for large outputs.
type ArtifactStore struct {
	mu        sync.RWMutex
	artifacts map[string]*artifactEntry
	baseDir   string
}

type artifactEntry struct {
	info ArtifactInfo
	data interface{}
}

// NewArtifactStore creates a new artifact store.
func NewArtifactStore(baseDir string) *ArtifactStore {
	return &ArtifactStore{
		artifacts: make(map[string]*artifactEntry),
		baseDir:   baseDir,
	}
}

const fileBackingThreshold = 100 * 1024 // 100KB

// Store saves an artifact.
func (as *ArtifactStore) Store(artifactID, name string, data interface{}) ArtifactInfo {
	as.mu.Lock()
	defer as.mu.Unlock()

	serialized, _ := json.Marshal(data)
	size := len(serialized)
	isFileBacked := size > fileBackingThreshold && as.baseDir != ""

	info := ArtifactInfo{
		ID:           artifactID,
		Name:         name,
		SizeBytes:    size,
		StoredAt:     time.Now(),
		IsFileBacked: isFileBacked,
	}

	var stored interface{} = data
	if isFileBacked {
		path := as.baseDir + "/artifacts/" + artifactID + ".json"
		writeFile(path, serialized)
		stored = path
	}

	as.artifacts[artifactID] = &artifactEntry{info: info, data: stored}
	return info
}

// Retrieve loads an artifact by ID.
func (as *ArtifactStore) Retrieve(artifactID string) (interface{}, error) {
	as.mu.RLock()
	defer as.mu.RUnlock()
	entry, ok := as.artifacts[artifactID]
	if !ok {
		return nil, fmt.Errorf("artifact not found: %s", artifactID)
	}
	if entry.info.IsFileBacked {
		path, ok := entry.data.(string)
		if !ok {
			return nil, fmt.Errorf("invalid file-backed artifact path")
		}
		data, err := readFile(path)
		if err != nil {
			return nil, err
		}
		var result interface{}
		json.Unmarshal(data, &result)
		return result, nil
	}
	return entry.data, nil
}

// Has returns whether an artifact exists.
func (as *ArtifactStore) Has(artifactID string) bool {
	as.mu.RLock()
	defer as.mu.RUnlock()
	_, ok := as.artifacts[artifactID]
	return ok
}

// List returns info for all artifacts.
func (as *ArtifactStore) List() []ArtifactInfo {
	as.mu.RLock()
	defer as.mu.RUnlock()
	result := make([]ArtifactInfo, 0, len(as.artifacts))
	for _, entry := range as.artifacts {
		result = append(result, entry.info)
	}
	return result
}
