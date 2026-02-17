package pipeline

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/ashka-vakil/attractor/pkg/pipeline/events"
)

// Handler is the interface for node execution (mirrors handler package to avoid circular import).
type Handler interface {
	Execute(node *Node, ctx *Context, graph *Graph, logsRoot string) (*Outcome, error)
}

// HandlerResolver resolves the appropriate handler for a node.
type HandlerResolver interface {
	Resolve(node *Node) Handler
}

// EngineConfig configures the pipeline engine.
type EngineConfig struct {
	LogsRoot string
}

// Engine orchestrates pipeline execution.
type Engine struct {
	config         EngineConfig
	handlerResolver HandlerResolver
	emitter        *events.Emitter
}

// NewEngine creates a new pipeline engine.
func NewEngine(config EngineConfig, resolver HandlerResolver, emitter *events.Emitter) *Engine {
	if emitter == nil {
		emitter = events.NewEmitter()
	}
	return &Engine{
		config:          config,
		handlerResolver: resolver,
		emitter:         emitter,
	}
}

// RunResult is the final result of a pipeline run.
type RunResult struct {
	Status         StageStatus
	CompletedNodes []string
	FinalOutcome   *Outcome
	NodeOutcomes   map[string]*Outcome
}

// Run executes a pipeline graph.
func (e *Engine) Run(graph *Graph) (*RunResult, error) {
	startTime := time.Now()
	pipelineID := fmt.Sprintf("run-%d", time.Now().UnixNano())

	e.emitter.EmitPipelineStarted(graph.Name, pipelineID)

	ctx := NewContext()
	mirrorGraphAttributes(graph, ctx)

	var completedNodes []string
	nodeOutcomes := make(map[string]*Outcome)

	// Find start node
	startNode := e.findStartNode(graph)
	if startNode == nil {
		err := fmt.Errorf("no start node found")
		e.emitter.EmitPipelineFailed(err.Error(), time.Since(startTime))
		return nil, err
	}

	currentNode := startNode
	stageIndex := 0

	for {
		node := graph.Nodes[currentNode.ID]
		if node == nil {
			err := fmt.Errorf("node %q not found in graph", currentNode.ID)
			e.emitter.EmitPipelineFailed(err.Error(), time.Since(startTime))
			return nil, err
		}

		// Step 1: Check for terminal node
		if isTerminal(node) {
			gateOK, failedGate := checkGoalGates(graph, nodeOutcomes)
			if !gateOK && failedGate != nil {
				retryTarget := getRetryTarget(failedGate, graph)
				if retryTarget != "" {
					if targetNode, ok := graph.Nodes[retryTarget]; ok {
						currentNode = targetNode
						continue
					}
				}
				err := fmt.Errorf("goal gate %q unsatisfied and no retry target", failedGate.ID)
				e.emitter.EmitPipelineFailed(err.Error(), time.Since(startTime))
				return &RunResult{
					Status:         StatusFail,
					CompletedNodes: completedNodes,
					NodeOutcomes:   nodeOutcomes,
				}, nil
			}
			break
		}

		// Step 2: Execute node handler with retry
		e.emitter.EmitStageStarted(node.Label, stageIndex)
		stageStart := time.Now()

		retryPolicy := buildRetryPolicy(node, graph)
		outcome, err := e.executeWithRetry(node, ctx, graph, retryPolicy, stageIndex)
		if err != nil {
			e.emitter.EmitStageFailed(node.Label, stageIndex, err.Error(), false)
			e.emitter.EmitPipelineFailed(err.Error(), time.Since(startTime))
			return nil, err
		}

		stageDuration := time.Since(stageStart)
		if outcome.Status == StatusSuccess || outcome.Status == StatusPartialSuccess {
			e.emitter.EmitStageCompleted(node.Label, stageIndex, stageDuration)
		} else {
			e.emitter.EmitStageFailed(node.Label, stageIndex, outcome.FailureReason, false)
		}

		// Step 3: Record completion
		completedNodes = append(completedNodes, node.ID)
		nodeOutcomes[node.ID] = outcome

		// Step 4: Apply context updates
		ctx.ApplyUpdates(outcome.ContextUpdates)
		ctx.Set("outcome", string(outcome.Status))
		if outcome.PreferredLabel != "" {
			ctx.Set("preferred_label", outcome.PreferredLabel)
		}

		// Step 4b: Handle auto_status - write status.json if handler didn't
		if e.config.LogsRoot != "" && node.AutoStatus {
			statusPath := filepath.Join(e.config.LogsRoot, node.ID, "status.json")
			if _, err := os.Stat(statusPath); os.IsNotExist(err) {
				autoOutcome := &Outcome{
					Status: StatusSuccess,
					Notes:  "auto-status: handler completed without writing status",
				}
				data, _ := json.MarshalIndent(autoOutcome, "", "  ")
				os.MkdirAll(filepath.Dir(statusPath), 0o755)
				os.WriteFile(statusPath, data, 0o644)
			}
		}

		// Step 5: Save checkpoint
		cp := &Checkpoint{
			Timestamp:      time.Now(),
			CurrentNode:    node.ID,
			CompletedNodes: completedNodes,
			NodeRetries:    make(map[string]int),
			ContextValues:  ctx.Snapshot(),
			Logs:           ctx.Logs(),
		}
		if e.config.LogsRoot != "" {
			cp.Save(filepath.Join(e.config.LogsRoot, "checkpoint.json"))
			e.emitter.EmitCheckpointSaved(node.ID)
		}

		// Step 6: Select next edge
		nextEdge := selectEdge(node, outcome, ctx, graph)
		if nextEdge == nil {
			if outcome.Status == StatusFail {
				err := fmt.Errorf("stage %q failed with no outgoing fail edge", node.ID)
				e.emitter.EmitPipelineFailed(err.Error(), time.Since(startTime))
				return &RunResult{
					Status:         StatusFail,
					CompletedNodes: completedNodes,
					FinalOutcome:   outcome,
					NodeOutcomes:   nodeOutcomes,
				}, nil
			}
			break
		}

		// Step 7: Handle loop_restart
		if nextEdge.LoopRestart {
			// Restart the pipeline - for simplicity, we just continue from the target
			currentNode = graph.Nodes[nextEdge.To]
			continue
		}

		// Step 8: Advance to next node
		currentNode = graph.Nodes[nextEdge.To]
		stageIndex++
	}

	duration := time.Since(startTime)
	e.emitter.EmitPipelineCompleted(duration, len(completedNodes))

	finalStatus := StatusSuccess
	for _, outcome := range nodeOutcomes {
		if outcome.Status == StatusFail {
			finalStatus = StatusFail
			break
		}
	}

	return &RunResult{
		Status:         finalStatus,
		CompletedNodes: completedNodes,
		NodeOutcomes:   nodeOutcomes,
	}, nil
}

func (e *Engine) findStartNode(graph *Graph) *Node {
	for _, node := range graph.Nodes {
		if node.Shape == "Mdiamond" {
			return node
		}
	}
	for _, node := range graph.Nodes {
		if node.ID == "start" || node.ID == "Start" {
			return node
		}
	}
	return nil
}

func isTerminal(node *Node) bool {
	return node.Shape == "Msquare"
}

func checkGoalGates(graph *Graph, nodeOutcomes map[string]*Outcome) (bool, *Node) {
	for nodeID, outcome := range nodeOutcomes {
		node := graph.Nodes[nodeID]
		if node != nil && node.GoalGate {
			if outcome.Status != StatusSuccess && outcome.Status != StatusPartialSuccess {
				return false, node
			}
		}
	}
	return true, nil
}

func getRetryTarget(node *Node, graph *Graph) string {
	if node.RetryTarget != "" {
		return node.RetryTarget
	}
	if node.FallbackRetryTarget != "" {
		return node.FallbackRetryTarget
	}
	if graph.RetryTarget != "" {
		return graph.RetryTarget
	}
	if graph.FallbackRetryTarget != "" {
		return graph.FallbackRetryTarget
	}
	return ""
}

// RetryPolicy controls retry behavior.
type RetryPolicy struct {
	MaxAttempts    int
	InitialDelay  time.Duration
	BackoffFactor float64
	MaxDelay      time.Duration
	Jitter        bool
}

func buildRetryPolicy(node *Node, graph *Graph) RetryPolicy {
	maxRetries := node.MaxRetries
	if maxRetries == 0 {
		maxRetries = graph.DefaultMaxRetry
	}

	return RetryPolicy{
		MaxAttempts:    maxRetries + 1,
		InitialDelay:  200 * time.Millisecond,
		BackoffFactor: 2.0,
		MaxDelay:      60 * time.Second,
		Jitter:        true,
	}
}

func (e *Engine) executeWithRetry(node *Node, ctx *Context, graph *Graph, policy RetryPolicy, stageIndex int) (*Outcome, error) {
	handler := e.handlerResolver.Resolve(node)
	if handler == nil {
		return &Outcome{
			Status:        StatusFail,
			FailureReason: fmt.Sprintf("no handler found for node %q", node.ID),
		}, nil
	}

	maxAttempts := policy.MaxAttempts
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		outcome, err := handler.Execute(node, ctx, graph, e.config.LogsRoot)
		if err != nil {
			if attempt < maxAttempts {
				delay := delayForAttempt(attempt, policy)
				e.emitter.EmitStageRetrying(node.Label, stageIndex, attempt, delay)
				time.Sleep(delay)
				continue
			}
			return &Outcome{
				Status:        StatusFail,
				FailureReason: err.Error(),
			}, nil
		}

		if outcome.Status == StatusSuccess || outcome.Status == StatusPartialSuccess {
			return outcome, nil
		}

		if outcome.Status == StatusRetry {
			if attempt < maxAttempts {
				delay := delayForAttempt(attempt, policy)
				e.emitter.EmitStageRetrying(node.Label, stageIndex, attempt, delay)
				time.Sleep(delay)
				continue
			}
			if node.AllowPartial {
				return &Outcome{
					Status: StatusPartialSuccess,
					Notes:  "retries exhausted, partial accepted",
				}, nil
			}
			return &Outcome{
				Status:        StatusFail,
				FailureReason: "max retries exceeded",
			}, nil
		}

		if outcome.Status == StatusFail {
			return outcome, nil
		}
	}

	return &Outcome{
		Status:        StatusFail,
		FailureReason: "max retries exceeded",
	}, nil
}

func delayForAttempt(attempt int, policy RetryPolicy) time.Duration {
	delay := float64(policy.InitialDelay) * math.Pow(policy.BackoffFactor, float64(attempt-1))
	if delay > float64(policy.MaxDelay) {
		delay = float64(policy.MaxDelay)
	}
	if policy.Jitter {
		delay *= 0.5 + rand.Float64()
	}
	return time.Duration(delay)
}

// selectEdge implements the 5-step edge selection algorithm.
func selectEdge(node *Node, outcome *Outcome, ctx *Context, graph *Graph) *Edge {
	edges := graph.OutgoingEdges(node.ID)
	if len(edges) == 0 {
		return nil
	}

	// Step 1: Condition matching
	var conditionMatched []*Edge
	for _, edge := range edges {
		if edge.Condition != "" {
			if evaluateConditionSimple(edge.Condition, outcome, ctx) {
				conditionMatched = append(conditionMatched, edge)
			}
		}
	}
	if len(conditionMatched) > 0 {
		return bestByWeightThenLexical(conditionMatched)
	}

	// Step 2: Preferred label
	if outcome != nil && outcome.PreferredLabel != "" {
		for _, edge := range edges {
			if normalizeLabel(edge.Label) == normalizeLabel(outcome.PreferredLabel) {
				return edge
			}
		}
	}

	// Step 3: Suggested next IDs
	if outcome != nil && len(outcome.SuggestedNextIDs) > 0 {
		for _, suggestedID := range outcome.SuggestedNextIDs {
			for _, edge := range edges {
				if edge.To == suggestedID {
					return edge
				}
			}
		}
	}

	// Step 4 & 5: Weight with lexical tiebreak (unconditional edges only)
	var unconditional []*Edge
	for _, edge := range edges {
		if edge.Condition == "" {
			unconditional = append(unconditional, edge)
		}
	}
	if len(unconditional) > 0 {
		return bestByWeightThenLexical(unconditional)
	}

	// Fallback: any edge
	return bestByWeightThenLexical(edges)
}

func bestByWeightThenLexical(edges []*Edge) *Edge {
	if len(edges) == 0 {
		return nil
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].Weight != edges[j].Weight {
			return edges[i].Weight > edges[j].Weight
		}
		return edges[i].To < edges[j].To
	})
	return edges[0]
}

func normalizeLabel(label string) string {
	label = strings.ToLower(strings.TrimSpace(label))
	// Strip accelerator prefixes: [Y] , Y) , Y -
	if len(label) > 3 {
		if label[0] == '[' {
			if idx := strings.Index(label, "] "); idx >= 0 {
				label = label[idx+2:]
			}
		} else if len(label) > 2 && label[1] == ')' && label[2] == ' ' {
			label = label[3:]
		} else if len(label) > 3 && label[1] == ' ' && label[2] == '-' && label[3] == ' ' {
			label = label[4:]
		}
	}
	return strings.TrimSpace(label)
}

// evaluateConditionSimple is a simple inline condition evaluator to avoid circular imports.
func evaluateConditionSimple(condition string, outcome *Outcome, ctx *Context) bool {
	condition = strings.TrimSpace(condition)
	if condition == "" {
		return true
	}

	clauses := strings.Split(condition, "&&")
	for _, clause := range clauses {
		clause = strings.TrimSpace(clause)
		if clause == "" {
			continue
		}
		if !evaluateClauseSimple(clause, outcome, ctx) {
			return false
		}
	}
	return true
}

func evaluateClauseSimple(clause string, outcome *Outcome, ctx *Context) bool {
	if idx := strings.Index(clause, "!="); idx >= 0 {
		key := strings.TrimSpace(clause[:idx])
		value := strings.TrimSpace(clause[idx+2:])
		return resolveKeySimple(key, outcome, ctx) != value
	}

	if idx := strings.Index(clause, "="); idx >= 0 {
		key := strings.TrimSpace(clause[:idx])
		value := strings.TrimSpace(clause[idx+1:])
		return resolveKeySimple(key, outcome, ctx) == value
	}

	resolved := resolveKeySimple(strings.TrimSpace(clause), outcome, ctx)
	return resolved != "" && resolved != "false" && resolved != "0"
}

func resolveKeySimple(key string, outcome *Outcome, ctx *Context) string {
	switch key {
	case "outcome":
		if outcome == nil {
			return ""
		}
		return string(outcome.Status)
	case "preferred_label":
		if outcome == nil {
			return ""
		}
		return outcome.PreferredLabel
	}

	if strings.HasPrefix(key, "context.") {
		if v, ok := ctx.Get(key); ok {
			return fmt.Sprint(v)
		}
		stripped := strings.TrimPrefix(key, "context.")
		if v, ok := ctx.Get(stripped); ok {
			return fmt.Sprint(v)
		}
		return ""
	}

	if v, ok := ctx.Get(key); ok {
		return fmt.Sprint(v)
	}
	return ""
}

func mirrorGraphAttributes(graph *Graph, ctx *Context) {
	ctx.Set("graph.goal", graph.Goal)
	if graph.Label != "" {
		ctx.Set("graph.label", graph.Label)
	}
}
