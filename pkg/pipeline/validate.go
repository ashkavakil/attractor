package pipeline

import (
	"fmt"
	"strings"
)

// Severity levels for diagnostics.
type Severity int

const (
	SeverityError   Severity = iota
	SeverityWarning
	SeverityInfo
)

func (s Severity) String() string {
	switch s {
	case SeverityError:
		return "ERROR"
	case SeverityWarning:
		return "WARNING"
	case SeverityInfo:
		return "INFO"
	default:
		return "UNKNOWN"
	}
}

// Diagnostic is a single validation finding.
type Diagnostic struct {
	Rule     string   `json:"rule"`
	Severity Severity `json:"severity"`
	Message  string   `json:"message"`
	NodeID   string   `json:"node_id,omitempty"`
	Edge     *[2]string `json:"edge,omitempty"`
	Fix      string   `json:"fix,omitempty"`
}

func (d Diagnostic) String() string {
	loc := ""
	if d.NodeID != "" {
		loc = fmt.Sprintf(" (node: %s)", d.NodeID)
	}
	if d.Edge != nil {
		loc = fmt.Sprintf(" (edge: %s -> %s)", d.Edge[0], d.Edge[1])
	}
	return fmt.Sprintf("[%s] %s: %s%s", d.Severity, d.Rule, d.Message, loc)
}

// LintRule is the interface for custom lint rules.
type LintRule interface {
	Name() string
	Apply(graph *Graph) []Diagnostic
}

// ValidationError is returned when validation fails.
type ValidationError struct {
	Diagnostics []Diagnostic
}

func (e *ValidationError) Error() string {
	var msgs []string
	for _, d := range e.Diagnostics {
		if d.Severity == SeverityError {
			msgs = append(msgs, d.String())
		}
	}
	return fmt.Sprintf("validation failed with %d error(s):\n%s", len(msgs), strings.Join(msgs, "\n"))
}

// Validate runs all lint rules and returns diagnostics.
func Validate(graph *Graph, extraRules ...LintRule) []Diagnostic {
	var diagnostics []Diagnostic

	// Built-in rules
	diagnostics = append(diagnostics, ruleStartNode(graph)...)
	diagnostics = append(diagnostics, ruleTerminalNode(graph)...)
	diagnostics = append(diagnostics, ruleStartNoIncoming(graph)...)
	diagnostics = append(diagnostics, ruleExitNoOutgoing(graph)...)
	diagnostics = append(diagnostics, ruleReachability(graph)...)
	diagnostics = append(diagnostics, ruleEdgeTargetExists(graph)...)
	diagnostics = append(diagnostics, ruleConditionSyntax(graph)...)
	diagnostics = append(diagnostics, ruleStylesheetSyntax(graph)...)
	diagnostics = append(diagnostics, ruleTypeKnown(graph)...)
	diagnostics = append(diagnostics, ruleFidelityValid(graph)...)
	diagnostics = append(diagnostics, ruleRetryTargetExists(graph)...)
	diagnostics = append(diagnostics, ruleGoalGateHasRetry(graph)...)
	diagnostics = append(diagnostics, rulePromptOnLLMNodes(graph)...)

	// Custom rules
	for _, rule := range extraRules {
		diagnostics = append(diagnostics, rule.Apply(graph)...)
	}

	return diagnostics
}

// ValidateOrRaise runs validation and returns an error if any error-severity diagnostics exist.
func ValidateOrRaise(graph *Graph, extraRules ...LintRule) ([]Diagnostic, error) {
	diagnostics := Validate(graph, extraRules...)
	var errors []Diagnostic
	for _, d := range diagnostics {
		if d.Severity == SeverityError {
			errors = append(errors, d)
		}
	}
	if len(errors) > 0 {
		return diagnostics, &ValidationError{Diagnostics: errors}
	}
	return diagnostics, nil
}

// --- Built-in lint rules ---

func findStartNode(graph *Graph) *Node {
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

func findExitNode(graph *Graph) *Node {
	for _, node := range graph.Nodes {
		if node.Shape == "Msquare" {
			return node
		}
	}
	for _, node := range graph.Nodes {
		if node.ID == "exit" || node.ID == "end" || node.ID == "done" {
			return node
		}
	}
	return nil
}

func ruleStartNode(graph *Graph) []Diagnostic {
	count := 0
	for _, node := range graph.Nodes {
		if node.Shape == "Mdiamond" {
			count++
		}
	}
	if count == 0 {
		return []Diagnostic{{
			Rule:     "start_node",
			Severity: SeverityError,
			Message:  "Pipeline must have exactly one start node (shape=Mdiamond)",
			Fix:      "Add a node with shape=Mdiamond",
		}}
	}
	if count > 1 {
		return []Diagnostic{{
			Rule:     "start_node",
			Severity: SeverityError,
			Message:  fmt.Sprintf("Pipeline has %d start nodes but must have exactly one", count),
		}}
	}
	return nil
}

func ruleTerminalNode(graph *Graph) []Diagnostic {
	count := 0
	for _, node := range graph.Nodes {
		if node.Shape == "Msquare" {
			count++
		}
	}
	if count == 0 {
		return []Diagnostic{{
			Rule:     "terminal_node",
			Severity: SeverityError,
			Message:  "Pipeline must have at least one terminal node (shape=Msquare)",
			Fix:      "Add a node with shape=Msquare",
		}}
	}
	return nil
}

func ruleStartNoIncoming(graph *Graph) []Diagnostic {
	start := findStartNode(graph)
	if start == nil {
		return nil
	}
	for _, e := range graph.Edges {
		if e.To == start.ID {
			edge := [2]string{e.From, e.To}
			return []Diagnostic{{
				Rule:     "start_no_incoming",
				Severity: SeverityError,
				Message:  "Start node must have no incoming edges",
				NodeID:   start.ID,
				Edge:     &edge,
			}}
		}
	}
	return nil
}

func ruleExitNoOutgoing(graph *Graph) []Diagnostic {
	exit := findExitNode(graph)
	if exit == nil {
		return nil
	}
	for _, e := range graph.Edges {
		if e.From == exit.ID {
			edge := [2]string{e.From, e.To}
			return []Diagnostic{{
				Rule:     "exit_no_outgoing",
				Severity: SeverityError,
				Message:  "Exit node must have no outgoing edges",
				NodeID:   exit.ID,
				Edge:     &edge,
			}}
		}
	}
	return nil
}

func ruleReachability(graph *Graph) []Diagnostic {
	start := findStartNode(graph)
	if start == nil {
		return nil
	}

	// BFS from start
	visited := make(map[string]bool)
	queue := []string{start.ID}
	visited[start.ID] = true

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, e := range graph.Edges {
			if e.From == current && !visited[e.To] {
				visited[e.To] = true
				queue = append(queue, e.To)
			}
		}
	}

	var diagnostics []Diagnostic
	for _, node := range graph.Nodes {
		if !visited[node.ID] {
			diagnostics = append(diagnostics, Diagnostic{
				Rule:     "reachability",
				Severity: SeverityError,
				Message:  fmt.Sprintf("Node %q is not reachable from the start node", node.ID),
				NodeID:   node.ID,
			})
		}
	}
	return diagnostics
}

func ruleEdgeTargetExists(graph *Graph) []Diagnostic {
	var diagnostics []Diagnostic
	for _, e := range graph.Edges {
		if _, ok := graph.Nodes[e.From]; !ok {
			edge := [2]string{e.From, e.To}
			diagnostics = append(diagnostics, Diagnostic{
				Rule:     "edge_target_exists",
				Severity: SeverityError,
				Message:  fmt.Sprintf("Edge source %q does not exist", e.From),
				Edge:     &edge,
			})
		}
		if _, ok := graph.Nodes[e.To]; !ok {
			edge := [2]string{e.From, e.To}
			diagnostics = append(diagnostics, Diagnostic{
				Rule:     "edge_target_exists",
				Severity: SeverityError,
				Message:  fmt.Sprintf("Edge target %q does not exist", e.To),
				Edge:     &edge,
			})
		}
	}
	return diagnostics
}

func ruleConditionSyntax(graph *Graph) []Diagnostic {
	var diagnostics []Diagnostic
	for _, e := range graph.Edges {
		if e.Condition != "" {
			if err := validateConditionSyntax(e.Condition); err != nil {
				edge := [2]string{e.From, e.To}
				diagnostics = append(diagnostics, Diagnostic{
					Rule:     "condition_syntax",
					Severity: SeverityError,
					Message:  fmt.Sprintf("Invalid condition expression: %v", err),
					Edge:     &edge,
				})
			}
		}
	}
	return diagnostics
}

func validateConditionSyntax(condition string) error {
	clauses := strings.Split(condition, "&&")
	for _, clause := range clauses {
		clause = strings.TrimSpace(clause)
		if clause == "" {
			continue
		}
		// Each clause must have at least a key
		if strings.Contains(clause, "!=") {
			parts := strings.SplitN(clause, "!=", 2)
			if strings.TrimSpace(parts[0]) == "" {
				return fmt.Errorf("empty key in clause: %q", clause)
			}
		} else if strings.Contains(clause, "=") {
			parts := strings.SplitN(clause, "=", 2)
			if strings.TrimSpace(parts[0]) == "" {
				return fmt.Errorf("empty key in clause: %q", clause)
			}
		}
	}
	return nil
}

func ruleStylesheetSyntax(graph *Graph) []Diagnostic {
	if graph.ModelStylesheet == "" {
		return nil
	}
	// Simple validation: check for balanced braces and valid structure
	ss := graph.ModelStylesheet
	braceCount := 0
	for _, ch := range ss {
		if ch == '{' {
			braceCount++
		}
		if ch == '}' {
			braceCount--
		}
		if braceCount < 0 {
			return []Diagnostic{{
				Rule:     "stylesheet_syntax",
				Severity: SeverityError,
				Message:  "model_stylesheet has unbalanced braces",
			}}
		}
	}
	if braceCount != 0 {
		return []Diagnostic{{
			Rule:     "stylesheet_syntax",
			Severity: SeverityError,
			Message:  "model_stylesheet has unbalanced braces",
		}}
	}
	return nil
}

var knownHandlerTypes = map[string]bool{
	"start": true, "exit": true, "codergen": true,
	"wait.human": true, "conditional": true,
	"parallel": true, "parallel.fan_in": true,
	"tool": true, "stack.manager_loop": true,
}

func ruleTypeKnown(graph *Graph) []Diagnostic {
	var diagnostics []Diagnostic
	for _, node := range graph.Nodes {
		if node.Type != "" && !knownHandlerTypes[node.Type] {
			diagnostics = append(diagnostics, Diagnostic{
				Rule:     "type_known",
				Severity: SeverityWarning,
				Message:  fmt.Sprintf("Node type %q is not a recognized handler type", node.Type),
				NodeID:   node.ID,
			})
		}
	}
	return diagnostics
}

var validFidelityModes = map[string]bool{
	"full": true, "truncate": true, "compact": true,
	"summary:low": true, "summary:medium": true, "summary:high": true,
}

func ruleFidelityValid(graph *Graph) []Diagnostic {
	var diagnostics []Diagnostic
	for _, node := range graph.Nodes {
		if node.Fidelity != "" && !validFidelityModes[node.Fidelity] {
			diagnostics = append(diagnostics, Diagnostic{
				Rule:     "fidelity_valid",
				Severity: SeverityWarning,
				Message:  fmt.Sprintf("Fidelity mode %q is not valid", node.Fidelity),
				NodeID:   node.ID,
			})
		}
	}
	return diagnostics
}

func ruleRetryTargetExists(graph *Graph) []Diagnostic {
	var diagnostics []Diagnostic
	for _, node := range graph.Nodes {
		if node.RetryTarget != "" {
			if _, ok := graph.Nodes[node.RetryTarget]; !ok {
				diagnostics = append(diagnostics, Diagnostic{
					Rule:     "retry_target_exists",
					Severity: SeverityWarning,
					Message:  fmt.Sprintf("retry_target %q does not reference an existing node", node.RetryTarget),
					NodeID:   node.ID,
				})
			}
		}
		if node.FallbackRetryTarget != "" {
			if _, ok := graph.Nodes[node.FallbackRetryTarget]; !ok {
				diagnostics = append(diagnostics, Diagnostic{
					Rule:     "retry_target_exists",
					Severity: SeverityWarning,
					Message:  fmt.Sprintf("fallback_retry_target %q does not reference an existing node", node.FallbackRetryTarget),
					NodeID:   node.ID,
				})
			}
		}
	}
	return diagnostics
}

func ruleGoalGateHasRetry(graph *Graph) []Diagnostic {
	var diagnostics []Diagnostic
	for _, node := range graph.Nodes {
		if node.GoalGate {
			if node.RetryTarget == "" && node.FallbackRetryTarget == "" &&
				graph.RetryTarget == "" && graph.FallbackRetryTarget == "" {
				diagnostics = append(diagnostics, Diagnostic{
					Rule:     "goal_gate_has_retry",
					Severity: SeverityWarning,
					Message:  "goal_gate node has no retry_target or fallback_retry_target",
					NodeID:   node.ID,
					Fix:      "Add retry_target or fallback_retry_target to this node or the graph",
				})
			}
		}
	}
	return diagnostics
}

func rulePromptOnLLMNodes(graph *Graph) []Diagnostic {
	var diagnostics []Diagnostic
	for _, node := range graph.Nodes {
		// Only for nodes that resolve to codergen handler
		if node.Shape == "box" && node.Type == "" {
			if node.Prompt == "" && node.Label == node.ID {
				diagnostics = append(diagnostics, Diagnostic{
					Rule:     "prompt_on_llm_nodes",
					Severity: SeverityWarning,
					Message:  "Codergen node has no prompt or label",
					NodeID:   node.ID,
					Fix:      "Add a prompt or label attribute",
				})
			}
		}
	}
	return diagnostics
}
