// Package condition implements the Attractor condition expression language.
package condition

import (
	"fmt"
	"strings"

	"github.com/ashka-vakil/attractor/pkg/pipeline"
)

// Evaluate evaluates a condition expression against an outcome and context.
// An empty condition always returns true.
func Evaluate(condition string, outcome *pipeline.Outcome, ctx *pipeline.Context) bool {
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
		if !evaluateClause(clause, outcome, ctx) {
			return false
		}
	}
	return true
}

func evaluateClause(clause string, outcome *pipeline.Outcome, ctx *pipeline.Context) bool {
	// Check for != operator first (before = to avoid partial match)
	if idx := strings.Index(clause, "!="); idx >= 0 {
		key := strings.TrimSpace(clause[:idx])
		value := strings.TrimSpace(clause[idx+2:])
		return resolveKey(key, outcome, ctx) != value
	}

	// Check for = operator
	if idx := strings.Index(clause, "="); idx >= 0 {
		key := strings.TrimSpace(clause[:idx])
		value := strings.TrimSpace(clause[idx+1:])
		return resolveKey(key, outcome, ctx) == value
	}

	// Bare key: check if truthy
	resolved := resolveKey(strings.TrimSpace(clause), outcome, ctx)
	return resolved != "" && resolved != "false" && resolved != "0"
}

func resolveKey(key string, outcome *pipeline.Outcome, ctx *pipeline.Context) string {
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

	// context.* prefix
	if strings.HasPrefix(key, "context.") {
		// Try with full key first
		if v, ok := ctx.Get(key); ok {
			return fmt.Sprint(v)
		}
		// Try without context. prefix
		stripped := strings.TrimPrefix(key, "context.")
		if v, ok := ctx.Get(stripped); ok {
			return fmt.Sprint(v)
		}
		return ""
	}

	// Direct context lookup
	if v, ok := ctx.Get(key); ok {
		return fmt.Sprint(v)
	}
	return ""
}

// Validate checks whether a condition expression is syntactically valid.
func Validate(condition string) error {
	condition = strings.TrimSpace(condition)
	if condition == "" {
		return nil
	}

	clauses := strings.Split(condition, "&&")
	for _, clause := range clauses {
		clause = strings.TrimSpace(clause)
		if clause == "" {
			continue
		}
		if err := validateClause(clause); err != nil {
			return err
		}
	}
	return nil
}

func validateClause(clause string) error {
	// Must have a key and optionally an operator with value
	if strings.Contains(clause, "!=") {
		parts := strings.SplitN(clause, "!=", 2)
		if strings.TrimSpace(parts[0]) == "" {
			return fmt.Errorf("empty key in clause: %q", clause)
		}
		return nil
	}
	if strings.Contains(clause, "=") {
		parts := strings.SplitN(clause, "=", 2)
		if strings.TrimSpace(parts[0]) == "" {
			return fmt.Errorf("empty key in clause: %q", clause)
		}
		return nil
	}
	// Bare key
	if strings.TrimSpace(clause) == "" {
		return fmt.Errorf("empty clause")
	}
	return nil
}
