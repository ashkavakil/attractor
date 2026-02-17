// Package stylesheet implements the Attractor model stylesheet parser and applicator.
package stylesheet

import (
	"fmt"
	"strings"

	"github.com/ashka-vakil/attractor/pkg/pipeline"
)

// SelectorType identifies the type of CSS-like selector.
type SelectorType int

const (
	SelectorUniversal SelectorType = iota // *
	SelectorShape                          // bare shape name (e.g., box)
	SelectorClass                          // .class_name
	SelectorID                             // #node_id
)

// Specificity values per the DoD: universal < shape < class < ID
const (
	SpecificityUniversal = 0
	SpecificityShape     = 1
	SpecificityClass     = 2
	SpecificityID        = 3
)

// Rule is a single stylesheet rule.
type Rule struct {
	Selector     string
	SelectorType SelectorType
	Properties   map[string]string
	Specificity  int
}

// Stylesheet is a parsed model stylesheet.
type Stylesheet struct {
	Rules []Rule
}

// knownShapes is the set of DOT shapes that can be used as selectors.
var knownShapes = map[string]bool{
	"box": true, "Mdiamond": true, "Msquare": true,
	"hexagon": true, "diamond": true, "component": true,
	"tripleoctagon": true, "parallelogram": true, "house": true,
	"ellipse": true, "circle": true, "oval": true,
	"record": true, "plaintext": true, "point": true,
}

// Parse parses a CSS-like model stylesheet string.
func Parse(source string) (*Stylesheet, error) {
	source = strings.TrimSpace(source)
	if source == "" {
		return &Stylesheet{}, nil
	}

	ss := &Stylesheet{}
	remaining := source

	for remaining != "" {
		remaining = strings.TrimSpace(remaining)
		if remaining == "" {
			break
		}

		// Find selector
		braceIdx := strings.Index(remaining, "{")
		if braceIdx < 0 {
			return nil, fmt.Errorf("expected '{' in stylesheet rule")
		}

		selector := strings.TrimSpace(remaining[:braceIdx])
		remaining = remaining[braceIdx+1:]

		// Find closing brace
		closeIdx := strings.Index(remaining, "}")
		if closeIdx < 0 {
			return nil, fmt.Errorf("expected '}' in stylesheet rule")
		}

		body := strings.TrimSpace(remaining[:closeIdx])
		remaining = remaining[closeIdx+1:]

		// Parse selector type
		rule := Rule{
			Selector:   selector,
			Properties: make(map[string]string),
		}

		switch {
		case selector == "*":
			rule.SelectorType = SelectorUniversal
			rule.Specificity = SpecificityUniversal
		case strings.HasPrefix(selector, "."):
			rule.SelectorType = SelectorClass
			rule.Selector = strings.TrimPrefix(selector, ".")
			rule.Specificity = SpecificityClass
		case strings.HasPrefix(selector, "#"):
			rule.SelectorType = SelectorID
			rule.Selector = strings.TrimPrefix(selector, "#")
			rule.Specificity = SpecificityID
		default:
			// Bare name - treat as shape selector
			rule.SelectorType = SelectorShape
			rule.Specificity = SpecificityShape
		}

		// Parse declarations
		declarations := strings.Split(body, ";")
		for _, decl := range declarations {
			decl = strings.TrimSpace(decl)
			if decl == "" {
				continue
			}
			colonIdx := strings.Index(decl, ":")
			if colonIdx < 0 {
				// Also support "=" as separator for compatibility
				colonIdx = strings.Index(decl, "=")
				if colonIdx < 0 {
					return nil, fmt.Errorf("invalid declaration (missing ':' or '='): %q", decl)
				}
			}
			prop := strings.TrimSpace(decl[:colonIdx])
			val := strings.TrimSpace(decl[colonIdx+1:])
			val = strings.TrimRight(val, ";")
			val = strings.Trim(val, `"' `)
			rule.Properties[prop] = val
		}

		ss.Rules = append(ss.Rules, rule)
	}

	return ss, nil
}

// Validate checks that the stylesheet is well-formed.
func (ss *Stylesheet) Validate() error {
	validProps := map[string]bool{
		"llm_model":        true,
		"llm_provider":     true,
		"reasoning_effort": true,
		"model":            true, // alias for llm_model
	}

	for _, rule := range ss.Rules {
		for prop := range rule.Properties {
			if !validProps[prop] {
				return fmt.Errorf("unknown stylesheet property: %q", prop)
			}
		}
	}
	return nil
}

// Apply applies the stylesheet to all nodes in the graph.
// Explicit node attributes always take highest precedence.
func (ss *Stylesheet) Apply(graph *pipeline.Graph) {
	for _, node := range graph.Nodes {
		// Save explicitly set values before stylesheet application.
		explicitModel := node.LLMModel
		explicitProvider := node.LLMProvider
		explicitEffort := node.ReasoningEffort

		// Clear so stylesheet can set them.
		node.LLMModel = ""
		node.LLMProvider = ""
		node.ReasoningEffort = ""

		// Collect matching rules, sorted by specificity.
		var matches []Rule
		for _, rule := range ss.Rules {
			if ss.matches(rule, node) {
				matches = append(matches, rule)
			}
		}

		// Apply in specificity order (lower specificity first, so higher overrides).
		sortBySpecificity(matches)

		for _, rule := range matches {
			for prop, val := range rule.Properties {
				applyProperty(node, prop, val)
			}
		}

		// Restore explicit values (highest precedence).
		if explicitModel != "" {
			node.LLMModel = explicitModel
		}
		if explicitProvider != "" {
			node.LLMProvider = explicitProvider
		}
		if explicitEffort != "" {
			node.ReasoningEffort = explicitEffort
		}
	}
}

func (ss *Stylesheet) matches(rule Rule, node *pipeline.Node) bool {
	switch rule.SelectorType {
	case SelectorUniversal:
		return true
	case SelectorShape:
		return node.Shape == rule.Selector
	case SelectorClass:
		classes := strings.Split(node.Class, ",")
		for _, c := range classes {
			if strings.TrimSpace(c) == rule.Selector {
				return true
			}
		}
		return false
	case SelectorID:
		return node.ID == rule.Selector
	}
	return false
}

func sortBySpecificity(rules []Rule) {
	for i := 1; i < len(rules); i++ {
		for j := i; j > 0 && rules[j].Specificity < rules[j-1].Specificity; j-- {
			rules[j], rules[j-1] = rules[j-1], rules[j]
		}
	}
}

func applyProperty(node *pipeline.Node, prop, val string) {
	switch prop {
	case "llm_model", "model":
		node.LLMModel = val
	case "llm_provider":
		node.LLMProvider = val
	case "reasoning_effort":
		node.ReasoningEffort = val
	}
}
