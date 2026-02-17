// Package transform implements AST transforms for the Attractor pipeline engine.
package transform

import (
	"strings"

	"github.com/ashka-vakil/attractor/pkg/pipeline"
	"github.com/ashka-vakil/attractor/pkg/pipeline/stylesheet"
)

// Transform is the interface for graph transformations.
type Transform interface {
	Apply(graph *pipeline.Graph) *pipeline.Graph
}

// TransformFunc is an adapter to use ordinary functions as transforms.
type TransformFunc func(graph *pipeline.Graph) *pipeline.Graph

func (f TransformFunc) Apply(graph *pipeline.Graph) *pipeline.Graph {
	return f(graph)
}

// VariableExpansion replaces $goal in node prompts.
func VariableExpansion() Transform {
	return TransformFunc(func(graph *pipeline.Graph) *pipeline.Graph {
		for _, node := range graph.Nodes {
			if strings.Contains(node.Prompt, "$goal") {
				node.Prompt = strings.ReplaceAll(node.Prompt, "$goal", graph.Goal)
			}
		}
		return graph
	})
}

// StylesheetApplication applies the model_stylesheet to nodes.
func StylesheetApplication() Transform {
	return TransformFunc(func(graph *pipeline.Graph) *pipeline.Graph {
		if graph.ModelStylesheet == "" {
			return graph
		}
		ss, err := stylesheet.Parse(graph.ModelStylesheet)
		if err != nil {
			return graph // silently skip invalid stylesheets
		}
		ss.Apply(graph)
		return graph
	})
}

// ApplyAll runs a series of transforms in order.
func ApplyAll(graph *pipeline.Graph, transforms ...Transform) *pipeline.Graph {
	for _, t := range transforms {
		graph = t.Apply(graph)
	}
	return graph
}

// DefaultTransforms returns the built-in transforms in their canonical order.
func DefaultTransforms() []Transform {
	return []Transform{
		VariableExpansion(),
		StylesheetApplication(),
	}
}
