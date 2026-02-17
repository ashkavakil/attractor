package agent

import (
	"context"
	"encoding/json"

	"github.com/ashka-vakil/attractor/pkg/agent/env"
	"github.com/ashka-vakil/attractor/pkg/agent/tools"
	"github.com/ashka-vakil/attractor/pkg/llm"
)

// ExecutionEnvironment is the interface for tool execution.
type ExecutionEnvironment interface {
	Execute(ctx context.Context, toolName string, arguments json.RawMessage) (string, error)
}

// NewLocalEnvironment creates a local execution environment.
func NewLocalEnvironment() ExecutionEnvironment {
	return env.NewLocalEnvironment("")
}

// DefaultToolSet returns the default set of tools.
func DefaultToolSet() []llm.Tool {
	return []llm.Tool{
		tools.ReadFile(),
		tools.WriteFile(),
		tools.EditFile(),
		tools.Bash(),
		tools.GlobSearch(),
		tools.GrepSearch(),
	}
}

// SubAgent represents a child agent spawned for parallel work.
type SubAgent struct {
	ID      string
	Session *Session
	Depth   int
}
