package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ashka-vakil/attractor/pkg/llm"
)

// ProviderProfile defines the tools and system prompt for a specific model family.
type ProviderProfile struct {
	Name                    string     `json:"name"`
	Provider                string     `json:"provider"`
	Model                   string     `json:"model"`
	SystemPrompt            string     `json:"system_prompt"`
	Tools                   []llm.Tool `json:"tools"`
	SupportsParallelToolCalls bool     `json:"supports_parallel_tool_calls"`
}

// RegisterTool adds a custom tool to the profile, overriding any existing tool with the same name.
func (p *ProviderProfile) RegisterTool(tool llm.Tool) {
	for i, t := range p.Tools {
		if t.Name == tool.Name {
			p.Tools[i] = tool
			return
		}
	}
	p.Tools = append(p.Tools, tool)
}

// BuildSystemPrompt generates the full system prompt including environment context.
func BuildSystemPrompt(profile *ProviderProfile, workDir string, userInstructions string) string {
	var parts []string

	// 1. Provider-specific base instructions
	parts = append(parts, profile.SystemPrompt)

	// 2. Environment context
	envContext := buildEnvironmentContext(workDir, profile.Model)
	parts = append(parts, envContext)

	// 3. Tool descriptions
	if len(profile.Tools) > 0 {
		toolDesc := buildToolDescriptions(profile.Tools)
		parts = append(parts, toolDesc)
	}

	// 4. Project documentation
	projectDocs := discoverProjectDocs(workDir, profile.Provider)
	if projectDocs != "" {
		parts = append(parts, projectDocs)
	}

	// 5. User instruction overrides (highest priority)
	if userInstructions != "" {
		parts = append(parts, "# User Instructions\n"+userInstructions)
	}

	return strings.Join(parts, "\n\n")
}

func buildEnvironmentContext(workDir string, model string) string {
	gitBranch := ""
	if data, err := os.ReadFile(filepath.Join(workDir, ".git", "HEAD")); err == nil {
		ref := strings.TrimSpace(string(data))
		if strings.HasPrefix(ref, "ref: refs/heads/") {
			gitBranch = strings.TrimPrefix(ref, "ref: refs/heads/")
		}
	}

	ctx := fmt.Sprintf(`# Environment
- Platform: %s/%s
- Working directory: %s
- Date: %s
- Model: %s`,
		runtime.GOOS, runtime.GOARCH,
		workDir,
		time.Now().Format("2006-01-02"),
		model,
	)

	if gitBranch != "" {
		ctx += "\n- Git branch: " + gitBranch
	}

	return ctx
}

func buildToolDescriptions(tools []llm.Tool) string {
	var lines []string
	lines = append(lines, "# Available Tools")
	for _, t := range tools {
		lines = append(lines, fmt.Sprintf("- **%s**: %s", t.Name, t.Description))
	}
	return strings.Join(lines, "\n")
}

func discoverProjectDocs(workDir string, provider string) string {
	var docs []string

	// Common project files
	agentsMD := filepath.Join(workDir, "AGENTS.md")
	if data, err := os.ReadFile(agentsMD); err == nil {
		docs = append(docs, "# Project Instructions (AGENTS.md)\n"+string(data))
	}

	// Provider-specific files
	providerFiles := map[string][]string{
		"anthropic": {"CLAUDE.md"},
		"openai":    {"CODEX.md"},
		"gemini":    {"GEMINI.md"},
	}

	if files, ok := providerFiles[provider]; ok {
		for _, f := range files {
			path := filepath.Join(workDir, f)
			if data, err := os.ReadFile(path); err == nil {
				docs = append(docs, fmt.Sprintf("# Project Instructions (%s)\n%s", f, string(data)))
			}
		}
	}

	return strings.Join(docs, "\n\n")
}

// DefaultAnthropicProfile returns the default profile for Anthropic models.
func DefaultAnthropicProfile(model string) *ProviderProfile {
	return &ProviderProfile{
		Name:     "anthropic",
		Provider: "anthropic",
		Model:    model,
		SupportsParallelToolCalls: true,
		SystemPrompt: `You are an expert coding assistant. You help users with software engineering tasks by reading files, editing code, running commands, and iterating until the task is done.

Use the edit_file tool for targeted edits using exact string matching (old_string/new_string pattern).
Use read_file to understand existing code before making changes.
Use bash for running tests, builds, and git operations.`,
		Tools: DefaultToolSet(),
	}
}

// DefaultOpenAIProfile returns the default profile for OpenAI models.
func DefaultOpenAIProfile(model string) *ProviderProfile {
	tools := DefaultToolSet()
	// Add apply_patch tool (v4a format) for OpenAI codex compatibility
	tools = append(tools, ApplyPatchTool())

	return &ProviderProfile{
		Name:     "openai",
		Provider: "openai",
		Model:    model,
		SupportsParallelToolCalls: true,
		SystemPrompt: `You are an expert coding assistant. You help users with software engineering tasks by reading files, editing code, running commands, and iterating until the task is done.

Use apply_patch for file modifications using the v4a diff format.
Use read_file to understand existing code before making changes.
Use bash for running tests, builds, and git operations.`,
		Tools: tools,
	}
}

// DefaultGeminiProfile returns the default profile for Gemini models.
func DefaultGeminiProfile(model string) *ProviderProfile {
	return &ProviderProfile{
		Name:     "gemini",
		Provider: "gemini",
		Model:    model,
		SupportsParallelToolCalls: true,
		SystemPrompt: `You are an expert coding assistant. You help users with software engineering tasks by reading files, editing code, running commands, and iterating until the task is done.

Use write_file and edit_file for code modifications.
Use read_file to understand existing code before making changes.
Use bash for running tests, builds, and git operations.`,
		Tools: DefaultToolSet(),
	}
}

// ApplyPatchTool returns the apply_patch tool definition (v4a format for OpenAI profile).
func ApplyPatchTool() llm.Tool {
	return llm.Tool{
		Name:        "apply_patch",
		Description: "Apply a patch to files using the v4a diff format. Supports adding, deleting, updating, and renaming files.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"patch": {
					"type": "string",
					"description": "The patch in v4a format. Must start with '*** Begin Patch' and end with '*** End Patch'."
				}
			},
			"required": ["patch"]
		}`),
	}
}

// SubagentTools returns tools for spawning and managing subagents.
func SubagentTools() []llm.Tool {
	return []llm.Tool{
		{
			Name:        "spawn_agent",
			Description: "Spawn a subagent to work on a scoped task. The subagent shares the filesystem but has independent conversation history.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"task": {
						"type": "string",
						"description": "The task for the subagent to work on"
					},
					"model": {
						"type": "string",
						"description": "Optional model to use for the subagent"
					}
				},
				"required": ["task"]
			}`),
		},
		{
			Name:        "send_input",
			Description: "Send additional input to a running subagent.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"agent_id": {"type": "string", "description": "The subagent ID"},
					"input": {"type": "string", "description": "The input to send"}
				},
				"required": ["agent_id", "input"]
			}`),
		},
		{
			Name:        "wait_agent",
			Description: "Wait for a subagent to complete and return its result.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"agent_id": {"type": "string", "description": "The subagent ID"}
				},
				"required": ["agent_id"]
			}`),
		},
		{
			Name:        "close_agent",
			Description: "Close a running subagent.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"agent_id": {"type": "string", "description": "The subagent ID"}
				},
				"required": ["agent_id"]
			}`),
		},
	}
}
