// Package tools implements the built-in tool definitions for the coding agent.
package tools

import (
	"encoding/json"

	"github.com/ashka-vakil/attractor/pkg/llm"
)

// ReadFile returns the read_file tool definition.
func ReadFile() llm.Tool {
	return llm.Tool{
		Name:        "read_file",
		Description: "Read the contents of a file at the given path. Returns the file contents as a string.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "The absolute or relative path to the file to read"
				},
				"offset": {
					"type": "integer",
					"description": "Line number to start reading from (1-indexed)"
				},
				"limit": {
					"type": "integer",
					"description": "Maximum number of lines to read"
				}
			},
			"required": ["path"]
		}`),
	}
}

// WriteFile returns the write_file tool definition.
func WriteFile() llm.Tool {
	return llm.Tool{
		Name:        "write_file",
		Description: "Write content to a file at the given path, creating it if it doesn't exist or overwriting if it does.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "The path to the file to write"
				},
				"content": {
					"type": "string",
					"description": "The content to write to the file"
				}
			},
			"required": ["path", "content"]
		}`),
	}
}

// EditFile returns the edit_file tool definition.
func EditFile() llm.Tool {
	return llm.Tool{
		Name:        "edit_file",
		Description: "Make targeted edits to a file by replacing specific text. The old_string must match exactly.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "The path to the file to edit"
				},
				"old_string": {
					"type": "string",
					"description": "The exact string to find and replace"
				},
				"new_string": {
					"type": "string",
					"description": "The replacement string"
				}
			},
			"required": ["path", "old_string", "new_string"]
		}`),
	}
}

// Bash returns the bash tool definition.
func Bash() llm.Tool {
	return llm.Tool{
		Name:        "bash",
		Description: "Execute a bash command and return the output. Use for running tests, building, git operations, etc.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"command": {
					"type": "string",
					"description": "The bash command to execute"
				},
				"timeout_ms": {
					"type": "integer",
					"description": "Timeout in milliseconds (default: 10000)"
				}
			},
			"required": ["command"]
		}`),
	}
}

// GlobSearch returns the glob search tool definition.
func GlobSearch() llm.Tool {
	return llm.Tool{
		Name:        "glob",
		Description: "Search for files matching a glob pattern.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"pattern": {
					"type": "string",
					"description": "The glob pattern to match (e.g., '**/*.go')"
				},
				"path": {
					"type": "string",
					"description": "The directory to search in"
				}
			},
			"required": ["pattern"]
		}`),
	}
}

// GrepSearch returns the grep search tool definition.
func GrepSearch() llm.Tool {
	return llm.Tool{
		Name:        "grep",
		Description: "Search file contents for a regex pattern.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"pattern": {
					"type": "string",
					"description": "The regex pattern to search for"
				},
				"path": {
					"type": "string",
					"description": "The directory or file to search"
				},
				"glob": {
					"type": "string",
					"description": "Glob pattern to filter files"
				}
			},
			"required": ["pattern"]
		}`),
	}
}
