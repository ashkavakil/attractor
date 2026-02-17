// Package env implements execution environments for the coding agent.
package env

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Environment is the interface for tool execution.
type Environment interface {
	Execute(ctx context.Context, toolName string, arguments json.RawMessage) (string, error)
}

// LocalEnvironment executes tools on the local filesystem.
type LocalEnvironment struct {
	WorkDir string
	Timeout time.Duration
}

// NewLocalEnvironment creates a local execution environment.
func NewLocalEnvironment(workDir string) *LocalEnvironment {
	if workDir == "" {
		workDir, _ = os.Getwd()
	}
	return &LocalEnvironment{
		WorkDir: workDir,
		Timeout: 10 * time.Second,
	}
}

// Execute runs a tool by name.
func (e *LocalEnvironment) Execute(ctx context.Context, toolName string, arguments json.RawMessage) (string, error) {
	switch toolName {
	case "read_file":
		return e.readFile(arguments)
	case "write_file":
		return e.writeFile(arguments)
	case "edit_file":
		return e.editFile(arguments)
	case "bash":
		return e.bash(ctx, arguments)
	case "glob":
		return e.glob(arguments)
	case "grep":
		return e.grep(ctx, arguments)
	default:
		return "", fmt.Errorf("unknown tool: %s", toolName)
	}
}

func (e *LocalEnvironment) readFile(args json.RawMessage) (string, error) {
	var params struct {
		Path   string `json:"path"`
		Offset int    `json:"offset"`
		Limit  int    `json:"limit"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	path := e.resolvePath(params.Path)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}

	content := string(data)
	if params.Offset > 0 || params.Limit > 0 {
		lines := strings.Split(content, "\n")
		start := params.Offset
		if start > len(lines) {
			start = len(lines)
		}
		end := len(lines)
		if params.Limit > 0 && start+params.Limit < end {
			end = start + params.Limit
		}
		content = strings.Join(lines[start:end], "\n")
	}

	return content, nil
}

func (e *LocalEnvironment) writeFile(args json.RawMessage) (string, error) {
	var params struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	path := e.resolvePath(params.Path)
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create directory: %w", err)
	}

	if err := os.WriteFile(path, []byte(params.Content), 0o644); err != nil {
		return "", fmt.Errorf("write file: %w", err)
	}

	return fmt.Sprintf("Wrote %d bytes to %s", len(params.Content), params.Path), nil
}

func (e *LocalEnvironment) editFile(args json.RawMessage) (string, error) {
	var params struct {
		Path      string `json:"path"`
		OldString string `json:"old_string"`
		NewString string `json:"new_string"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	path := e.resolvePath(params.Path)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}

	content := string(data)
	count := strings.Count(content, params.OldString)
	if count == 0 {
		return "", fmt.Errorf("old_string not found in file")
	}
	if count > 1 {
		return "", fmt.Errorf("old_string found %d times; must be unique", count)
	}

	newContent := strings.Replace(content, params.OldString, params.NewString, 1)
	if err := os.WriteFile(path, []byte(newContent), 0o644); err != nil {
		return "", fmt.Errorf("write file: %w", err)
	}

	return fmt.Sprintf("Edited %s", params.Path), nil
}

// sensitiveEnvPatterns lists environment variable patterns to filter out.
var sensitiveEnvPatterns = []string{
	"_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_CREDENTIAL",
	"AWS_ACCESS_KEY", "AWS_SECRET", "PRIVATE_KEY",
}

// filterEnvironment returns a copy of the environment with sensitive vars removed.
func filterEnvironment() []string {
	var filtered []string
	for _, env := range os.Environ() {
		key := env
		if idx := strings.Index(env, "="); idx >= 0 {
			key = env[:idx]
		}
		isSensitive := false
		upperKey := strings.ToUpper(key)
		for _, pattern := range sensitiveEnvPatterns {
			if strings.Contains(upperKey, pattern) {
				isSensitive = true
				break
			}
		}
		if !isSensitive {
			filtered = append(filtered, env)
		}
	}
	return filtered
}

func (e *LocalEnvironment) bash(ctx context.Context, args json.RawMessage) (string, error) {
	var params struct {
		Command   string `json:"command"`
		TimeoutMs int    `json:"timeout_ms"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	timeout := e.Timeout
	if params.TimeoutMs > 0 {
		timeout = time.Duration(params.TimeoutMs) * time.Millisecond
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "bash", "-c", params.Command)
	cmd.Dir = e.WorkDir
	cmd.Env = filterEnvironment()

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	output := stdout.String()
	if stderr.Len() > 0 {
		output += "\nSTDERR:\n" + stderr.String()
	}
	if err != nil {
		output += "\nExit code: " + err.Error()
	}

	return output, nil
}

func (e *LocalEnvironment) glob(args json.RawMessage) (string, error) {
	var params struct {
		Pattern string `json:"pattern"`
		Path    string `json:"path"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	searchDir := e.WorkDir
	if params.Path != "" {
		searchDir = e.resolvePath(params.Path)
	}

	pattern := filepath.Join(searchDir, params.Pattern)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return "", fmt.Errorf("glob: %w", err)
	}

	return strings.Join(matches, "\n"), nil
}

func (e *LocalEnvironment) grep(ctx context.Context, args json.RawMessage) (string, error) {
	var params struct {
		Pattern string `json:"pattern"`
		Path    string `json:"path"`
		Glob    string `json:"glob"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	searchPath := e.WorkDir
	if params.Path != "" {
		searchPath = e.resolvePath(params.Path)
	}

	cmdArgs := []string{"-rn", params.Pattern, searchPath}
	if params.Glob != "" {
		cmdArgs = append([]string{"--include=" + params.Glob}, cmdArgs...)
	}

	cmd := exec.CommandContext(ctx, "grep", cmdArgs...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	cmd.Run() // grep returns non-zero if no matches
	return stdout.String(), nil
}

func (e *LocalEnvironment) resolvePath(path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(e.WorkDir, path)
}
