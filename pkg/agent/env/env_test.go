package env

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// helper to create a LocalEnvironment pointing at a temp directory.
func setupEnv(t *testing.T) (*LocalEnvironment, string) {
	t.Helper()
	dir := t.TempDir()
	env := NewLocalEnvironment(dir)
	return env, dir
}

// --- read_file tests ---

func TestReadFile(t *testing.T) {
	e, dir := setupEnv(t)
	ctx := context.Background()

	// Write a multi-line temp file to read back.
	content := "line0\nline1\nline2\nline3\nline4\n"
	path := filepath.Join(dir, "read_test.txt")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("setup: write temp file: %v", err)
	}

	t.Run("full file", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"path": path})
		result, err := e.Execute(ctx, "read_file", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != content {
			t.Errorf("got %q, want %q", result, content)
		}
	})

	t.Run("with offset", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"path": path, "offset": 2})
		result, err := e.Execute(ctx, "read_file", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// lines[2:] = "line2\nline3\nline4\n" (the trailing newline produces an empty final element)
		want := "line2\nline3\nline4\n"
		if result != want {
			t.Errorf("got %q, want %q", result, want)
		}
	})

	t.Run("with limit", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"path": path, "limit": 2})
		result, err := e.Execute(ctx, "read_file", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := "line0\nline1"
		if result != want {
			t.Errorf("got %q, want %q", result, want)
		}
	})

	t.Run("with offset and limit", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"path": path, "offset": 1, "limit": 2})
		result, err := e.Execute(ctx, "read_file", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := "line1\nline2"
		if result != want {
			t.Errorf("got %q, want %q", result, want)
		}
	})

	t.Run("nonexistent file", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"path": filepath.Join(dir, "no_such_file.txt")})
		_, err := e.Execute(ctx, "read_file", args)
		if err == nil {
			t.Fatal("expected error for nonexistent file")
		}
	})

	t.Run("relative path", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"path": "read_test.txt"})
		result, err := e.Execute(ctx, "read_file", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != content {
			t.Errorf("got %q, want %q", result, content)
		}
	})
}

// --- write_file tests ---

func TestWriteFile(t *testing.T) {
	e, dir := setupEnv(t)
	ctx := context.Background()

	t.Run("simple write", func(t *testing.T) {
		path := filepath.Join(dir, "out.txt")
		args, _ := json.Marshal(map[string]interface{}{"path": path, "content": "hello world"})
		result, err := e.Execute(ctx, "write_file", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "11") { // "hello world" is 11 bytes
			t.Errorf("result should mention byte count, got %q", result)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read back: %v", err)
		}
		if string(data) != "hello world" {
			t.Errorf("file content = %q, want %q", string(data), "hello world")
		}
	})

	t.Run("creates intermediate directories", func(t *testing.T) {
		path := filepath.Join(dir, "sub", "deep", "file.txt")
		args, _ := json.Marshal(map[string]interface{}{"path": path, "content": "nested"})
		_, err := e.Execute(ctx, "write_file", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read back: %v", err)
		}
		if string(data) != "nested" {
			t.Errorf("file content = %q, want %q", string(data), "nested")
		}
	})

	t.Run("overwrites existing file", func(t *testing.T) {
		path := filepath.Join(dir, "overwrite.txt")
		if err := os.WriteFile(path, []byte("old"), 0o644); err != nil {
			t.Fatalf("setup: %v", err)
		}
		args, _ := json.Marshal(map[string]interface{}{"path": path, "content": "new"})
		_, err := e.Execute(ctx, "write_file", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		data, _ := os.ReadFile(path)
		if string(data) != "new" {
			t.Errorf("file content = %q, want %q", string(data), "new")
		}
	})
}

// --- edit_file tests ---

func TestEditFile(t *testing.T) {
	e, dir := setupEnv(t)
	ctx := context.Background()

	t.Run("unique match", func(t *testing.T) {
		path := filepath.Join(dir, "edit_unique.txt")
		if err := os.WriteFile(path, []byte("foo bar baz"), 0o644); err != nil {
			t.Fatalf("setup: %v", err)
		}
		args, _ := json.Marshal(map[string]interface{}{
			"path":       path,
			"old_string": "bar",
			"new_string": "qux",
		})
		result, err := e.Execute(ctx, "edit_file", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "Edited") {
			t.Errorf("expected 'Edited' in result, got %q", result)
		}
		data, _ := os.ReadFile(path)
		if string(data) != "foo qux baz" {
			t.Errorf("file content = %q, want %q", string(data), "foo qux baz")
		}
	})

	t.Run("non-unique match errors", func(t *testing.T) {
		path := filepath.Join(dir, "edit_nonunique.txt")
		if err := os.WriteFile(path, []byte("aaa bbb aaa"), 0o644); err != nil {
			t.Fatalf("setup: %v", err)
		}
		args, _ := json.Marshal(map[string]interface{}{
			"path":       path,
			"old_string": "aaa",
			"new_string": "ccc",
		})
		_, err := e.Execute(ctx, "edit_file", args)
		if err == nil {
			t.Fatal("expected error for non-unique old_string")
		}
		if !strings.Contains(err.Error(), "2 times") {
			t.Errorf("expected error about 2 occurrences, got %q", err.Error())
		}
	})

	t.Run("not found errors", func(t *testing.T) {
		path := filepath.Join(dir, "edit_notfound.txt")
		if err := os.WriteFile(path, []byte("hello world"), 0o644); err != nil {
			t.Fatalf("setup: %v", err)
		}
		args, _ := json.Marshal(map[string]interface{}{
			"path":       path,
			"old_string": "MISSING",
			"new_string": "replacement",
		})
		_, err := e.Execute(ctx, "edit_file", args)
		if err == nil {
			t.Fatal("expected error for missing old_string")
		}
		if !strings.Contains(err.Error(), "not found") {
			t.Errorf("expected 'not found' in error, got %q", err.Error())
		}
	})
}

// --- bash tests ---

func TestBash(t *testing.T) {
	e, _ := setupEnv(t)
	ctx := context.Background()

	t.Run("stdout capture", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"command": "echo hello"})
		result, err := e.Execute(ctx, "bash", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "hello") {
			t.Errorf("expected 'hello' in output, got %q", result)
		}
	})

	t.Run("stderr capture", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"command": "echo oops >&2"})
		result, err := e.Execute(ctx, "bash", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "STDERR:") {
			t.Errorf("expected 'STDERR:' in output, got %q", result)
		}
		if !strings.Contains(result, "oops") {
			t.Errorf("expected 'oops' in output, got %q", result)
		}
	})

	t.Run("exit code reporting", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"command": "exit 42"})
		result, err := e.Execute(ctx, "bash", args)
		// bash tool does not return an error for non-zero exit; it includes exit info in output.
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "Exit code") {
			t.Errorf("expected 'Exit code' in output, got %q", result)
		}
	})

	t.Run("working directory", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"command": "pwd"})
		result, err := e.Execute(ctx, "bash", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, e.WorkDir) {
			t.Errorf("expected working dir %q in output, got %q", e.WorkDir, result)
		}
	})
}

func TestBashTimeout(t *testing.T) {
	e, _ := setupEnv(t)
	ctx := context.Background()

	// Use a very short timeout so the sleep gets killed quickly.
	args, _ := json.Marshal(map[string]interface{}{
		"command":    "sleep 30",
		"timeout_ms": 200,
	})

	start := time.Now()
	result, err := e.Execute(ctx, "bash", args)
	elapsed := time.Since(start)

	// The command should have been terminated long before 30 seconds.
	if elapsed > 5*time.Second {
		t.Fatalf("timeout did not trigger: command took %v", elapsed)
	}

	// bash returns output (with exit info) rather than an error.
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The output should contain an exit code or signal info.
	if !strings.Contains(result, "Exit code") && result == "" {
		t.Logf("result after timeout: %q", result)
	}
}

// --- glob tests ---

func TestGlob(t *testing.T) {
	e, dir := setupEnv(t)
	ctx := context.Background()

	// Create a few temp files with known extensions.
	for _, name := range []string{"a.go", "b.go", "c.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatalf("setup: %v", err)
		}
	}

	t.Run("match go files", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"pattern": "*.go"})
		result, err := e.Execute(ctx, "glob", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "a.go") || !strings.Contains(result, "b.go") {
			t.Errorf("expected a.go and b.go in result, got %q", result)
		}
		if strings.Contains(result, "c.txt") {
			t.Errorf("did not expect c.txt in result, got %q", result)
		}
	})

	t.Run("match txt files", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"pattern": "*.txt"})
		result, err := e.Execute(ctx, "glob", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "c.txt") {
			t.Errorf("expected c.txt in result, got %q", result)
		}
	})

	t.Run("no matches returns empty", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"pattern": "*.rs"})
		result, err := e.Execute(ctx, "glob", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != "" {
			t.Errorf("expected empty result, got %q", result)
		}
	})

	t.Run("explicit path", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"pattern": "*.go", "path": dir})
		result, err := e.Execute(ctx, "glob", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "a.go") {
			t.Errorf("expected a.go in result, got %q", result)
		}
	})
}

// --- grep tests ---

func TestGrep(t *testing.T) {
	e, dir := setupEnv(t)
	ctx := context.Background()

	// Create files with known content.
	if err := os.WriteFile(filepath.Join(dir, "haystack.txt"), []byte("needle in a haystack\nno match here\nneedle again\n"), 0o644); err != nil {
		t.Fatalf("setup: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "other.go"), []byte("func main() {}\n"), 0o644); err != nil {
		t.Fatalf("setup: %v", err)
	}

	t.Run("finds pattern", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"pattern": "needle", "path": dir})
		result, err := e.Execute(ctx, "grep", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "needle") {
			t.Errorf("expected 'needle' in result, got %q", result)
		}
		// Should match two lines.
		lines := strings.Split(strings.TrimSpace(result), "\n")
		if len(lines) < 2 {
			t.Errorf("expected at least 2 matching lines, got %d", len(lines))
		}
	})

	t.Run("no matches returns empty", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"pattern": "ZZZMISSING", "path": dir})
		result, err := e.Execute(ctx, "grep", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if strings.TrimSpace(result) != "" {
			t.Errorf("expected empty result, got %q", result)
		}
	})

	t.Run("glob filter", func(t *testing.T) {
		args, _ := json.Marshal(map[string]interface{}{"pattern": "func", "path": dir, "glob": "*.go"})
		result, err := e.Execute(ctx, "grep", args)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(result, "func") {
			t.Errorf("expected 'func' in result, got %q", result)
		}
		if strings.Contains(result, "haystack") {
			t.Errorf("glob filter should have excluded haystack.txt, got %q", result)
		}
	})
}

// --- resolvePath tests ---

func TestResolvePath(t *testing.T) {
	e, dir := setupEnv(t)

	t.Run("absolute path returned as-is", func(t *testing.T) {
		absPath := "/tmp/some/absolute/path"
		got := e.resolvePath(absPath)
		if got != absPath {
			t.Errorf("got %q, want %q", got, absPath)
		}
	})

	t.Run("relative path joined with WorkDir", func(t *testing.T) {
		got := e.resolvePath("foo/bar.txt")
		want := filepath.Join(dir, "foo/bar.txt")
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("empty path", func(t *testing.T) {
		got := e.resolvePath("")
		want := dir // filepath.Join(dir, "") == dir
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

// --- filterEnvironment tests ---

func TestFilterEnvironment(t *testing.T) {
	// Set some sensitive and non-sensitive env vars for testing.
	sensitiveVars := map[string]string{
		"MY_API_KEY":       "secret1",
		"DB_SECRET":        "secret2",
		"AUTH_TOKEN":       "secret3",
		"DB_PASSWORD":      "secret4",
		"AWS_ACCESS_KEY":   "secret5",
		"AWS_SECRET":       "secret6",
		"MY_PRIVATE_KEY":   "secret7",
		"APP_CREDENTIAL":   "secret8",
	}
	safeVar := "ATTRACTOR_TEST_SAFE_VAR"

	for k, v := range sensitiveVars {
		t.Setenv(k, v)
	}
	t.Setenv(safeVar, "visible")

	filtered := filterEnvironment()
	filteredMap := make(map[string]string)
	for _, entry := range filtered {
		parts := strings.SplitN(entry, "=", 2)
		if len(parts) == 2 {
			filteredMap[parts[0]] = parts[1]
		}
	}

	// The safe var should be present.
	if _, ok := filteredMap[safeVar]; !ok {
		t.Errorf("expected %s to be present in filtered env", safeVar)
	}

	// All sensitive vars should be absent.
	for k := range sensitiveVars {
		if _, ok := filteredMap[k]; ok {
			t.Errorf("expected %s to be filtered out, but it was present", k)
		}
	}
}

// --- unknown tool tests ---

func TestUnknownTool(t *testing.T) {
	e, _ := setupEnv(t)
	ctx := context.Background()

	args := json.RawMessage(`{}`)
	_, err := e.Execute(ctx, "no_such_tool", args)
	if err == nil {
		t.Fatal("expected error for unknown tool")
	}
	if !strings.Contains(err.Error(), "unknown tool") {
		t.Errorf("expected 'unknown tool' in error, got %q", err.Error())
	}
}

// --- NewLocalEnvironment default WorkDir ---

func TestNewLocalEnvironmentDefaults(t *testing.T) {
	e := NewLocalEnvironment("")
	cwd, _ := os.Getwd()
	if e.WorkDir != cwd {
		t.Errorf("default WorkDir = %q, want %q", e.WorkDir, cwd)
	}
	if e.Timeout != 10*time.Second {
		t.Errorf("default Timeout = %v, want %v", e.Timeout, 10*time.Second)
	}
}
