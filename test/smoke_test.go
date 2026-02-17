package integration_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var binaryPath string

func TestMain(m *testing.M) {
	// Build the binary.
	tmpDir, err := os.MkdirTemp("", "attractor-smoke-*")
	if err != nil {
		panic("failed to create temp dir: " + err.Error())
	}
	defer os.RemoveAll(tmpDir)

	binaryPath = filepath.Join(tmpDir, "attractor")
	cmd := exec.Command("go", "build", "-o", binaryPath, "../cmd/attractor")
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stdout
	if err := cmd.Run(); err != nil {
		panic("failed to build binary: " + err.Error())
	}

	os.Exit(m.Run())
}

func TestCLIVersion(t *testing.T) {
	out, err := exec.Command(binaryPath, "version").CombinedOutput()
	if err != nil {
		t.Fatalf("attractor version failed: %v\n%s", err, out)
	}

	output := string(out)
	if !strings.Contains(strings.ToLower(output), "attractor") {
		t.Errorf("expected output to contain 'attractor', got: %s", output)
	}
}

func TestCLIValidateGoodPipeline(t *testing.T) {
	// Write a valid DOT file to temp.
	tmpDir := t.TempDir()
	dotFile := filepath.Join(tmpDir, "good.dot")
	dotContent := `digraph good {
		goal = "Test"
		start [shape=Mdiamond, label="Start"]
		work  [shape=box, label="Work", prompt="Do something"]
		done  [shape=Msquare, label="Done"]
		start -> work
		work -> done
	}`
	if err := os.WriteFile(dotFile, []byte(dotContent), 0o644); err != nil {
		t.Fatalf("failed to write DOT file: %v", err)
	}

	out, err := exec.Command(binaryPath, "validate", dotFile).CombinedOutput()
	if err != nil {
		t.Fatalf("attractor validate (good pipeline) failed: %v\n%s", err, out)
	}
}

func TestCLIValidateBadPipeline(t *testing.T) {
	// Write an invalid DOT file (no start node).
	tmpDir := t.TempDir()
	dotFile := filepath.Join(tmpDir, "bad.dot")
	dotContent := `digraph bad {
		work [shape=box, label="Work"]
		done [shape=box, label="Done"]
		work -> done
	}`
	if err := os.WriteFile(dotFile, []byte(dotContent), 0o644); err != nil {
		t.Fatalf("failed to write DOT file: %v", err)
	}

	cmd := exec.Command(binaryPath, "validate", dotFile)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected attractor validate (bad pipeline) to fail, but it succeeded: %s", out)
	}

	// Verify exit code is non-zero (the error from exec means it was non-zero).
	exitErr, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("expected *exec.ExitError, got %T: %v", err, err)
	}
	if exitErr.ExitCode() == 0 {
		t.Error("expected non-zero exit code for invalid pipeline")
	}
}

func TestCLIHelp(t *testing.T) {
	out, err := exec.Command(binaryPath, "help").CombinedOutput()
	if err != nil {
		t.Fatalf("attractor help failed: %v\n%s", err, out)
	}

	output := string(out)
	if !strings.Contains(output, "Commands:") {
		t.Errorf("expected output to contain 'Commands:', got: %s", output)
	}
}
