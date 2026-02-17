package agent

import (
	"strings"
	"testing"
)

func TestTruncateToolOutputUnderLimit(t *testing.T) {
	content := "short content"
	result := truncateToolOutput(content, 1000)
	if result != content {
		t.Error("content under limit should be unchanged")
	}
}

func TestTruncateToolOutputOverLimit(t *testing.T) {
	content := strings.Repeat("x", 10000)
	result := truncateToolOutput(content, 1000)

	if len(result) > 1200 { // some overhead for marker
		t.Errorf("truncated output too long: %d", len(result))
	}

	if !strings.Contains(result, "[WARNING: Tool output was truncated.") {
		t.Error("expected truncation warning marker")
	}

	if !strings.Contains(result, "characters removed from the middle") {
		t.Error("expected character count in warning")
	}
}

func TestTruncateLinesUnderLimit(t *testing.T) {
	content := "line1\nline2\nline3"
	result := truncateLines(content, 10)
	if result != content {
		t.Error("content under line limit should be unchanged")
	}
}

func TestTruncateLinesOverLimit(t *testing.T) {
	var lines []string
	for i := 0; i < 100; i++ {
		lines = append(lines, "line content")
	}
	content := strings.Join(lines, "\n")

	result := truncateLines(content, 20)

	if strings.Count(result, "\n") > 25 { // 20 lines + marker line + some slack
		t.Errorf("expected ~20 lines, got more: %d", strings.Count(result, "\n"))
	}

	if !strings.Contains(result, "lines omitted") {
		t.Error("expected lines omitted marker")
	}
}

func TestTwoStageTruncation(t *testing.T) {
	// Create content with many lines, each moderately long
	var lines []string
	for i := 0; i < 500; i++ {
		lines = append(lines, strings.Repeat("x", 100))
	}
	content := strings.Join(lines, "\n")

	config := DefaultSessionConfig()
	session := &Session{Config: config}

	// bash tool has both char limit (30k) and line limit (256)
	result := session.applyTruncation("bash", content)

	// Should be truncated by char limit first, then by line limit
	if len(result) > 35000 { // char limit + marker overhead
		t.Errorf("character truncation should have reduced output, got %d", len(result))
	}
}

func TestLoopDetectorPatternLength1(t *testing.T) {
	ld := newLoopDetector(4)

	// Same call 4 times
	for i := 0; i < 3; i++ {
		if ld.recordAndCheck("read_file", `{"path":"a.txt"}`) {
			t.Error("should not detect loop before window is full")
		}
	}
	if !ld.recordAndCheck("read_file", `{"path":"a.txt"}`) {
		t.Error("should detect loop after 4 identical calls (window=4)")
	}
}

func TestLoopDetectorPatternLength2(t *testing.T) {
	ld := newLoopDetector(4)

	// Alternating pattern: A, B, A, B
	ld.recordAndCheck("read_file", `{"path":"a.txt"}`)
	ld.recordAndCheck("write_file", `{"path":"b.txt"}`)
	ld.recordAndCheck("read_file", `{"path":"a.txt"}`)
	result := ld.recordAndCheck("write_file", `{"path":"b.txt"}`)

	if !result {
		t.Error("should detect alternating pattern of length 2")
	}
}

func TestLoopDetectorNoLoop(t *testing.T) {
	ld := newLoopDetector(4)

	ld.recordAndCheck("read_file", `{"path":"a.txt"}`)
	ld.recordAndCheck("write_file", `{"path":"b.txt"}`)
	ld.recordAndCheck("bash", `{"command":"ls"}`)
	result := ld.recordAndCheck("grep", `{"pattern":"foo"}`)

	if result {
		t.Error("should not detect loop for different calls")
	}
}

func TestDefaultToolOutputLimits(t *testing.T) {
	session := &Session{Config: DefaultSessionConfig()}

	if session.getToolOutputLimit("read_file") != 50000 {
		t.Errorf("expected 50000 for read_file, got %d", session.getToolOutputLimit("read_file"))
	}
	if session.getToolOutputLimit("bash") != 30000 {
		t.Errorf("expected 30000 for bash, got %d", session.getToolOutputLimit("bash"))
	}
	if session.getToolOutputLimit("grep") != 20000 {
		t.Errorf("expected 20000 for grep, got %d", session.getToolOutputLimit("grep"))
	}
	if session.getToolOutputLimit("unknown_tool") != 50000 {
		t.Errorf("expected 50000 fallback for unknown tool, got %d", session.getToolOutputLimit("unknown_tool"))
	}
}

func TestCustomToolOutputLimit(t *testing.T) {
	config := DefaultSessionConfig()
	config.ToolOutputLimits = map[string]int{
		"read_file": 100000, // override default
	}
	session := &Session{Config: config}

	if session.getToolOutputLimit("read_file") != 100000 {
		t.Errorf("expected custom limit 100000, got %d", session.getToolOutputLimit("read_file"))
	}
}
