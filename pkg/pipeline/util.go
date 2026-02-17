package pipeline

import (
	"fmt"
	"os"
	"path/filepath"
)

// writeFile writes data to a file, creating parent directories as needed.
func writeFile(path string, data []byte) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create directory %s: %w", dir, err)
	}
	return os.WriteFile(path, data, 0o644)
}

// readFile reads data from a file.
func readFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

// truncate trims a string to the given max length, appending "..." if truncated.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}
