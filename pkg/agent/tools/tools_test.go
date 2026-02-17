package tools

import (
	"encoding/json"
	"testing"

	"github.com/ashka-vakil/attractor/pkg/llm"
)

// toolTestCase describes an expected tool definition.
type toolTestCase struct {
	name           string
	fn             func() llm.Tool
	requiredFields []string // expected entries in the "required" JSON array
}

var allTools = []toolTestCase{
	{
		name:           "ReadFile",
		fn:             ReadFile,
		requiredFields: []string{"path"},
	},
	{
		name:           "WriteFile",
		fn:             WriteFile,
		requiredFields: []string{"path", "content"},
	},
	{
		name:           "EditFile",
		fn:             EditFile,
		requiredFields: []string{"path", "old_string", "new_string"},
	},
	{
		name:           "Bash",
		fn:             Bash,
		requiredFields: []string{"command"},
	},
	{
		name:           "GlobSearch",
		fn:             GlobSearch,
		requiredFields: []string{"pattern"},
	},
	{
		name:           "GrepSearch",
		fn:             GrepSearch,
		requiredFields: []string{"pattern"},
	},
}

func TestToolDefinitions(t *testing.T) {
	for _, tc := range allTools {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			tool := tc.fn()

			// Name must not be empty.
			if tool.Name == "" {
				t.Error("Name is empty")
			}

			// Description must not be empty.
			if tool.Description == "" {
				t.Error("Description is empty")
			}

			// Parameters must be valid JSON.
			var schema map[string]interface{}
			if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
				t.Fatalf("Parameters is not valid JSON: %v", err)
			}

			// Parameters must contain "type" and "properties".
			if _, ok := schema["type"]; !ok {
				t.Error("Parameters JSON missing 'type' field")
			}
			if _, ok := schema["properties"]; !ok {
				t.Error("Parameters JSON missing 'properties' field")
			}

			// Parameters must contain "required" array.
			reqRaw, ok := schema["required"]
			if !ok {
				t.Fatal("Parameters JSON missing 'required' field")
			}

			// Parse the "required" array.
			reqSlice, ok := reqRaw.([]interface{})
			if !ok {
				t.Fatalf("'required' is not an array, got %T", reqRaw)
			}

			reqSet := make(map[string]bool)
			for _, v := range reqSlice {
				s, ok := v.(string)
				if !ok {
					t.Fatalf("'required' array contains non-string: %v", v)
				}
				reqSet[s] = true
			}

			// Verify each expected required field is present.
			for _, field := range tc.requiredFields {
				if !reqSet[field] {
					t.Errorf("expected %q in required array, got %v", field, reqSlice)
				}
			}

			// Verify no unexpected required fields.
			if len(reqSlice) != len(tc.requiredFields) {
				t.Errorf("required array length = %d, want %d; got %v",
					len(reqSlice), len(tc.requiredFields), reqSlice)
			}

			// Verify each required field appears in properties.
			props, _ := schema["properties"].(map[string]interface{})
			if props != nil {
				for _, field := range tc.requiredFields {
					if _, ok := props[field]; !ok {
						t.Errorf("required field %q not found in properties", field)
					}
				}
			}
		})
	}
}

// TestToolParameterPropertyTypes verifies each property in the schema has a type.
func TestToolParameterPropertyTypes(t *testing.T) {
	for _, tc := range allTools {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			tool := tc.fn()
			var schema map[string]interface{}
			if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
				t.Fatalf("Parameters is not valid JSON: %v", err)
			}
			props, ok := schema["properties"].(map[string]interface{})
			if !ok {
				t.Fatal("properties is not an object")
			}
			for propName, propVal := range props {
				propObj, ok := propVal.(map[string]interface{})
				if !ok {
					t.Errorf("property %q is not an object", propName)
					continue
				}
				if _, ok := propObj["type"]; !ok {
					t.Errorf("property %q missing 'type' field", propName)
				}
				if _, ok := propObj["description"]; !ok {
					t.Errorf("property %q missing 'description' field", propName)
				}
			}
		})
	}
}

// TestToolNamesUnique verifies all tool names are distinct.
func TestToolNamesUnique(t *testing.T) {
	seen := make(map[string]string)
	for _, tc := range allTools {
		tool := tc.fn()
		if prev, ok := seen[tool.Name]; ok {
			t.Errorf("duplicate tool name %q: %s and %s", tool.Name, prev, tc.name)
		}
		seen[tool.Name] = tc.name
	}
}
