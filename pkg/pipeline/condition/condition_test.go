package condition

import (
	"testing"

	"github.com/ashka-vakil/attractor/pkg/pipeline"
)

func TestEvaluateEmpty(t *testing.T) {
	if !Evaluate("", nil, pipeline.NewContext()) {
		t.Error("empty condition should return true")
	}
}

func TestEvaluateOutcomeEquals(t *testing.T) {
	outcome := &pipeline.Outcome{Status: pipeline.StatusSuccess}
	ctx := pipeline.NewContext()

	if !Evaluate("outcome=success", outcome, ctx) {
		t.Error("expected outcome=success to be true")
	}
	if Evaluate("outcome=fail", outcome, ctx) {
		t.Error("expected outcome=fail to be false")
	}
}

func TestEvaluateOutcomeNotEquals(t *testing.T) {
	outcome := &pipeline.Outcome{Status: pipeline.StatusSuccess}
	ctx := pipeline.NewContext()

	if !Evaluate("outcome!=fail", outcome, ctx) {
		t.Error("expected outcome!=fail to be true")
	}
	if Evaluate("outcome!=success", outcome, ctx) {
		t.Error("expected outcome!=success to be false")
	}
}

func TestEvaluateANDConjunction(t *testing.T) {
	outcome := &pipeline.Outcome{Status: pipeline.StatusSuccess}
	ctx := pipeline.NewContext()
	ctx.Set("tests_passed", "true")

	if !Evaluate("outcome=success && context.tests_passed=true", outcome, ctx) {
		t.Error("expected AND with both true to be true")
	}
	if Evaluate("outcome=success && context.tests_passed=false", outcome, ctx) {
		t.Error("expected AND with one false to be false")
	}
}

func TestEvaluatePreferredLabel(t *testing.T) {
	outcome := &pipeline.Outcome{
		Status:         pipeline.StatusSuccess,
		PreferredLabel: "Fix",
	}
	ctx := pipeline.NewContext()

	if !Evaluate("preferred_label=Fix", outcome, ctx) {
		t.Error("expected preferred_label=Fix to be true")
	}
	if Evaluate("preferred_label=Approve", outcome, ctx) {
		t.Error("expected preferred_label=Approve to be false")
	}
}

func TestEvaluateContextValues(t *testing.T) {
	outcome := &pipeline.Outcome{Status: pipeline.StatusSuccess}
	ctx := pipeline.NewContext()
	ctx.Set("loop_state", "active")

	if !Evaluate("context.loop_state=active", outcome, ctx) {
		t.Error("expected context.loop_state=active to be true")
	}
	if Evaluate("context.loop_state=exhausted", outcome, ctx) {
		t.Error("expected context.loop_state=exhausted to be false")
	}
}

func TestEvaluateMissingContextKey(t *testing.T) {
	outcome := &pipeline.Outcome{Status: pipeline.StatusSuccess}
	ctx := pipeline.NewContext()

	if Evaluate("context.nonexistent=value", outcome, ctx) {
		t.Error("missing key should not equal any value")
	}
	if !Evaluate("context.nonexistent!=value", outcome, ctx) {
		t.Error("missing key should be != any non-empty value")
	}
}

func TestValidateConditionValid(t *testing.T) {
	cases := []string{
		"outcome=success",
		"outcome!=fail",
		"outcome=success && context.x=y",
		"",
	}
	for _, c := range cases {
		if err := Validate(c); err != nil {
			t.Errorf("expected valid condition %q, got error: %v", c, err)
		}
	}
}

func TestValidateConditionInvalid(t *testing.T) {
	cases := []string{
		"=value",
		"!=value",
	}
	for _, c := range cases {
		if err := Validate(c); err == nil {
			t.Errorf("expected invalid condition %q to fail validation", c)
		}
	}
}
