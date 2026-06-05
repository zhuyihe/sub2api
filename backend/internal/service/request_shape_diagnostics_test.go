package service

import (
	"strings"
	"testing"
)

func TestDiagnoseOpenAIChatCompletionsShapeReportsPathsWithoutContent(t *testing.T) {
	body := []byte(`{
		"messages": [
			{"role":"user","content":"SECRET_USER_TEXT"},
			{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"arguments":"SECRET_ARGUMENTS"}}]}
		],
		"tools": [{"type":"function","function":{"name":"lookup"}}]
	}`)
	responsesBody := []byte(`{
		"input": [
			{"role":"user","content":"SECRET_USER_TEXT"},
			{"type":"function_call","call_id":"call_1","arguments":"SECRET_ARGUMENTS"}
		]
	}`)

	diag := diagnoseOpenAIChatCompletionsShape(body, responsesBody)
	if !diag.hasFindings() {
		t.Fatal("expected diagnostics")
	}
	joined := strings.Join(diag.Findings, "\n")
	for _, want := range []string{
		"chat.messages.1.tool_calls.0.function.name:missing",
		"responses.input.1.name:missing",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("diagnostics missing %q: %v", want, diag.Findings)
		}
	}
	for _, forbidden := range []string{"SECRET_USER_TEXT", "SECRET_ARGUMENTS"} {
		if strings.Contains(joined, forbidden) {
			t.Fatalf("diagnostics leaked content %q: %v", forbidden, diag.Findings)
		}
	}
}

func TestDiagnoseAnthropicMessagesShapeReportsPathsWithoutContent(t *testing.T) {
	body := []byte(`{
		"max_tokens": 100,
		"thinking": {"type":"enabled","budget_tokens":100},
		"stream_options": {"include_usage": true},
		"messages": [
			{"role":"user","content":"SECRET_USER_TEXT","cache_control":{"type":"ephemeral"}}
		],
		"tools": [
			{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}},
			{"name":"transform","defer_loading":true,"cache_control":{"type":"ephemeral"}}
		]
	}`)

	diag := diagnoseAnthropicMessagesShape(body)
	if !diag.hasFindings() {
		t.Fatal("expected diagnostics")
	}
	joined := strings.Join(diag.Findings, "\n")
	for _, want := range []string{
		"anthropic.stream_options:present",
		"anthropic.messages.0.cache_control:present",
		"anthropic.tools.0.function:wrapped",
		"anthropic.tools.1.defer_loading_cache_control:conflict",
		"anthropic.thinking.budget_tokens:gte_max_tokens",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("diagnostics missing %q: %v", want, diag.Findings)
		}
	}
	if strings.Contains(joined, "SECRET_USER_TEXT") {
		t.Fatalf("diagnostics leaked content: %v", diag.Findings)
	}
}
