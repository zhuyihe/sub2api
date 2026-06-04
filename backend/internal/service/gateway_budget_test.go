package service

import (
	"testing"

	"github.com/tidwall/gjson"
)

// TestIsThinkingBudgetConstraintError 覆盖两种实际遇到的上游错误信息形态：
//   - flavour A: budget 太小，上游报 "budget_tokens >= 1024"
//   - flavour B: max_tokens ≤ budget_tokens，上游报
//     "`max_tokens` must be greater than `thinking.budget_tokens`"
//
// 同时验证反例：仅含 budget_tokens 但不在 thinking 上下文 → 不应匹配。
func TestIsThinkingBudgetConstraintError(t *testing.T) {
	tests := []struct {
		name string
		msg  string
		want bool
	}{
		{
			name: "flavour A: budget_tokens >= 1024",
			msg:  "thinking.budget_tokens: Input should be greater than or equal to 1024",
			want: true,
		},
		{
			name: "flavour A: >= 1024 简写",
			msg:  "thinking.budget_tokens must be >= 1024",
			want: true,
		},
		{
			name: "flavour A: input should be + 1024",
			msg:  "thinking.budget_tokens: input should be ge 1024",
			want: true,
		},
		{
			name: "flavour B (本次 user 104 实际遇到): max_tokens > budget_tokens",
			msg: "`max_tokens` must be greater than `thinking.budget_tokens`. " +
				"Please consult our documentation at https://docs.claude.com/en/docs/build-with-claude/extended-thinking#max-tokens-and-context-window-size",
			want: true,
		},
		{
			name: "flavour B 大小写不敏感",
			msg:  "MAX_TOKENS must be GREATER than THINKING.BUDGET_TOKENS",
			want: true,
		},
		{
			name: "反例: budget_tokens 但无 thinking 上下文",
			msg:  "invalid budget_tokens parameter, must be a number",
			want: false,
		},
		{
			name: "反例: thinking 但无 budget_tokens",
			msg:  "thinking enabled but no budget specified",
			want: false,
		},
		{
			name: "反例: 完全无关",
			msg:  "rate limit exceeded",
			want: false,
		},
		{
			name: "反例: 只有 budget+thinking 但无任何约束指示符",
			msg:  "thinking.budget_tokens is malformed",
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isThinkingBudgetConstraintError(tt.msg)
			if got != tt.want {
				t.Fatalf("isThinkingBudgetConstraintError(%q) = %v, want %v", tt.msg, got, tt.want)
			}
		})
	}
}

// TestRectifyThinkingBudget_MaxTokensTooSmall 验证 max_tokens < budget_tokens 时整流后
// 同时调整两个字段，让 max_tokens (64000) 严格大于 budget_tokens (32000)。
func TestRectifyThinkingBudget_MaxTokensTooSmall(t *testing.T) {
	// 客户端原本 budget=50000, max=8192 → 50000 ≥ 8192 触发上游 400
	body := []byte(`{"model":"claude-opus-4-6","thinking":{"type":"enabled","budget_tokens":50000},"max_tokens":8192,"messages":[{"role":"user","content":"hi"}]}`)
	out, changed := RectifyThinkingBudget(body)
	if !changed {
		t.Fatal("expected rectification to apply")
	}
	// 强制 budget 整为 32000，max 整为 64000，恒满足 max > budget
	if got := gjson.GetBytes(out, "thinking.budget_tokens").Int(); got != 32000 {
		t.Fatalf("thinking.budget_tokens=%d, want 32000", got)
	}
	if got := gjson.GetBytes(out, "max_tokens").Int(); got != 64000 {
		t.Fatalf("max_tokens=%d, want 64000", got)
	}
	if got := gjson.GetBytes(out, "thinking.type").String(); got != "enabled" {
		t.Fatalf("thinking.type=%q, want enabled", got)
	}
}

// TestRectifyThinkingBudget_AdaptiveSkipped 保留行为：thinking.type=="adaptive" 时不整流。
func TestRectifyThinkingBudget_AdaptiveSkipped(t *testing.T) {
	body := []byte(`{"thinking":{"type":"adaptive","budget_tokens":50000},"max_tokens":100,"messages":[]}`)
	out, changed := RectifyThinkingBudget(body)
	if changed {
		t.Fatal("adaptive 类型不应整流")
	}
	if got := gjson.GetBytes(out, "thinking.type").String(); got != "adaptive" {
		t.Fatalf("thinking.type=%q, want adaptive", got)
	}
	if got := gjson.GetBytes(out, "thinking.budget_tokens").Int(); got != 50000 {
		t.Fatalf("thinking.budget_tokens=%d, want 50000", got)
	}
	if got := gjson.GetBytes(out, "max_tokens").Int(); got != 100 {
		t.Fatalf("max_tokens=%d, want 100", got)
	}
}
