package service

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestTruncateOversizedPrompt(t *testing.T) {
	// 构造一个超过阈值的 body：很多条消息，每条约 8KB
	var sb strings.Builder
	sb.WriteString(`{"system":"sys","messages":[`)
	const n = 120
	chunk := strings.Repeat("x", 8*1024)
	for i := 0; i < n; i++ {
		if i > 0 {
			sb.WriteByte(',')
		}
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		sb.WriteString(`{"role":"` + role + `","content":"` + chunk + `"}`)
	}
	sb.WriteString(`]}`)
	body := []byte(sb.String())

	if len(body) <= maxPromptBodyBytes {
		t.Fatalf("test body too small: %d <= %d", len(body), maxPromptBodyBytes)
	}

	out, ok := truncateOversizedPrompt(body)
	if !ok {
		t.Fatal("expected truncation")
	}
	if len(out) >= len(body) {
		t.Fatalf("truncated size %d not smaller than %d", len(out), len(body))
	}
	if !gjson.ValidBytes(out) {
		t.Fatal("truncated body invalid json")
	}
	// 首条必须为 user
	if gjson.GetBytes(out, "messages.0.role").String() != "user" {
		t.Fatalf("first message role=%s, want user", gjson.GetBytes(out, "messages.0.role").String())
	}
	// system 保留
	if gjson.GetBytes(out, "system").String() != "sys" {
		t.Fatal("system should be preserved")
	}
	// 消息数应减少
	if n2 := len(gjson.GetBytes(out, "messages").Array()); n2 >= n {
		t.Fatalf("messages not reduced: %d", n2)
	}
}

func TestTruncateOversizedPrompt_SmallNoop(t *testing.T) {
	body := []byte(`{"system":"s","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},{"role":"user","content":"bye"}]}`)
	if _, ok := truncateOversizedPrompt(body); ok {
		t.Fatal("small body should not be truncated")
	}
}
