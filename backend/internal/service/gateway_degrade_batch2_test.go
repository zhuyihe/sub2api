package service

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestBackfillMissingTools(t *testing.T) {
	// tool_use 引用 TaskCreate/TodoWrite，tools 只声明了 TaskCreate
	body := `{
		"tools":[{"name":"TaskCreate","input_schema":{"type":"object"}}],
		"messages":[{"role":"assistant","content":[
			{"type":"tool_use","id":"a","name":"TaskCreate","input":{}},
			{"type":"tool_use","id":"b","name":"TodoWrite","input":{}}
		]}]
	}`
	out, ok := backfillMissingTools([]byte(body))
	if !ok {
		t.Fatal("expected backfill")
	}
	names := gjson.GetBytes(out, "tools.#.name").Array()
	if len(names) != 2 {
		t.Fatalf("tools count=%d, want 2", len(names))
	}
	found := false
	for _, n := range names {
		if n.String() == "TodoWrite" {
			found = true
		}
	}
	if !found {
		t.Fatal("TodoWrite placeholder not added")
	}
	// 全部已声明时不动
	if _, ok := backfillMissingTools([]byte(`{"tools":[{"name":"X"}],"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"X","input":{}}]}]}`)); ok {
		t.Fatal("all-declared should not change")
	}
}

func TestAppendUserForAssistantPrefill(t *testing.T) {
	body := `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"prefix"}]}`
	out, ok := appendUserForAssistantPrefill([]byte(body))
	if !ok {
		t.Fatal("expected append")
	}
	msgs := gjson.GetBytes(out, "messages").Array()
	if len(msgs) != 3 {
		t.Fatalf("messages=%d, want 3", len(msgs))
	}
	if last := msgs[2]; last.Get("role").String() != "user" || last.Get("content").String() != "Continue." {
		t.Fatalf("appended message wrong: %s", msgs[2].Raw)
	}
	// 原 assistant prefill 仍保留
	if msgs[1].Get("role").String() != "assistant" {
		t.Fatal("original assistant prefill should be kept")
	}
	// 末条 user 不动
	if _, ok := appendUserForAssistantPrefill([]byte(`{"messages":[{"role":"user","content":"hi"}]}`)); ok {
		t.Fatal("user-ending should not change")
	}
}

func TestNormalizeCacheControlTTL(t *testing.T) {
	body := `{"messages":[{"role":"user","content":[
		{"type":"text","text":"a","cache_control":{"type":"ephemeral","ttl":"1h"}},
		{"type":"text","text":"b","cache_control":{"type":"ephemeral","ttl":"5m"}}
	]}]}`
	out, ok := normalizeCacheControlTTL([]byte(body))
	if !ok {
		t.Fatal("expected ttl normalize")
	}
	if gjson.GetBytes(out, `messages.0.content.0.cache_control.ttl`).Exists() {
		t.Fatal("ttl should be removed")
	}
	// cache_control 本身保留
	if gjson.GetBytes(out, `messages.0.content.0.cache_control.type`).String() != "ephemeral" {
		t.Fatal("cache_control type should remain")
	}
	// 无 ttl 时不动
	if _, ok := normalizeCacheControlTTL([]byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]}]}`)); ok {
		t.Fatal("no-ttl should not change")
	}
}
