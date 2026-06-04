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

func TestPairOrphanToolResults(t *testing.T) {
	// messages[1](assistant) 无 tool_use; messages[2](user) 含孤儿 tool_result
	body := `{"messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":[{"type":"text","text":"thinking..."}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_orphan_1","content":"x"}]}
	]}`
	out, ok := pairOrphanToolResults([]byte(body))
	if !ok {
		t.Fatal("expected pair")
	}
	// messages[1].content 末尾应有 tool_use id 匹配
	blocks := gjson.GetBytes(out, "messages.1.content").Array()
	if len(blocks) != 2 {
		t.Fatalf("messages[1].content blocks=%d, want 2(原 text+占位 tool_use)", len(blocks))
	}
	added := blocks[1]
	if added.Get("type").String() != "tool_use" || added.Get("id").String() != "toolu_orphan_1" {
		t.Fatalf("placeholder tool_use 不正确: %s", added.Raw)
	}
	if added.Get("name").String() != placeholderOrphanToolUseName {
		t.Fatal("placeholder name 不匹配")
	}

	// 已正确配对时不动
	paired := `{"messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_A","name":"x","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_A","content":"x"}]}
	]}`
	if _, ok := pairOrphanToolResults([]byte(paired)); ok {
		t.Fatal("已配对时不应改动")
	}
}

// TestPairOrphanToolResults_CrossAssistantReference 复现 user 104 实际 400：
// 早期 assistant 声明过 tool_use，但紧邻前一条 assistant 未声明同名 tool_use，
// 再次出现的 tool_result 必须被视为孤儿（Anthropic 严格按 previous message 校验）。
func TestPairOrphanToolResults_CrossAssistantReference(t *testing.T) {
	body := `{"messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_A","name":"x","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_A","content":"ok"}]},
		{"role":"assistant","content":[{"type":"text","text":"done"}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_A","content":"再次引用"}]}
	]}`
	out, ok := pairOrphanToolResults([]byte(body))
	if !ok {
		t.Fatal("expected pair: 跨越中间 assistant 的 tool_use 引用应判为孤儿")
	}
	// messages[3](紧邻 messages[4] 的 assistant) 末尾应被追加占位 tool_use A
	blocks := gjson.GetBytes(out, "messages.3.content").Array()
	if len(blocks) != 2 {
		t.Fatalf("messages[3].content blocks=%d, want 2(原 text+占位 tool_use)", len(blocks))
	}
	added := blocks[1]
	if added.Get("type").String() != "tool_use" || added.Get("id").String() != "toolu_A" {
		t.Fatalf("placeholder tool_use 不正确: %s", added.Raw)
	}
	if added.Get("name").String() != placeholderOrphanToolUseName {
		t.Fatal("placeholder name 不匹配")
	}
	// messages[1] 原 tool_use 不动
	orig := gjson.GetBytes(out, "messages.1.content.0")
	if orig.Get("name").String() == placeholderOrphanToolUseName {
		t.Fatal("原 messages[1] 的 tool_use 被误改")
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
