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

// TestStripCacheControlOnDeferLoadingTools 复现生产 Account=44 的 400：
// "Tool 'modify_mcp22' cannot have both defer_loading=true and cache_control set.
// Tools with defer_loading cannot use prompt caching."
// 工具同时带 defer_loading:true 与 cache_control 时上游直接拒；须删 cache_control
// (仅缓存优化，删之对功能无损)、保留 defer_loading(功能性语义)。
func TestStripCacheControlOnDeferLoadingTools(t *testing.T) {
	t.Run("defer_loading+cache_control -> 删 cache_control", func(t *testing.T) {
		body := `{"tools":[
			{"name":"modify_mcp22","defer_loading":true,"cache_control":{"type":"ephemeral"},"input_schema":{"type":"object"}}
		]}`
		out, ok := stripCacheControlOnDeferLoadingTools([]byte(body))
		if !ok {
			t.Fatal("expected change")
		}
		if gjson.GetBytes(out, "tools.0.cache_control").Exists() {
			t.Fatal("cache_control 应被删除")
		}
		if gjson.GetBytes(out, "tools.0.defer_loading").Bool() != true {
			t.Fatal("defer_loading 应保留")
		}
		if gjson.GetBytes(out, "tools.0.name").String() != "modify_mcp22" {
			t.Fatal("工具名应保留")
		}
	})

	t.Run("仅 defer_loading 无 cache_control -> 不动", func(t *testing.T) {
		body := `{"tools":[{"name":"t","defer_loading":true,"input_schema":{}}]}`
		if _, ok := stripCacheControlOnDeferLoadingTools([]byte(body)); ok {
			t.Fatal("无 cache_control 不应改动")
		}
	})

	t.Run("有 cache_control 但 defer_loading 非 true -> 不动(缓存合法)", func(t *testing.T) {
		body := `{"tools":[{"name":"t","cache_control":{"type":"ephemeral"},"input_schema":{}}]}`
		if _, ok := stripCacheControlOnDeferLoadingTools([]byte(body)); ok {
			t.Fatal("无 defer_loading 时 cache_control 合法，不应改动")
		}
	})

	t.Run("混合: 仅清洗冲突项, 合法缓存工具保留", func(t *testing.T) {
		body := `{"tools":[
			{"name":"a","defer_loading":true,"cache_control":{"type":"ephemeral"}},
			{"name":"b","cache_control":{"type":"ephemeral"}}
		]}`
		out, ok := stripCacheControlOnDeferLoadingTools([]byte(body))
		if !ok {
			t.Fatal("expected change")
		}
		if gjson.GetBytes(out, "tools.0.cache_control").Exists() {
			t.Fatal("冲突项 a 的 cache_control 应删")
		}
		if !gjson.GetBytes(out, "tools.1.cache_control").Exists() {
			t.Fatal("合法项 b 的 cache_control 应保留")
		}
	})

	t.Run("无 tools 字段不变", func(t *testing.T) {
		if _, ok := stripCacheControlOnDeferLoadingTools([]byte(`{"messages":[]}`)); ok {
			t.Fatal("无 tools 不应改动")
		}
	})
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
		t.Fatal("expected change")
	}
	// 孤儿 tool_result 应就地转 text 块；位置 (messages[2].content[0]) 与数组长度不变
	block := gjson.GetBytes(out, "messages.2.content.0")
	if block.Get("type").String() != "text" {
		t.Fatalf("expected type=text, got %s", block.Get("type").String())
	}
	want := "[tool_result toolu_orphan_1] x"
	if got := block.Get("text").String(); got != want {
		t.Fatalf("text=%q, want %q", got, want)
	}
	if n := len(gjson.GetBytes(out, "messages.2.content").Array()); n != 1 {
		t.Fatalf("content blocks=%d, want 1 (原地替换不改长度)", n)
	}
	// messages[1] 原 assistant 文本不应被改动（不再注入占位 tool_use）
	if n := len(gjson.GetBytes(out, "messages.1.content").Array()); n != 1 {
		t.Fatalf("messages[1].content blocks=%d, want 1 (不应追加占位)", n)
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
// 修复后：孤儿就地转 text 块，messages[1] 的原 tool_use 不被改动。
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
		t.Fatal("expected change: 跨越中间 assistant 的 tool_use 引用应判为孤儿")
	}
	// messages[2] 的 tool_result 合法（紧邻 messages[1] 声明 toolu_A），保留不动
	mid := gjson.GetBytes(out, "messages.2.content.0")
	if mid.Get("type").String() != "tool_result" {
		t.Fatalf("messages[2].content[0] 应保留 tool_result，got %s", mid.Raw)
	}
	// messages[4] 的 tool_result 在紧邻 messages[3] 中无声明 → 转 text
	tail := gjson.GetBytes(out, "messages.4.content.0")
	if tail.Get("type").String() != "text" {
		t.Fatalf("messages[4].content[0] 应转 text，got %s", tail.Raw)
	}
	want := "[tool_result toolu_A] 再次引用"
	if got := tail.Get("text").String(); got != want {
		t.Fatalf("text=%q, want %q", got, want)
	}
	// messages[1] 原 tool_use 不动（不再追加占位）
	if n := len(gjson.GetBytes(out, "messages.1.content").Array()); n != 1 {
		t.Fatalf("messages[1].content blocks=%d, want 1", n)
	}
	// messages[3] 原 assistant text 不动
	if n := len(gjson.GetBytes(out, "messages.3.content").Array()); n != 1 {
		t.Fatalf("messages[3].content blocks=%d, want 1", n)
	}
}

// TestPairOrphanToolResults_OrphanInMessages0 复现 12:55:43 实际 400：
// 孤儿 tool_result 出现在 messages[0]，根本没有可挂的前置消息。
// 修复后：i=0 也进入循环，就地转 text。
func TestPairOrphanToolResults_OrphanInMessages0(t *testing.T) {
	body := `{"messages":[
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01UGnaFy","content":"early result"}]},
		{"role":"assistant","content":[{"type":"text","text":"ok"}]}
	]}`
	out, ok := pairOrphanToolResults([]byte(body))
	if !ok {
		t.Fatal("expected change: messages[0] 孤儿 tool_result 必须被转换")
	}
	block := gjson.GetBytes(out, "messages.0.content.0")
	if block.Get("type").String() != "text" {
		t.Fatalf("expected type=text, got %s", block.Get("type").String())
	}
	want := "[tool_result toolu_01UGnaFy] early result"
	if got := block.Get("text").String(); got != want {
		t.Fatalf("text=%q, want %q", got, want)
	}
}

// TestPairOrphanToolResults_NoAssistantSequence 复现非交替序列：
// messages 全是 user，i=2 含孤儿 tool_result。修复后：转 text，不破坏长度。
func TestPairOrphanToolResults_NoAssistantSequence(t *testing.T) {
	body := `{"messages":[
		{"role":"user","content":"hi"},
		{"role":"user","content":"again"},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_X","content":"orphan"}]}
	]}`
	out, ok := pairOrphanToolResults([]byte(body))
	if !ok {
		t.Fatal("expected change: 无 assistant 序列中的孤儿应转 text")
	}
	block := gjson.GetBytes(out, "messages.2.content.0")
	if block.Get("type").String() != "text" {
		t.Fatalf("expected type=text, got %s", block.Get("type").String())
	}
	if got := block.Get("text").String(); got != "[tool_result toolu_X] orphan" {
		t.Fatalf("text=%q", got)
	}
}

// TestPairOrphanToolResults_ContentVariants 覆盖 tool_result.content 的多种形态：
// 字符串 / 文本块数组 / 含图像块 / 缺省。
func TestPairOrphanToolResults_ContentVariants(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "content 是块数组（多个 text 块拼接）",
			body: `{"messages":[{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"id1","content":[
					{"type":"text","text":"line1"},
					{"type":"text","text":"line2"}
				]}
			]}]}`,
			want: "[tool_result id1] line1\nline2",
		},
		{
			name: "content 含 image 块（兜底 non-text content omitted）",
			body: `{"messages":[{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"id2","content":[
					{"type":"image","source":{"type":"base64","data":"xx"}}
				]}
			]}]}`,
			want: "[tool_result id2] (non-text content omitted)",
		},
		{
			name: "content 缺省（兜底标记）",
			body: `{"messages":[{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"id3"}
			]}]}`,
			want: "[tool_result id3]",
		},
		{
			name: "content 为空字符串（兜底标记，不带空格）",
			body: `{"messages":[{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"id4","content":""}
			]}]}`,
			want: "[tool_result id4]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, ok := pairOrphanToolResults([]byte(tt.body))
			if !ok {
				t.Fatal("expected change")
			}
			block := gjson.GetBytes(out, "messages.0.content.0")
			if block.Get("type").String() != "text" {
				t.Fatalf("type=%s, want text", block.Get("type").String())
			}
			if got := block.Get("text").String(); got != tt.want {
				t.Fatalf("text=%q, want %q", got, tt.want)
			}
		})
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
