package service

import (
	"testing"

	"github.com/tidwall/gjson"
)

// TestPairOrphanToolUses 覆盖 "tool_use ids were found without tool_result blocks
// immediately after" 400：assistant 的 tool_use 块在紧邻下一条消息无对应 tool_result。
func TestPairOrphanToolUses(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantOK    bool
		assertion func(t *testing.T, out []byte)
	}{
		{
			name: "末尾 assistant tool_use 无后续(本次 user 实际遇到形态)",
			body: `{"messages":[` +
				`{"role":"user","content":"改文件"},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_A","name":"replace_file_content","input":{"path":"a.go"}}]}` +
				`]}`,
			wantOK: true,
			assertion: func(t *testing.T, out []byte) {
				// tool_use 应被转 text，不再有 tool_use 块
				if gjson.GetBytes(out, "messages.1.content.0.type").String() != "text" {
					t.Fatalf("orphan tool_use 应转 text, got type=%s",
						gjson.GetBytes(out, "messages.1.content.0.type").String())
				}
				// 工具名与入参保留在 text 中
				txt := gjson.GetBytes(out, "messages.1.content.0.text").String()
				if txt == "" {
					t.Fatal("转换后 text 不应为空")
				}
			},
		},
		{
			name: "下一条 user 无对应 tool_result -> 转 text",
			body: `{"messages":[` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_X","name":"foo","input":{}}]},` +
				`{"role":"user","content":"继续"}` +
				`]}`,
			wantOK: true,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "messages.0.content.0.type").String() != "text" {
					t.Fatal("orphan tool_use 应转 text")
				}
			},
		},
		{
			name: "合法配对不动(tool_use + 紧邻 tool_result)",
			body: `{"messages":[` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_A","name":"foo","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_A","content":"ok"}]}` +
				`]}`,
			wantOK: false,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "messages.0.content.0.type").String() != "tool_use" {
					t.Fatal("合法 tool_use 应保留")
				}
			},
		},
		{
			name: "混合 text+tool_use: 仅转孤儿 tool_use,保留 text",
			body: `{"messages":[` +
				`{"role":"assistant","content":[{"type":"text","text":"思考中"},{"type":"tool_use","id":"toolu_B","name":"bar","input":{"k":"v"}}]},` +
				`{"role":"user","content":"hi"}` +
				`]}`,
			wantOK: true,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "messages.0.content.0.text").String() != "思考中" {
					t.Fatal("原 text 块应保留")
				}
				if gjson.GetBytes(out, "messages.0.content.1.type").String() != "text" {
					t.Fatal("孤儿 tool_use 应转 text")
				}
			},
		},
		{
			name: "部分配对: 一个有 result 一个没有",
			body: `{"messages":[` +
				`{"role":"assistant","content":[` +
				`{"type":"tool_use","id":"toolu_P","name":"p","input":{}},` +
				`{"type":"tool_use","id":"toolu_Q","name":"q","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_P","content":"ok"}]}` +
				`]}`,
			wantOK: true,
			assertion: func(t *testing.T, out []byte) {
				// P 被回应 -> 保留 tool_use
				if gjson.GetBytes(out, "messages.0.content.0.type").String() != "tool_use" {
					t.Fatal("已回应的 tool_use(P) 应保留")
				}
				// Q 未回应 -> 转 text
				if gjson.GetBytes(out, "messages.0.content.1.type").String() != "text" {
					t.Fatal("未回应的 tool_use(Q) 应转 text")
				}
			},
		},
		{
			name: "下一条是 assistant(中间无 user tool_result) -> 转 text",
			body: `{"messages":[` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_C","name":"c","input":{}}]},` +
				`{"role":"assistant","content":"prefill"}` +
				`]}`,
			wantOK: true,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "messages.0.content.0.type").String() != "text" {
					t.Fatal("无 tool_result 回应的 tool_use 应转 text")
				}
			},
		},
		{
			name:   "无 tool_use 字段不变",
			body:   `{"messages":[{"role":"user","content":"hi"}]}`,
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, ok := pairOrphanToolUses([]byte(tt.body))
			if ok != tt.wantOK {
				t.Fatalf("pairOrphanToolUses ok=%v, want %v", ok, tt.wantOK)
			}
			if !gjson.ValidBytes(out) {
				t.Fatalf("输出非合法 JSON: %s", out)
			}
			if tt.assertion != nil {
				tt.assertion(t, out)
			}
		})
	}
}

// TestDegradeAnthropicRequestParams_OrphanToolUse 验证孤儿 tool_use 处理已接入 pipeline
// 并产生 degrade 字段标记（Passthrough 路径依赖此预防性步骤）。
func TestDegradeAnthropicRequestParams_OrphanToolUse(t *testing.T) {
	body := []byte(`{"model":"claude-opus-4-7","messages":[` +
		`{"role":"user","content":"改文件"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_A","name":"replace_file_content","input":{"path":"a.go"}}]}` +
		`]}`)
	out, fields := DegradeAnthropicRequestParams(body, "claude-opus-4-7")

	has := false
	for _, f := range fields {
		if f == "orphan_tool_use:paired" {
			has = true
		}
	}
	if !has {
		t.Fatalf("expected orphan_tool_use:paired in degraded fields, got %v", fields)
	}
	// 整流后不应再有未配对 tool_use 块
	if gjson.GetBytes(out, "messages.1.content.0.type").String() == "tool_use" {
		t.Fatal("pipeline 整流后 orphan tool_use 仍存在")
	}
}
