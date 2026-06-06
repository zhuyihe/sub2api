package service

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestDegradeAnthropicRequestParams(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		model     string
		wantNum   int // 期望发生的降级项数量
		assertion func(t *testing.T, out []byte)
	}{
		{
			name:    "effort xhigh 顶层 reasoning_effort 改 max",
			body:    `{"model":"claude-opus-4-8","reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-opus-4-8",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if got := gjson.GetBytes(out, "reasoning_effort").String(); got != "max" {
					t.Fatalf("reasoning_effort=%q, want max", got)
				}
			},
		},
		{
			name:    "effort xhigh output_config.effort 改 max",
			body:    `{"model":"claude-opus-4-8","output_config":{"effort":"xhigh"},"messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-opus-4-8",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if got := gjson.GetBytes(out, "output_config.effort").String(); got != "max" {
					t.Fatalf("output_config.effort=%q, want max", got)
				}
			},
		},
		{
			name:    "wrapped custom tool invalid name 被扁平并清洗",
			body:    `{"model":"claude-opus-4-8","tools":[{"type":"custom","custom":{"name":"bad.tool/name","input_schema":{"type":"object"}}}],"tool_choice":{"type":"tool","name":"bad.tool/name"},"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"bad.tool/name","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]}]}`,
			model:   "claude-opus-4-8",
			wantNum: 2,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "tools.0.custom").Exists() {
					t.Fatal("custom wrapper should be removed")
				}
				for _, path := range []string{"tools.0.name", "tool_choice.name", "messages.1.content.0.name"} {
					if got := gjson.GetBytes(out, path).String(); got != "bad_tool_name" {
						t.Fatalf("%s=%q, want bad_tool_name", path, got)
					}
				}
			},
		},
		{
			name:    "4.x 模型 temperature 与 top_p 均弃用,全部删除",
			body:    `{"model":"claude-sonnet-4-6","temperature":0.7,"top_p":0.9,"messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-sonnet-4-6",
			wantNum: 2, // 4.x: top_p 删除 + temperature 删除
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "temperature").Exists() {
					t.Fatal("temperature should be removed")
				}
				if gjson.GetBytes(out, "top_p").Exists() {
					t.Fatal("top_p should be removed")
				}
			},
		},
		{
			name:    "4.x 模型删 deprecated temperature(无 top_p)",
			body:    `{"model":"claude-opus-4-7","temperature":0.5,"messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-opus-4-7",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "temperature").Exists() {
					t.Fatal("temperature should be removed for 4.x")
				}
			},
		},
		{
			name:    "4.x 模型删 top_k",
			body:    `{"model":"claude-opus-4-7","top_k":40,"messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-opus-4-7",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "top_k").Exists() {
					t.Fatal("top_k should be removed for 4.x")
				}
			},
		},
		{
			name:    "4.x 模型同时删 top_k 和 top_p",
			body:    `{"model":"claude-opus-4-7","top_k":40,"top_p":0.9,"messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-opus-4-7",
			wantNum: 2,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "top_k").Exists() || gjson.GetBytes(out, "top_p").Exists() {
					t.Fatal("top_k and top_p should both be removed for 4.x")
				}
			},
		},
		{
			name:    "thinking.adaptive.effort 完全删除(Extra inputs not permitted)",
			body:    `{"model":"claude-opus-4-7","thinking":{"type":"enabled","adaptive":{"effort":"xhigh"}},"messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-opus-4-7",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "thinking.adaptive.effort").Exists() {
					t.Fatal("thinking.adaptive.effort should be removed")
				}
				// thinking 本身保留
				if !gjson.GetBytes(out, "thinking.type").Exists() {
					t.Fatal("thinking.type should be kept")
				}
			},
		},
		{
			name:    "顶层 speed 字段删除(Extra inputs not permitted, user 104 实际遇到)",
			body:    `{"model":"claude-opus-4-7","speed":"fast","messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-opus-4-7",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "speed").Exists() {
					t.Fatal("top-level speed should be removed")
				}
			},
		},
		{
			name:    "OpenAI 顶层 stream_options 和 group 字段删除",
			body:    `{"model":"claude-opus-4-8","stream_options":{"include_usage":true},"group":"debug","messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-opus-4-8",
			wantNum: 2,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "stream_options").Exists() {
					t.Fatal("stream_options should be removed")
				}
				if gjson.GetBytes(out, "group").Exists() {
					t.Fatal("group should be removed")
				}
			},
		},
		{
			name:    "message 对象级 name 字段删除但保留工具块 name",
			body:    `{"model":"claude-sonnet-4-6","tools":[{"name":"run_cmd","input_schema":{"type":"object"}}],"messages":[{"role":"user","name":"alice","content":"hi"},{"role":"assistant","name":"bot","content":[{"type":"tool_use","id":"toolu_1","name":"run_cmd","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}]}`,
			model:   "claude-sonnet-4-6",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if gjson.GetBytes(out, "messages.0.name").Exists() {
					t.Fatal("messages.0.name should be removed")
				}
				if gjson.GetBytes(out, "messages.1.name").Exists() {
					t.Fatal("messages.1.name should be removed")
				}
				if got := gjson.GetBytes(out, "messages.1.content.0.name").String(); got != "run_cmd" {
					t.Fatalf("tool_use name=%q, want run_cmd", got)
				}
			},
		},
		{
			name:    "老模型保留 top_k",
			body:    `{"model":"claude-3-5-sonnet","top_k":40,"messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-3-5-sonnet",
			wantNum: 0,
			assertion: func(t *testing.T, out []byte) {
				if !gjson.GetBytes(out, "top_k").Exists() {
					t.Fatal("top_k should be kept for legacy model")
				}
			},
		},
		{
			name:    "老模型保留 temperature",
			body:    `{"model":"claude-3-5-sonnet","temperature":0.5,"messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-3-5-sonnet",
			wantNum: 0,
			assertion: func(t *testing.T, out []byte) {
				if !gjson.GetBytes(out, "temperature").Exists() {
					t.Fatal("temperature should be kept for legacy model")
				}
			},
		},
		{
			name:    "role system 合并进顶层 system(原无 system)",
			body:    `{"model":"claude-sonnet-4-6","messages":[{"role":"system","content":"你是助手"},{"role":"user","content":"hi"}]}`,
			model:   "claude-sonnet-4-6",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if got := gjson.GetBytes(out, "system").String(); got != "你是助手" {
					t.Fatalf("system=%q, want 你是助手", got)
				}
				if n := len(gjson.GetBytes(out, "messages").Array()); n != 1 {
					t.Fatalf("messages len=%d, want 1", n)
				}
				if role := gjson.GetBytes(out, "messages.0.role").String(); role != "user" {
					t.Fatalf("first message role=%q, want user", role)
				}
			},
		},
		{
			name:    "role system 前置合并进已有字符串 system",
			body:    `{"model":"claude-sonnet-4-6","system":"原始","messages":[{"role":"system","content":"附加"},{"role":"user","content":"hi"}]}`,
			model:   "claude-sonnet-4-6",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if got := gjson.GetBytes(out, "system").String(); got != "附加\n\n原始" {
					t.Fatalf("system=%q, want 附加\\n\\n原始", got)
				}
			},
		},
		{
			name:    "role system content 为内容块数组",
			body:    `{"model":"claude-sonnet-4-6","messages":[{"role":"system","content":[{"type":"text","text":"块文本"}]},{"role":"user","content":"hi"}]}`,
			model:   "claude-sonnet-4-6",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				if got := gjson.GetBytes(out, "system").String(); got != "块文本" {
					t.Fatalf("system=%q, want 块文本", got)
				}
			},
		},
		{
			name:    "顶层 system 为数组时前置 text 块",
			body:    `{"model":"claude-sonnet-4-6","system":[{"type":"text","text":"原块"}],"messages":[{"role":"system","content":"新"},{"role":"user","content":"hi"}]}`,
			model:   "claude-sonnet-4-6",
			wantNum: 1,
			assertion: func(t *testing.T, out []byte) {
				arr := gjson.GetBytes(out, "system").Array()
				if len(arr) != 2 {
					t.Fatalf("system array len=%d, want 2", len(arr))
				}
				if got := gjson.GetBytes(out, "system.0.text").String(); got != "新" {
					t.Fatalf("system[0].text=%q, want 新", got)
				}
			},
		},
		{
			name:    "role system 含非文本内容时保持不动(不丢数据)",
			body:    `{"model":"claude-sonnet-4-6","messages":[{"role":"system","content":[{"type":"image","source":{"type":"base64","data":"x"}}]},{"role":"user","content":"hi"}]}`,
			model:   "claude-sonnet-4-6",
			wantNum: 0,
			assertion: func(t *testing.T, out []byte) {
				if n := len(gjson.GetBytes(out, "messages").Array()); n != 2 {
					t.Fatalf("messages len=%d, want 2 (system 消息应保留)", n)
				}
			},
		},
		{
			name:    "无需降级时 body 不变(老模型保留 top_p)",
			body:    `{"model":"claude-3-5-sonnet-20241022","top_p":0.9,"system":"x","messages":[{"role":"user","content":"hi"}]}`,
			model:   "claude-3-5-sonnet-20241022",
			wantNum: 0,
			assertion: func(t *testing.T, out []byte) {
				if !gjson.GetBytes(out, "top_p").Exists() {
					t.Fatal("top_p should remain for legacy model")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, fields := DegradeAnthropicRequestParams([]byte(tt.body), tt.model)
			if len(fields) != tt.wantNum {
				t.Fatalf("degraded fields=%d %v, want %d", len(fields), fields, tt.wantNum)
			}
			if !gjson.ValidBytes(out) {
				t.Fatalf("output is not valid JSON: %s", out)
			}
			if tt.assertion != nil {
				tt.assertion(t, out)
			}
		})
	}
}
