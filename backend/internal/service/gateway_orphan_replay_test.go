package service

import (
	"testing"

	"github.com/tidwall/gjson"
)

// TestPairOrphanToolResults_ProductionReplay 用真实失败请求的 tool_use_id 形态
// （含 toolu_bdrk_ 前缀 / TodoWrite_ 前缀）复现生产侧失败，看修复是否覆盖。
func TestPairOrphanToolResults_ProductionReplay(t *testing.T) {
	tests := []struct {
		name string
		body string
		// 期望转换后 messages.2.content.0 是 text 块
	}{
		{
			name: "toolu_bdrk_ 前缀（AWS Bedrock SDK 生成）",
			body: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":[{"type":"text","text":"thinking..."}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_bdrk_01LeMLVy4KvsbPW6sg1CwS3e","content":"x"}]}
			]}`,
		},
		{
			name: "工具名前缀 TodoWrite_207",
			body: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":[{"type":"text","text":"thinking..."}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"TodoWrite_207","content":"x"}]}
			]}`,
		},
		{
			name: "messages[2].content.0 但 messages[1] 含其他 tool_use 但 id 不同",
			body: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":[
					{"type":"tool_use","id":"toolu_OTHER","name":"Bash","input":{}}
				]},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"toolu_bdrk_01LeMLVy4KvsbPW6sg1CwS3e","content":"x"}
				]}
			]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, ok := pairOrphanToolResults([]byte(tt.body))
			if !ok {
				t.Fatalf("expected change, but pairOrphanToolResults returned false on body:\n%s", tt.body)
			}
			typ := gjson.GetBytes(out, "messages.2.content.0.type").String()
			if typ != "text" {
				t.Fatalf("messages.2.content.0.type=%q, want text", typ)
			}
		})
	}
}
