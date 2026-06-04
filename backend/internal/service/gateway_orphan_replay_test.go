package service

import (
	"fmt"
	"strings"
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

// TestPairOrphanToolResults_NonAdjacentPrev 复现生产 user 104 的
// "messages.6.content.1: unexpected tool_use_id ... must have a corresponding
// tool_use block in the previous message" 400。
//
// 拓扑：messages[3] 的 tool_result 紧邻前一条(messages[2])是另一条 user(非 assistant)，
// 但更早的 assistant(messages[1])声明了同一 tool_use_id。旧实现经 findPrevAssistantIdx
// 向前跳过非 assistant，命中早期 assistant 并误判「已声明→保留」；而 Anthropic 严格按
// 紧邻前一条消息校验 → 上游 400。修复后须严格只看 messages[i-1]，把该孤儿转 text。
func TestPairOrphanToolResults_NonAdjacentPrev(t *testing.T) {
	body := `{"messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_01Mov","name":"x","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01Mov","content":"ok"}]},
		{"role":"user","content":[{"type":"text","text":"continue"},{"type":"tool_result","tool_use_id":"toolu_01Mov","content":"dup"}]}
	]}`
	out, ok := pairOrphanToolResults([]byte(body))
	if !ok {
		t.Fatal("expected change: messages[3] 的 tool_result 紧邻前一条是 user，应判孤儿并转 text")
	}
	// messages[2] 合法配对(prev=messages[1] assistant 声明了 toolu_01Mov)，保留
	if got := gjson.GetBytes(out, "messages.2.content.0.type").String(); got != "tool_result" {
		t.Fatalf("messages[2] 合法 tool_result 应保留, got=%s", got)
	}
	// messages[3].content[1] 紧邻前一条是 user(messages[2])，按 Anthropic 应判孤儿 → text
	if got := gjson.GetBytes(out, "messages.3.content.1.type").String(); got != "text" {
		t.Fatalf("messages[3].content[1] 应转 text, got=%s", got)
	}
}

// TestDegradeAnthropicRequestParams_TruncateThenOrphan 复现 user 104 真实 case：
// body 含合法配对 + 整体超过 maxPromptBodyBytes (650KB)，触发 truncateOversizedPrompt
// 从尾部保留消息。截断丢掉了原本声明 tool_use 的 assistant 消息，留下的 tool_result
// 在新 messages[1] 中找不到对应 tool_use → 必须 step 13a 再跑一次 pairOrphanToolResults
// 把孤儿转 text 块，否则上游 400。
//
// 构造方法：制造一个 ~1MB 的对话，结尾恰好留下「孤儿 tool_result + 截断前合法
// 的早期 tool_use」。验证最终结果里 tool_result 已被转 text 块（不再有 tool_use_id）。
func TestDegradeAnthropicRequestParams_TruncateThenOrphan(t *testing.T) {
	// 制造一条 50KB 的填充文本，让前几轮消息体积膨胀
	filler := strings.Repeat("x", 50*1024)

	// 26 轮 user/assistant 配对，每轮 ~100KB → 总 ~2.6MB，远超 650KB 阈值
	var msgs []string
	for i := 0; i < 26; i++ {
		toolID := fmt.Sprintf("toolu_orphan_round_%02d", i)
		msgs = append(msgs,
			fmt.Sprintf(`{"role":"user","content":[{"type":"text","text":%q}]}`, filler),
			fmt.Sprintf(`{"role":"assistant","content":[{"type":"tool_use","id":%q,"name":"x","input":{}}]}`, toolID),
			fmt.Sprintf(`{"role":"user","content":[{"type":"tool_result","tool_use_id":%q,"content":"early result"}]}`, toolID),
		)
	}
	// 末尾再追加最近一轮（这一轮应在截断后保留）
	finalToolID := "toolu_final_ROUND"
	msgs = append(msgs,
		fmt.Sprintf(`{"role":"assistant","content":[{"type":"text","text":"thinking"},{"type":"tool_use","id":%q,"name":"x","input":{}}]}`, finalToolID),
		fmt.Sprintf(`{"role":"user","content":[{"type":"tool_result","tool_use_id":%q,"content":"final ok"}]}`, finalToolID),
	)
	body := []byte(fmt.Sprintf(`{"model":"claude-opus-4-7","max_tokens":1024,"messages":[%s]}`, strings.Join(msgs, ",")))

	if len(body) <= maxPromptBodyBytes {
		t.Fatalf("test setup error: body_len=%d not exceeding maxPromptBodyBytes=%d", len(body), maxPromptBodyBytes)
	}

	out, fields := DegradeAnthropicRequestParams(body, "claude-opus-4-7")

	// 确认 oversized_prompt:truncated 触发了
	hasTrunc := false
	hasPostTruncPair := false
	for _, f := range fields {
		if f == "oversized_prompt:truncated" {
			hasTrunc = true
		}
		if f == "orphan_tool_result:post_truncate_paired" {
			hasPostTruncPair = true
		}
	}
	if !hasTrunc {
		t.Fatalf("expected oversized_prompt:truncated in degraded fields, got %v", fields)
	}
	if !hasPostTruncPair {
		t.Fatalf("expected orphan_tool_result:post_truncate_paired (step 13a) in degraded fields, got %v", fields)
	}

	// 截断后的 messages 中所有 tool_result 块都必须有合法 prev assistant 的 tool_use
	// 配对，否则上游 400。简化校验：扫每条含 tool_result 的 user 消息，确认其 tool_use_id
	// 在紧邻前一条 assistant 中声明了。
	msgsArr := gjson.GetBytes(out, "messages").Array()
	if len(msgsArr) < 2 {
		t.Fatalf("messages after degrade len=%d, too few", len(msgsArr))
	}
	for i := 1; i < len(msgsArr); i++ {
		curr := msgsArr[i]
		if curr.Get("role").String() != "user" {
			continue
		}
		content := curr.Get("content")
		if !content.IsArray() {
			continue
		}
		for _, blk := range content.Array() {
			if blk.Get("type").String() != "tool_result" {
				continue
			}
			id := blk.Get("tool_use_id").String()
			// 紧邻前一条 assistant 必须声明这个 tool_use_id
			prev := msgsArr[i-1]
			if prev.Get("role").String() != "assistant" {
				t.Fatalf("orphan tool_result at messages[%d] (id=%s): prev role=%s not assistant",
					i, id, prev.Get("role").String())
			}
			found := false
			for _, pb := range prev.Get("content").Array() {
				if pb.Get("type").String() == "tool_use" && pb.Get("id").String() == id {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("orphan tool_result at messages[%d] (id=%s): not declared in prev assistant",
					i, id)
			}
		}
	}
}
