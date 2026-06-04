package service

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	_ "image/gif"  // 注册 gif 解码器(供 DecodeConfig)
	_ "image/jpeg" // 注册 jpeg 解码器
	_ "image/png"  // 注册 png 解码器
	"regexp"
	"sort"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	_ "golang.org/x/image/webp" // 注册 webp 解码器
)

// placeholderOrphanToolUseName 标记由网关为孤儿 tool_result 补齐的占位工具调用。
const placeholderOrphanToolUseName = "_gateway_orphan_tool_use_placeholder"

// pairOrphanToolResults 修复"unexpected tool_use_id"对话完整性错误：扫描每条消息
// 的 tool_result 块，若其 tool_use_id 在「紧邻前一条 assistant 消息」的 tool_use
// 块中未声明，则在该 assistant 消息 content 末尾追加 id 匹配的占位 tool_use。
// 注：Anthropic 上游严格要求 tool_result 对应紧邻 "previous message" 的 tool_use；
// 即使早期某条 assistant 声明过同名 tool_use，跨越后再次引用也会触发 400。
// 有损：注入占位 tool_use(name 为标记常量)，改变了对话历史的完整性，
// 但保留客户端的 tool_result 数据(不丢失工具调用结果)。
func pairOrphanToolResults(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte("tool_result")) ||
		!bytes.Contains(body, []byte("tool_use_id")) {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok || len(messages) < 2 {
		return body, false
	}

	changed := false
	for i := 1; i < len(messages); i++ {
		orphans := orphanToolUseIDsInMessage(messages, i)
		if len(orphans) == 0 {
			continue
		}
		prevIdx := findPrevAssistantIdx(messages, i)
		if prevIdx < 0 {
			continue
		}
		if appendPlaceholderToolUses(messages[prevIdx], orphans) {
			changed = true
		}
	}
	if !changed {
		return body, false
	}
	return rewriteMessages(body, messages)
}

// orphanToolUseIDsInMessage 返回 messages[i] 中所有 tool_result 块引用的、
// 在「紧邻前一条 assistant 消息」的 tool_use 块中未声明的 tool_use_id 列表。
func orphanToolUseIDsInMessage(messages []any, i int) []string {
	curr, ok := messages[i].(map[string]any)
	if !ok {
		return nil
	}
	contentArr, ok := curr["content"].([]any)
	if !ok {
		return nil
	}
	declared := declaredToolUseIDsInPrevAssistant(messages, i)
	var orphans []string
	for _, blk := range contentArr {
		bm, ok := blk.(map[string]any)
		if !ok || bm["type"] != "tool_result" {
			continue
		}
		id, ok := bm["tool_use_id"].(string)
		if !ok || id == "" || declared[id] {
			continue
		}
		orphans = append(orphans, id)
	}
	return orphans
}

// declaredToolUseIDsInPrevAssistant 收集紧邻 messages[i] 前一条 assistant 消息
// 的 tool_use 块 id 集合。无紧邻 assistant 时返回空集合。
// 这里刻意「只看紧邻」而非「所有前置」：Anthropic 严格按 previous message 校验，
// 跨越中间消息的早期 tool_use 不算数（否则会触发上游 400）。
func declaredToolUseIDsInPrevAssistant(messages []any, i int) map[string]bool {
	declared := map[string]bool{}
	prevIdx := findPrevAssistantIdx(messages, i)
	if prevIdx < 0 {
		return declared
	}
	pm, ok := messages[prevIdx].(map[string]any)
	if !ok {
		return declared
	}
	pContent, ok := pm["content"].([]any)
	if !ok {
		return declared
	}
	for _, pb := range pContent {
		pbm, ok := pb.(map[string]any)
		if !ok || pbm["type"] != "tool_use" {
			continue
		}
		if id, ok := pbm["id"].(string); ok {
			declared[id] = true
		}
	}
	return declared
}

// findPrevAssistantIdx 从 i-1 向前查找最近的 assistant 消息下标，找不到返回 -1。
func findPrevAssistantIdx(messages []any, i int) int {
	for j := i - 1; j >= 0; j-- {
		pm, ok := messages[j].(map[string]any)
		if ok && pm["role"] == "assistant" {
			return j
		}
	}
	return -1
}

// appendPlaceholderToolUses 向给定 assistant 消息的 content 末尾追加占位 tool_use
// 块（对每个 id 各一个）。若 content 当前是字符串，先转成包含原文本的内容块数组。
// 返回是否实际修改了消息。
func appendPlaceholderToolUses(msg any, ids []string) bool {
	prev, ok := msg.(map[string]any)
	if !ok {
		return false
	}
	var prevContent []any
	switch c := prev["content"].(type) {
	case []any:
		prevContent = c
	case string:
		prevContent = []any{map[string]any{"type": "text", "text": c}}
	default:
		return false
	}
	for _, id := range ids {
		prevContent = append(prevContent, map[string]any{
			"type":  "tool_use",
			"id":    id,
			"name":  placeholderOrphanToolUseName,
			"input": map[string]any{},
		})
	}
	prev["content"] = prevContent
	return true
}

// normalizeToolFunctionType 删除 tools[i].type == "function" 字段（OpenAI schema 误用）。
// Anthropic 仅接受预定义工具 type 白名单(bash/code_execution/text_editor/web_fetch 等)
// 或省略 type 让其默认为 custom。删除该字段即让客户端定义的工具按 custom 处理。
func normalizeToolFunctionType(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte(`"type":"function"`)) &&
		!bytes.Contains(body, []byte(`"type": "function"`)) {
		return body, false
	}
	toolsRes := gjson.GetBytes(body, "tools")
	if !toolsRes.Exists() || !toolsRes.IsArray() {
		return body, false
	}
	var tools []any
	if err := json.Unmarshal(sliceRawFromBody(body, toolsRes), &tools); err != nil {
		return body, false
	}
	changed := false
	for _, t := range tools {
		tm, ok := t.(map[string]any)
		if !ok {
			continue
		}
		if tm["type"] == "function" {
			delete(tm, "type")
			changed = true
		}
	}
	if !changed {
		return body, false
	}
	tb, err := json.Marshal(tools)
	if err != nil {
		return body, false
	}
	out, err := sjson.SetRawBytes(body, "tools", tb)
	if err != nil {
		return body, false
	}
	return out, true
}

// normalizeToolChoice 把字符串形式的 tool_choice 包装为对象 {"type": <value>}。
// 上游要求 tool_choice 为对象，客户端误传字符串(如 "auto")会触发 400。
func normalizeToolChoice(body []byte) ([]byte, bool) {
	tc := gjson.GetBytes(body, "tool_choice")
	if !tc.Exists() || tc.Type != gjson.String {
		return body, false
	}
	out, err := sjson.SetBytes(body, "tool_choice", map[string]string{"type": tc.String()})
	if err != nil {
		return body, false
	}
	return out, true
}

// imageMagicPrefixes 是 base64 编码后图片数据的起始前缀 → 真实 media_type 的映射。
var imageMagicPrefixes = []struct {
	prefix    string
	mediaType string
}{
	{"/9j/", "image/jpeg"},
	{"iVBORw0KGgo", "image/png"},
	{"R0lGOD", "image/gif"},
	{"UklGR", "image/webp"},
}

// detectImageMediaType 依据 base64 数据前缀判断真实图片格式，未知返回空串。
func detectImageMediaType(b64 string) string {
	for _, m := range imageMagicPrefixes {
		if strings.HasPrefix(b64, m.prefix) {
			return m.mediaType
		}
	}
	return ""
}

// normalizeImageMediaType 修正 base64 图片块中与真实格式不符的 media_type，
// 避免上游"声明 jpeg 实为 png"之类的 400。
func normalizeImageMediaType(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte(`"type":"image"`)) &&
		!bytes.Contains(body, []byte(`"type": "image"`)) {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok {
		return body, false
	}

	changed := false
	forEachContentBlock(messages, func(bm map[string]any) {
		if bm["type"] != "image" {
			return
		}
		src, ok := bm["source"].(map[string]any)
		if !ok || src["type"] != "base64" {
			return
		}
		data, ok := src["data"].(string)
		if !ok {
			return
		}
		real := detectImageMediaType(data)
		if real == "" {
			return
		}
		if mt, _ := src["media_type"].(string); mt != real {
			src["media_type"] = real
			changed = true
		}
	})
	if !changed {
		return body, false
	}
	return rewriteMessages(body, messages)
}

// invalidToolIDChars 匹配 tool id 中不被上游接受的字符（合法集 [a-zA-Z0-9_-]）。
var invalidToolIDChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// sanitizeToolUseIDs 清洗 tool_use.id 与 tool_result.tool_use_id 中的非法字符。
// 清洗是确定性的(同一原始 id 映射到同一新 id)，因此引用两端自动保持一致。
func sanitizeToolUseIDs(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte(`"tool_use`)) { // 同时覆盖 tool_use 与 tool_use_id
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok {
		return body, false
	}

	clean := func(id string) string {
		if id == "" || !invalidToolIDChars.MatchString(id) {
			return id
		}
		return invalidToolIDChars.ReplaceAllString(id, "_")
	}
	changed := false
	forEachContentBlock(messages, func(bm map[string]any) {
		switch bm["type"] {
		case "tool_use":
			if id, ok := bm["id"].(string); ok {
				if nid := clean(id); nid != id {
					bm["id"] = nid
					changed = true
				}
			}
		case "tool_result":
			if id, ok := bm["tool_use_id"].(string); ok {
				if nid := clean(id); nid != id {
					bm["tool_use_id"] = nid
					changed = true
				}
			}
		}
	})
	if !changed {
		return body, false
	}
	return rewriteMessages(body, messages)
}

// limitCacheControlBlocks 当 cache_control 断点超过上限时，删除多余的(保留靠前的)，
// 避免上游"超过 N 个 cache_control"的 400。删除仅降低缓存效率，不改请求语义。
func limitCacheControlBlocks(body []byte, max int) ([]byte, bool) {
	if bytes.Count(body, []byte(`"cache_control"`)) <= max {
		return body, false
	}
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		return body, false
	}

	count := 0
	var walk func(v any)
	walk = func(v any) {
		switch t := v.(type) {
		case map[string]any:
			if _, ok := t["cache_control"]; ok {
				count++
				if count > max {
					delete(t, "cache_control")
				}
			}
			for _, vv := range t {
				walk(vv)
			}
		case []any:
			for _, vv := range t {
				walk(vv)
			}
		}
	}
	walk(root)

	out, err := json.Marshal(root)
	if err != nil {
		return body, false
	}
	return out, true
}

// maxPromptBodyBytes 是触发 prompt 截断的 body 字节阈值（best-effort）。
// 粗略对应 Claude ~190K token；这是字符近似而非精确 token 计数，
// 可能误伤接近上限的正常长对话，也可能漏掉略超的请求。
const maxPromptBodyBytes = 650 * 1024

// truncateOversizedPrompt 当请求体过大时，从最旧消息开始丢弃、保留最近消息，
// 使其落入预算内，避免 prompt too long 的 400。保留顶层 system/tools 不动，
// 并确保截断后首条消息为 user。有损：丢弃历史，模型可能缺失上下文。
func truncateOversizedPrompt(body []byte) ([]byte, bool) {
	if len(body) <= maxPromptBodyBytes {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok || len(messages) <= 2 {
		return body, false
	}

	budget := maxPromptBodyBytes * 8 / 10
	used := len(body) - len(gjson.GetBytes(body, "messages").Raw) // system/tools 等固定开销
	kept := make([]any, 0, len(messages))
	for i := len(messages) - 1; i >= 0; i-- {
		raw, err := json.Marshal(messages[i])
		if err != nil {
			return body, false
		}
		if used+len(raw) > budget && len(kept) > 0 {
			break
		}
		used += len(raw)
		kept = append([]any{messages[i]}, kept...)
	}
	// 确保首条为 user（Anthropic 要求）
	for len(kept) > 0 {
		if first, ok := kept[0].(map[string]any); ok && first["role"] == "user" {
			break
		}
		kept = kept[1:]
	}
	if len(kept) == 0 || len(kept) == len(messages) {
		return body, false
	}
	return rewriteMessages(body, kept)
}

// looksLikeInvalidJSONError 判断上游 400 错误体是否为 JSON/转义格式问题，
// 用于决定是否做"事后规范化重试"。
func looksLikeInvalidJSONError(respBody []byte) bool {
	m := strings.ToLower(string(respBody))
	return strings.Contains(m, "not valid json") ||
		strings.Contains(m, "invalid escaped character") ||
		strings.Contains(m, "invalid \\escape")
}

// renormalizeJSONBody 对整个 body 做 Unmarshal+Marshal，用 Go 标准编码重写字符串
// 转义，以修复上游挑剔的非法转义。仅当结果与原 body 不同才返回 changed=true。
// 注:能解析到此说明 body 对 Go 合法；纯语法错的 body 在更早阶段已失败、到不了这里。
func renormalizeJSONBody(body []byte) ([]byte, bool) {
	var v any
	if err := json.Unmarshal(body, &v); err != nil {
		return body, false
	}
	out, err := json.Marshal(v)
	if err != nil || bytes.Equal(out, body) {
		return body, false
	}
	return out, true
}

// unmarshalMessages 解析顶层 messages 数组为 []any；非数组或失败返回 ok=false。
func unmarshalMessages(body []byte) ([]any, bool) {
	msgsRes := gjson.GetBytes(body, "messages")
	if !msgsRes.Exists() || !msgsRes.IsArray() {
		return nil, false
	}
	var messages []any
	if err := json.Unmarshal(sliceRawFromBody(body, msgsRes), &messages); err != nil {
		return nil, false
	}
	return messages, true
}

// forEachContentBlock 遍历每条消息 content 数组中的对象块，对其调用 fn。
func forEachContentBlock(messages []any, fn func(block map[string]any)) {
	for _, m := range messages {
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		content, ok := mm["content"].([]any)
		if !ok {
			continue
		}
		for _, blk := range content {
			if bm, ok := blk.(map[string]any); ok {
				fn(bm)
			}
		}
	}
}

// rewriteMessages 将修改后的 messages 写回 body 的 messages 字段。
func rewriteMessages(body []byte, messages []any) ([]byte, bool) {
	mb, err := json.Marshal(messages)
	if err != nil {
		return body, false
	}
	out, err := sjson.SetRawBytes(body, "messages", mb)
	if err != nil {
		return body, false
	}
	return out, true
}

// --- 批2: 有损降级（会改变请求语义/质量，发生时务必记录日志）---

// placeholderToolDescription 标记由网关自动补全的占位工具，便于排查。
const placeholderToolDescription = "Auto-generated placeholder by gateway to satisfy tool reference."

// backfillMissingTools 为 messages 中被 tool_use 引用、但未在顶层 tools 声明的工具，
// 补一个最小占位定义，避免上游"tool reference not found"的 400。
// 有损：占位 schema 不精确，仅用于通过历史 tool_use 的校验。
func backfillMissingTools(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte(`"tool_use"`)) {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok {
		return body, false
	}

	used := map[string]bool{}
	forEachContentBlock(messages, func(bm map[string]any) {
		if bm["type"] == "tool_use" {
			if name, ok := bm["name"].(string); ok && name != "" {
				used[name] = true
			}
		}
	})
	if len(used) == 0 {
		return body, false
	}

	var tools []any
	declared := map[string]bool{}
	if tRes := gjson.GetBytes(body, "tools"); tRes.Exists() && tRes.IsArray() {
		if err := json.Unmarshal(sliceRawFromBody(body, tRes), &tools); err != nil {
			return body, false
		}
		for _, t := range tools {
			if tm, ok := t.(map[string]any); ok {
				if name, ok := tm["name"].(string); ok {
					declared[name] = true
				}
			}
		}
	}

	var missing []string
	for name := range used {
		if !declared[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) == 0 {
		return body, false
	}
	sort.Strings(missing) // 确定性顺序

	for _, name := range missing {
		tools = append(tools, map[string]any{
			"name":        name,
			"description": placeholderToolDescription,
			"input_schema": map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		})
	}
	tb, err := json.Marshal(tools)
	if err != nil {
		return body, false
	}
	out, err := sjson.SetRawBytes(body, "tools", tb)
	if err != nil {
		return body, false
	}
	return out, true
}

// prefillContinuationText 是为 assistant 结尾的请求注入的最小 user 内容。
const prefillContinuationText = "Continue."

// appendUserForAssistantPrefill 当 messages 以 assistant(prefill) 结尾时追加一条
// user 消息，满足"末条必须为 user"的模型约束并保留前文。
// 有损：注入了网关生成的 user 内容，改变了原 prefill 续写意图。
func appendUserForAssistantPrefill(body []byte) ([]byte, bool) {
	// Fast-path: 用 gjson 轻量判断末条 role，非 assistant 直接返回，避免全量 unmarshal。
	arr := gjson.GetBytes(body, "messages").Array()
	if len(arr) == 0 || arr[len(arr)-1].Get("role").String() != "assistant" {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok || len(messages) == 0 {
		return body, false
	}
	messages = append(messages, map[string]any{
		"role":    "user",
		"content": prefillContinuationText,
	})
	return rewriteMessages(body, messages)
}

// 图片限制（Anthropic）：单图 ~5MB、单边 ~8000px。
const (
	maxImageBytes     = 5 * 1024 * 1024
	maxImageDimension = 8000
)

// removeOversizedImages 删除超出尺寸/大小限制的 base64 图片块，让请求得以通过。
// 用 base64 长度判断大小、DecodeConfig 只读 header 判断尺寸，均不解码全图，无 OOM。
// 有损：模型将看不到被删图片，若用户正询问该图会答非所问。
func removeOversizedImages(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte(`"type":"image"`)) &&
		!bytes.Contains(body, []byte(`"type": "image"`)) {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok {
		return body, false
	}

	changed := false
	for _, m := range messages {
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		content, ok := mm["content"].([]any)
		if !ok {
			continue
		}
		kept := make([]any, 0, len(content))
		for _, blk := range content {
			if isOversizedImageBlock(blk) {
				changed = true
				continue
			}
			kept = append(kept, blk)
		}
		if len(kept) != len(content) {
			mm["content"] = kept
		}
	}
	if !changed {
		return body, false
	}
	return rewriteMessages(body, messages)
}

// isOversizedImageBlock 判断一个内容块是否为超限的 base64 图片。
func isOversizedImageBlock(blk any) bool {
	bm, ok := blk.(map[string]any)
	if !ok || bm["type"] != "image" {
		return false
	}
	src, ok := bm["source"].(map[string]any)
	if !ok || src["type"] != "base64" {
		return false
	}
	data, ok := src["data"].(string)
	if !ok || data == "" {
		return false
	}
	// 大小：base64 长度 * 3/4 ≈ 原始字节，无需解码。
	if len(data)/4*3 > maxImageBytes {
		return true
	}
	// 尺寸：解码 base64 到原始字节(几 MB 可控)，DecodeConfig 仅读 header。
	raw, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return false
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		return false
	}
	return cfg.Width > maxImageDimension || cfg.Height > maxImageDimension
}

// normalizeCacheControlTTL 删除所有 cache_control 块中的 ttl 字段，使其退回默认 TTL。
// 以此消除"1h 块不能排在 5m 块之后"的顺序约束 400。有损：失去自定义(如 1h)长缓存。
func normalizeCacheControlTTL(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte(`"ttl"`)) || !bytes.Contains(body, []byte(`"cache_control"`)) {
		return body, false
	}
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		return body, false
	}

	changed := false
	var walk func(v any)
	walk = func(v any) {
		switch t := v.(type) {
		case map[string]any:
			if cc, ok := t["cache_control"].(map[string]any); ok {
				if _, has := cc["ttl"]; has {
					delete(cc, "ttl")
					changed = true
				}
			}
			for _, vv := range t {
				walk(vv)
			}
		case []any:
			for _, vv := range t {
				walk(vv)
			}
		}
	}
	walk(root)
	if !changed {
		return body, false
	}
	out, err := json.Marshal(root)
	if err != nil {
		return body, false
	}
	return out, true
}
