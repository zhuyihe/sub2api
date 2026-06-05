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
	"unicode/utf8"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	_ "golang.org/x/image/webp" // 注册 webp 解码器
)

// orphanToolResultTextPrefix 是把孤儿 tool_result 块转换为 text 块时的前缀模板。
// 出现在最终 text 字段里，便于在日志/客户端响应中识别为网关改写。
const orphanToolResultTextPrefix = "[tool_result "

// pairOrphanToolResults 修复"unexpected tool_use_id"对话完整性错误：扫描全部消息
// 的 tool_result 块，若其 tool_use_id 在「紧邻前一条 assistant 消息」的 tool_use
// 块中未声明，则把该 tool_result 块就地替换为 text 块，保留原文本内容。
//
// 注：Anthropic 上游严格要求 tool_result 对应紧邻 "previous message" 的 tool_use；
// 即使早期某条 assistant 声明过同名 tool_use，跨越后再次引用也会触发 400；
// 若孤儿 tool_result 出现在 messages[0]，根本没有可挂的 assistant。
// 统一转为 text 块的好处：不破坏 messages 长度/交替结构、永不为空、上游绝对接受。
//
// 有损：模型会把这段当成普通文字，无法识别为上一步工具调用的结果；
// 但客户端的原始文本数据被完整保留在 text 字段中。
func pairOrphanToolResults(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte("tool_result")) ||
		!bytes.Contains(body, []byte("tool_use_id")) {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok || len(messages) == 0 {
		return body, false
	}

	changed := false
	for i := 0; i < len(messages); i++ {
		declared := declaredToolUseIDsInPrevAssistant(messages, i)
		if convertOrphansToText(messages[i], declared) {
			changed = true
		}
	}
	if !changed {
		return body, false
	}
	return rewriteMessages(body, messages)
}

// convertOrphansToText 扫描 msg 的 content 数组，将所有 tool_use_id 不在 declared
// 集合中的 tool_result 块就地替换为 text 块，返回是否实际改动过。
// content 为字符串或缺省时直接返回 false（不可能有 tool_result 块）。
func convertOrphansToText(msg any, declared map[string]bool) bool {
	mm, ok := msg.(map[string]any)
	if !ok {
		return false
	}
	contentArr, ok := mm["content"].([]any)
	if !ok {
		return false
	}
	changed := false
	for idx, blk := range contentArr {
		bm, ok := blk.(map[string]any)
		if !ok || bm["type"] != "tool_result" {
			continue
		}
		id, _ := bm["tool_use_id"].(string)
		if id != "" && declared[id] {
			continue // 合法 tool_result：保留
		}
		contentArr[idx] = buildTextBlockFromToolResult(id, bm["content"])
		changed = true
	}
	if changed {
		mm["content"] = contentArr
	}
	return changed
}

// buildTextBlockFromToolResult 把一个孤儿 tool_result 块的原 content（可能是
// 字符串 / 内容块数组 / 缺省）抽出可读文本，包装为 {type:text, text:<...>}。
// 保证 text 字段非空：缺省情况兜底为 "[tool_result <id>]" 标记。
func buildTextBlockFromToolResult(id string, content any) map[string]any {
	prefix := orphanToolResultTextPrefix + id + "]"
	switch v := content.(type) {
	case string:
		if v == "" {
			return map[string]any{"type": "text", "text": prefix}
		}
		return map[string]any{"type": "text", "text": prefix + " " + v}
	case []any:
		text := extractTextFromBlocks(v)
		if text == "" {
			return map[string]any{"type": "text", "text": prefix + " (non-text content omitted)"}
		}
		return map[string]any{"type": "text", "text": prefix + " " + text}
	default:
		return map[string]any{"type": "text", "text": prefix}
	}
}

// extractTextFromBlocks 从 tool_result.content 块数组中拼接所有 type=="text" 子块的
// text 字段；非文本块（image/document 等）会被跳过（不丢错，仅丢内容）。
func extractTextFromBlocks(blocks []any) string {
	var parts []string
	for _, b := range blocks {
		bm, ok := b.(map[string]any)
		if !ok {
			continue
		}
		if bm["type"] != "text" {
			continue
		}
		if s, ok := bm["text"].(string); ok && s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, "\n")
}

// declaredToolUseIDsInPrevAssistant 收集 messages[i] 严格紧邻前一条消息(messages[i-1])
// 中的 tool_use 块 id 集合。仅当 messages[i-1] 确为 assistant 时才收集，否则返回空集。
//
// 关键：必须「严格只看 i-1」而非「向前跳过非 assistant 找最近 assistant」。Anthropic
// 按 previous message 位置校验 tool_result/tool_use 配对——若 i-1 是另一条 user(典型成因：
// 客户端连发两条 user，或截断后相邻关系被破坏)，则更早 assistant 声明的 tool_use 不算数，
// 此时 messages[i] 的 tool_result 即为孤儿。早期实现经 findPrevAssistantIdx 跳过 i-1 命中
// 早期 assistant，会误判「已声明→保留」，正是线上 user 104 的 unexpected tool_use_id 400 根因。
func declaredToolUseIDsInPrevAssistant(messages []any, i int) map[string]bool {
	declared := map[string]bool{}
	if i-1 < 0 {
		return declared
	}
	pm, ok := messages[i-1].(map[string]any)
	if !ok || pm["role"] != "assistant" {
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

// orphanToolUseTextPrefix 是把孤儿 tool_use 块转换为 text 块时的前缀模板。
const orphanToolUseTextPrefix = "[tool_use "

// pairOrphanToolUses 修复 "tool_use ids were found without tool_result blocks
// immediately after" 对话完整性错误：扫描 assistant 消息的 tool_use 块，若其 id
// 未出现在「紧邻下一条消息」的 tool_result.tool_use_id 集合中，则把该 tool_use 块
// 就地替换为 text 块，保留工具名与入参摘要。
//
// 与 pairOrphanToolResults 完全对称：后者看 prev assistant 的 tool_use 声明，
// 前者看 next message 的 tool_result 回应。Anthropic 严格要求 assistant 的每个
// tool_use 在紧邻下一条 user 消息里有对应 tool_result（典型成因：客户端在工具
// 调用被中断时，把半截的 assistant turn 连同后续请求一并发出）。
//
// 有损：模型会把这段当成普通文字，无法识别为待执行的工具调用；但工具名/入参被
// 完整保留在 text 字段中，不破坏 messages 长度/交替结构、永不为空、上游绝对接受。
func pairOrphanToolUses(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte("tool_use")) {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok || len(messages) == 0 {
		return body, false
	}

	changed := false
	for i := 0; i < len(messages); i++ {
		mm, ok := messages[i].(map[string]any)
		if !ok || mm["role"] != "assistant" {
			continue
		}
		answered := answeredToolUseIDsInNextMessage(messages, i)
		if convertOrphanToolUsesToText(mm, answered) {
			changed = true
		}
	}
	if !changed {
		return body, false
	}
	return rewriteMessages(body, messages)
}

// answeredToolUseIDsInNextMessage 收集「紧邻 messages[i] 下一条消息」中 tool_result
// 块的 tool_use_id 集合。无下一条时返回空集合。只看紧邻下一条：Anthropic 严格按
// next message 校验，跨越中间消息的后续 tool_result 不算数。
func answeredToolUseIDsInNextMessage(messages []any, i int) map[string]bool {
	answered := map[string]bool{}
	if i+1 >= len(messages) {
		return answered
	}
	nm, ok := messages[i+1].(map[string]any)
	if !ok {
		return answered
	}
	nContent, ok := nm["content"].([]any)
	if !ok {
		return answered
	}
	for _, nb := range nContent {
		nbm, ok := nb.(map[string]any)
		if !ok || nbm["type"] != "tool_result" {
			continue
		}
		if id, ok := nbm["tool_use_id"].(string); ok {
			answered[id] = true
		}
	}
	return answered
}

// convertOrphanToolUsesToText 把 mm 中 id 不在 answered 集合的 tool_use 块就地替换
// 为 text 块，返回是否实际改动过。content 非数组时直接返回 false。
func convertOrphanToolUsesToText(mm map[string]any, answered map[string]bool) bool {
	contentArr, ok := mm["content"].([]any)
	if !ok {
		return false
	}
	changed := false
	for idx, blk := range contentArr {
		bm, ok := blk.(map[string]any)
		if !ok || bm["type"] != "tool_use" {
			continue
		}
		id, _ := bm["id"].(string)
		if id != "" && answered[id] {
			continue // 已被紧邻下条回应：保留
		}
		contentArr[idx] = buildTextBlockFromToolUse(bm)
		changed = true
	}
	if changed {
		mm["content"] = contentArr
	}
	return changed
}

// buildTextBlockFromToolUse 把一个孤儿 tool_use 块抽取工具名与 input 摘要，包装为
// {type:text, text:<...>}。保证 text 字段非空：缺省兜底为 "[tool_use <name>]" 标记。
func buildTextBlockFromToolUse(bm map[string]any) map[string]any {
	name, _ := bm["name"].(string)
	prefix := orphanToolUseTextPrefix + name + "]"
	if input, ok := bm["input"]; ok {
		if raw, err := json.Marshal(input); err == nil && len(raw) > 0 && string(raw) != "null" {
			return map[string]any{"type": "text", "text": prefix + " " + string(raw)}
		}
	}
	return map[string]any{"type": "text", "text": prefix}
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

// normalizeWrappedToolSchemas flattens OpenAI/wrapper-shaped tool definitions into
// Anthropic's custom tool shape: {name, description, input_schema}.
//
// Covered client variants:
//   - {"type":"function","function":{"name":"x","parameters":{...}}}
//   - {"type":"custom","custom":{"name":"x","input_schema":{...}}}
//   - {"type":"custom","name":"x","parameters":{...}}
//
// This is a lossy schema normalization only in the sense that wrapper keys are removed;
// tool identity and input schema are preserved where present.
func normalizeWrappedToolSchemas(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte(`"tools"`)) ||
		(!bytes.Contains(body, []byte(`"function"`)) &&
			!bytes.Contains(body, []byte(`"custom"`)) &&
			!bytes.Contains(body, []byte(`"parameters"`))) {
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
	for _, tool := range tools {
		tm, ok := tool.(map[string]any)
		if !ok {
			continue
		}
		if normalizeWrappedToolSchema(tm) {
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

func normalizeWrappedToolSchema(tm map[string]any) bool {
	changed := false
	if wrapper, ok := tm["function"].(map[string]any); ok {
		_ = liftWrappedToolFields(tm, wrapper)
		delete(tm, "function")
		changed = true
	}
	if wrapper, ok := tm["custom"].(map[string]any); ok {
		_ = liftWrappedToolFields(tm, wrapper)
		delete(tm, "custom")
		changed = true
	}
	if typ, _ := tm["type"].(string); typ == "function" || typ == "custom" {
		delete(tm, "type")
		changed = true
	}
	if _, hasInputSchema := tm["input_schema"]; !hasInputSchema {
		if params, hasParams := tm["parameters"]; hasParams {
			tm["input_schema"] = params
			delete(tm, "parameters")
			changed = true
		}
	} else if _, hasParams := tm["parameters"]; hasParams {
		delete(tm, "parameters")
		changed = true
	}
	return changed
}

func liftWrappedToolFields(dst, src map[string]any) bool {
	changed := false
	for _, key := range []string{"name", "description"} {
		if _, exists := dst[key]; exists {
			continue
		}
		if value, exists := src[key]; exists {
			dst[key] = value
			changed = true
		}
	}
	if _, exists := dst["input_schema"]; !exists {
		if value, ok := src["input_schema"]; ok {
			dst["input_schema"] = value
			changed = true
		} else if value, ok := src["parameters"]; ok {
			dst["input_schema"] = value
			changed = true
		}
	}
	return changed
}

// stripCacheControlOnDeferLoadingTools 删除同时设置了 defer_loading:true 与 cache_control
// 的工具上的 cache_control 字段。Anthropic 规定 defer_loading 工具不能用 prompt caching，
// 二者并存触发 400("Tools with defer_loading cannot use prompt caching")。删 cache_control
// (仅缓存优化)、保留 defer_loading(功能性语义)，对工具行为无损。
func stripCacheControlOnDeferLoadingTools(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte("defer_loading")) ||
		!bytes.Contains(body, []byte("cache_control")) {
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
		// namespace 容器: defer_loading 在嵌套 .tools[] 的 function 上, 需下钻
		if tm["type"] == "namespace" {
			if stripDeferLoadingCacheControlInNamespace(tm) {
				changed = true
			}
			continue
		}
		// 扁平 function: 顶层直接带 defer_loading
		if tm["defer_loading"] == true {
			if _, has := tm["cache_control"]; has {
				delete(tm, "cache_control")
				changed = true
			}
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

// stripDeferLoadingCacheControlInNamespace 处理 namespace 容器: 下钻其 tools[],
// 删除带 defer_loading:true 的 function 上的 cache_control; 若该 namespace 下存在
// defer_loading function, 则容器级 cache_control 也一并删除(整个 namespace 不能用
// prompt caching)。返回是否有改动。
func stripDeferLoadingCacheControlInNamespace(ns map[string]any) bool {
	subTools, ok := ns["tools"].([]any)
	if !ok {
		return false
	}
	changed := false
	hasDeferLoading := false
	for _, st := range subTools {
		stm, ok := st.(map[string]any)
		if !ok || stm["defer_loading"] != true {
			continue
		}
		hasDeferLoading = true
		if _, has := stm["cache_control"]; has {
			delete(stm, "cache_control")
			changed = true
		}
	}
	// namespace 下有 defer_loading function -> 整体不能 caching, 删容器级 cache_control
	if hasDeferLoading {
		if _, has := ns["cache_control"]; has {
			delete(ns, "cache_control")
			changed = true
		}
	}
	return changed
}

// stripMessageLevelCacheControl removes cache_control directly attached to a
// messages[i] object. Anthropic accepts cache_control on supported content blocks,
// but rejects message-level cache_control with "Extra inputs are not permitted".
func stripMessageLevelCacheControl(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte("cache_control")) {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok {
		return body, false
	}
	changed := false
	for _, msg := range messages {
		mm, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		if _, has := mm["cache_control"]; has {
			delete(mm, "cache_control")
			changed = true
		}
	}
	if !changed {
		return body, false
	}
	return rewriteMessages(body, messages)
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

const maxAnthropicToolNameLength = 128

// invalidToolNameChars 匹配 Anthropic custom tool name 中不被接受的字符。
// 合法格式为 ^[a-zA-Z0-9_-]{1,128}$。
var invalidToolNameChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

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

// sanitizeAnthropicToolName 将工具名确定性清洗为 Anthropic 接受的 custom tool name。
func sanitizeAnthropicToolName(name string) string {
	cleaned := strings.TrimSpace(name)
	cleaned = invalidToolNameChars.ReplaceAllString(cleaned, "_")
	if cleaned == "" {
		cleaned = "tool"
	}
	if len(cleaned) > maxAnthropicToolNameLength {
		cleaned = cleaned[:maxAnthropicToolNameLength]
	}
	return cleaned
}

// sanitizeAnthropicToolNames 清洗 tools[].name、tool_choice.name 和历史 tool_use.name。
// 同一个原始工具名使用同一映射，避免 tool_choice/tool_use 与 tools 声明脱节。
func sanitizeAnthropicToolNames(body []byte) ([]byte, bool) {
	if !bytes.Contains(body, []byte(`"tools"`)) &&
		!bytes.Contains(body, []byte(`"tool_choice"`)) &&
		!bytes.Contains(body, []byte(`"tool_use"`)) {
		return body, false
	}

	out := body
	nameMap := map[string]string{}
	changed := false

	if toolsRes := gjson.GetBytes(out, "tools"); toolsRes.Exists() && toolsRes.IsArray() {
		var tools []any
		if err := json.Unmarshal(sliceRawFromBody(out, toolsRes), &tools); err != nil {
			return body, false
		}
		toolsChanged := false
		for _, tool := range tools {
			tm, ok := tool.(map[string]any)
			if !ok {
				continue
			}
			if sanitizeToolNameInMap(tm, nameMap) {
				toolsChanged = true
			}
			for _, wrapperKey := range []string{"custom", "function"} {
				wrapper, ok := tm[wrapperKey].(map[string]any)
				if ok && sanitizeToolNameInMap(wrapper, nameMap) {
					toolsChanged = true
				}
			}
		}
		if toolsChanged {
			tb, err := json.Marshal(tools)
			if err != nil {
				return body, false
			}
			next, err := sjson.SetRawBytes(out, "tools", tb)
			if err != nil {
				return body, false
			}
			out = next
			changed = true
		}
	}

	if tcName := gjson.GetBytes(out, "tool_choice.name"); tcName.Exists() && tcName.Type == gjson.String {
		if sanitized := mappedAnthropicToolName(tcName.String(), nameMap); sanitized != tcName.String() {
			if next, err := sjson.SetBytes(out, "tool_choice.name", sanitized); err == nil {
				out = next
				changed = true
			}
		}
	}

	messages, ok := unmarshalMessages(out)
	if ok {
		messagesChanged := false
		forEachContentBlock(messages, func(bm map[string]any) {
			if bm["type"] != "tool_use" {
				return
			}
			name, ok := bm["name"].(string)
			if !ok {
				return
			}
			if sanitized := mappedAnthropicToolName(name, nameMap); sanitized != name {
				bm["name"] = sanitized
				messagesChanged = true
			}
		})
		if messagesChanged {
			next, ok := rewriteMessages(out, messages)
			if !ok {
				return body, false
			}
			out = next
			changed = true
		}
	}

	if !changed {
		return body, false
	}
	return out, true
}

func sanitizeToolNameInMap(obj map[string]any, nameMap map[string]string) bool {
	name, ok := obj["name"].(string)
	if !ok {
		return false
	}
	sanitized := sanitizeAnthropicToolName(name)
	nameMap[name] = sanitized
	if sanitized == name {
		return false
	}
	obj["name"] = sanitized
	return true
}

func mappedAnthropicToolName(name string, nameMap map[string]string) string {
	if mapped, ok := nameMap[name]; ok {
		return mapped
	}
	sanitized := sanitizeAnthropicToolName(name)
	nameMap[name] = sanitized
	return sanitized
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

const truncatedPromptTextPrefix = "[gateway truncated oversized prompt]\n"

// truncateOversizedPrompt 当请求体过大时，从最旧消息开始丢弃、保留最近消息。
// 如果单条消息本身就超限，则继续裁剪最旧文本内容的前缀、保留尾部近期内容。
// 保留顶层 system/tools 不动，并确保截断后首条消息为 user。
// 有损：丢弃历史/文本前缀，模型可能缺失上下文。
func truncateOversizedPrompt(body []byte) ([]byte, bool) {
	if len(body) <= maxPromptBodyBytes {
		return body, false
	}
	messages, ok := unmarshalMessages(body)
	if !ok {
		return body, false
	}

	budget := maxPromptBodyBytes * 8 / 10
	out := body
	changed := false

	if len(messages) > 2 {
		used := len(out) - len(gjson.GetBytes(out, "messages").Raw) // system/tools 等固定开销
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
		if len(kept) > 0 && len(kept) < len(messages) {
			if next, ok := rewriteMessages(out, kept); ok {
				out = next
				messages = kept
				changed = true
			}
		}
	}

	for pass := 0; pass < 4 && len(out) > budget; pass++ {
		currentMessages, ok := unmarshalMessages(out)
		if !ok {
			break
		}
		excess := len(out) - budget + 4096
		if !trimOldestMessageText(currentMessages, excess) {
			break
		}
		next, ok := rewriteMessages(out, currentMessages)
		if !ok {
			break
		}
		out = next
		changed = true
	}

	if !changed {
		return body, false
	}
	return out, true
}

func trimOldestMessageText(messages []any, targetReduction int) bool {
	if targetReduction <= 0 {
		return false
	}
	remaining := targetReduction
	changed := false
	for _, msg := range messages {
		mm, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		switch content := mm["content"].(type) {
		case string:
			next, reduced, ok := trimTextPrefixForBudget(content, remaining)
			if !ok {
				continue
			}
			mm["content"] = next
			remaining -= reduced
			changed = true
		case []any:
			for _, block := range content {
				bm, ok := block.(map[string]any)
				if !ok || bm["type"] != "text" {
					continue
				}
				text, ok := bm["text"].(string)
				if !ok {
					continue
				}
				next, reduced, ok := trimTextPrefixForBudget(text, remaining)
				if !ok {
					continue
				}
				bm["text"] = next
				remaining -= reduced
				changed = true
				if remaining <= 0 {
					return changed
				}
			}
		}
		if remaining <= 0 {
			return changed
		}
	}
	return changed
}

func trimTextPrefixForBudget(text string, targetReduction int) (string, int, bool) {
	if text == "" || targetReduction <= 0 {
		return text, 0, false
	}
	if len(text) <= len(truncatedPromptTextPrefix) {
		return text, 0, false
	}
	needToRemove := targetReduction + len(truncatedPromptTextPrefix)
	if needToRemove >= len(text) {
		return truncatedPromptTextPrefix, len(text) - len(truncatedPromptTextPrefix), true
	}
	keepBytes := len(text) - needToRemove
	if keepBytes <= 0 {
		return truncatedPromptTextPrefix, len(text) - len(truncatedPromptTextPrefix), true
	}
	start := len(text) - keepBytes
	for start < len(text) && !utf8.RuneStart(text[start]) {
		start++
	}
	if start >= len(text) {
		return truncatedPromptTextPrefix, len(text) - len(truncatedPromptTextPrefix), true
	}
	next := truncatedPromptTextPrefix + text[start:]
	reduced := len(text) - len(next)
	if reduced <= 0 {
		return text, 0, false
	}
	return next, reduced, true
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
