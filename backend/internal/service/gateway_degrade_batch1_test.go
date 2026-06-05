package service

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestNormalizeToolFunctionType(t *testing.T) {
	// OpenAI 风格 tools[].type="function" 应被删除，name 保留
	body := `{"tools":[{"type":"function","name":"x","input_schema":{"type":"object"}},{"type":"web_search_20260209","name":"y"}],"messages":[]}`
	out, ok := normalizeToolFunctionType([]byte(body))
	if !ok {
		t.Fatal("expected change")
	}
	if gjson.GetBytes(out, "tools.0.type").Exists() {
		t.Fatal("tools.0.type=function should be removed")
	}
	if gjson.GetBytes(out, "tools.0.name").String() != "x" {
		t.Fatal("tools.0.name should be kept")
	}
	// 白名单 type 保留
	if gjson.GetBytes(out, "tools.1.type").String() != "web_search_20260209" {
		t.Fatal("tools.1.type=web_search should be kept")
	}
	// 无 function type 时不动
	if _, ok := normalizeToolFunctionType([]byte(`{"tools":[{"name":"x"}],"messages":[]}`)); ok {
		t.Fatal("no function type should not change")
	}
}

func TestNormalizeWrappedToolSchemas(t *testing.T) {
	t.Run("OpenAI function wrapper flattened", func(t *testing.T) {
		body := `{"tools":[{"type":"function","function":{"name":"lookup","description":"Lookup data","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}],"messages":[]}`
		out, ok := normalizeWrappedToolSchemas([]byte(body))
		if !ok {
			t.Fatal("expected change")
		}
		if gjson.GetBytes(out, "tools.0.type").Exists() {
			t.Fatal("function type should be removed")
		}
		if gjson.GetBytes(out, "tools.0.function").Exists() {
			t.Fatal("function wrapper should be removed")
		}
		if got := gjson.GetBytes(out, "tools.0.name").String(); got != "lookup" {
			t.Fatalf("name=%q, want lookup", got)
		}
		if got := gjson.GetBytes(out, "tools.0.input_schema.properties.q.type").String(); got != "string" {
			t.Fatalf("input_schema not lifted, got %q", got)
		}
	})

	t.Run("custom wrapper missing nested name fixed by flattening", func(t *testing.T) {
		body := `{"tools":[{"type":"custom","name":"edit","custom":{"input_schema":{"type":"object"}}}],"messages":[]}`
		out, ok := normalizeWrappedToolSchemas([]byte(body))
		if !ok {
			t.Fatal("expected change")
		}
		if gjson.GetBytes(out, "tools.0.type").Exists() {
			t.Fatal("custom type should be removed")
		}
		if gjson.GetBytes(out, "tools.0.custom").Exists() {
			t.Fatal("custom wrapper should be removed")
		}
		if got := gjson.GetBytes(out, "tools.0.name").String(); got != "edit" {
			t.Fatalf("name=%q, want edit", got)
		}
		if got := gjson.GetBytes(out, "tools.0.input_schema.type").String(); got != "object" {
			t.Fatalf("input_schema.type=%q, want object", got)
		}
	})

	t.Run("flat custom type removed and parameters renamed", func(t *testing.T) {
		body := `{"tools":[{"type":"custom","name":"run","parameters":{"type":"object"}}],"messages":[]}`
		out, ok := normalizeWrappedToolSchemas([]byte(body))
		if !ok {
			t.Fatal("expected change")
		}
		if gjson.GetBytes(out, "tools.0.type").Exists() {
			t.Fatal("custom type should be removed")
		}
		if gjson.GetBytes(out, "tools.0.parameters").Exists() {
			t.Fatal("parameters should be removed after being renamed")
		}
		if got := gjson.GetBytes(out, "tools.0.input_schema.type").String(); got != "object" {
			t.Fatalf("input_schema.type=%q, want object", got)
		}
	})
}

func TestNormalizeToolChoice(t *testing.T) {
	out, ok := normalizeToolChoice([]byte(`{"tool_choice":"auto","messages":[]}`))
	if !ok {
		t.Fatal("expected change")
	}
	if gjson.GetBytes(out, "tool_choice.type").String() != "auto" {
		t.Fatalf("tool_choice.type=%s, want auto", gjson.GetBytes(out, "tool_choice.type").String())
	}
	// 已是对象时不动
	if _, ok := normalizeToolChoice([]byte(`{"tool_choice":{"type":"auto"}}`)); ok {
		t.Fatal("object tool_choice should not change")
	}
}

func TestStripMessageLevelCacheControl(t *testing.T) {
	body := `{"messages":[
		{"role":"user","content":"hi","cache_control":{"type":"ephemeral"}},
		{"role":"assistant","content":[{"type":"text","text":"ok","cache_control":{"type":"ephemeral"}}]}
	]}`
	out, ok := stripMessageLevelCacheControl([]byte(body))
	if !ok {
		t.Fatal("expected message-level cache_control stripped")
	}
	if gjson.GetBytes(out, "messages.0.cache_control").Exists() {
		t.Fatal("message-level cache_control should be removed")
	}
	if !gjson.GetBytes(out, "messages.1.content.0.cache_control").Exists() {
		t.Fatal("content block cache_control should be kept")
	}
}

func TestNormalizeImageMediaType(t *testing.T) {
	// 声明 jpeg 实为 png(iVBORw0KGgo 前缀)
	body := `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"iVBORw0KGgoAAAA"}}]}]}`
	out, ok := normalizeImageMediaType([]byte(body))
	if !ok {
		t.Fatal("expected media_type corrected")
	}
	got := gjson.GetBytes(out, "messages.0.content.0.source.media_type").String()
	if got != "image/png" {
		t.Fatalf("media_type=%s, want image/png", got)
	}
	// 一致时不动
	ok2body := `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo"}}]}]}`
	if _, ok := normalizeImageMediaType([]byte(ok2body)); ok {
		t.Fatal("matching media_type should not change")
	}
}

func TestSanitizeToolUseIDs(t *testing.T) {
	// tool_use.id 与 tool_result.tool_use_id 含非法字符，应被一致清洗
	body := `{"messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"call:abc#1","name":"x","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"call:abc#1","content":"ok"}]}
	]}`
	out, ok := sanitizeToolUseIDs([]byte(body))
	if !ok {
		t.Fatal("expected sanitize")
	}
	id := gjson.GetBytes(out, "messages.0.content.0.id").String()
	ref := gjson.GetBytes(out, "messages.1.content.0.tool_use_id").String()
	if id != "call_abc_1" {
		t.Fatalf("id=%s, want call_abc_1", id)
	}
	if id != ref {
		t.Fatalf("id(%s) != tool_use_id(%s), 引用不一致", id, ref)
	}
	// 合法 id 不动
	if _, ok := sanitizeToolUseIDs([]byte(`{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_01ABC","name":"x","input":{}}]}]}`)); ok {
		t.Fatal("valid id should not change")
	}
}

func TestLimitCacheControlBlocks(t *testing.T) {
	// 5 个 cache_control，上限 4，应删 1
	body := `{"system":[
		{"type":"text","text":"a","cache_control":{"type":"ephemeral"}},
		{"type":"text","text":"b","cache_control":{"type":"ephemeral"}}
	],"messages":[
		{"role":"user","content":[
			{"type":"text","text":"c","cache_control":{"type":"ephemeral"}},
			{"type":"text","text":"d","cache_control":{"type":"ephemeral"}},
			{"type":"text","text":"e","cache_control":{"type":"ephemeral"}}
		]}
	]}`
	out, ok := limitCacheControlBlocks([]byte(body), 4)
	if !ok {
		t.Fatal("expected trim")
	}
	if !gjson.ValidBytes(out) {
		t.Fatalf("invalid json: %s", out)
	}
	if n := countCacheControl(out); n != 4 {
		t.Fatalf("remaining cache_control=%d, want 4", n)
	}
	// 恰好 4 个不动
	body4 := `{"messages":[{"role":"user","content":[
		{"type":"text","text":"a","cache_control":{"type":"ephemeral"}},
		{"type":"text","text":"b","cache_control":{"type":"ephemeral"}},
		{"type":"text","text":"c","cache_control":{"type":"ephemeral"}},
		{"type":"text","text":"d","cache_control":{"type":"ephemeral"}}
	]}]}`
	if _, ok := limitCacheControlBlocks([]byte(body4), 4); ok {
		t.Fatal("exactly-max should not change")
	}
}

func countCacheControl(body []byte) int {
	return strings.Count(string(body), `"cache_control"`)
}
