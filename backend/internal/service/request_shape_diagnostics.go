package service

import (
	"context"
	"fmt"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
)

const requestShapeFindingLimit = 40

type requestShapeDiagnostics struct {
	Findings       []string
	MessagesCount  int
	ToolsCount     int
	FunctionsCount int
	InputCount     int
}

func (d requestShapeDiagnostics) hasFindings() bool {
	return len(d.Findings) > 0
}

func (d requestShapeDiagnostics) zapFields() []zap.Field {
	return []zap.Field{
		zap.Strings("shape_findings", d.Findings),
		zap.Int("shape_messages_count", d.MessagesCount),
		zap.Int("shape_tools_count", d.ToolsCount),
		zap.Int("shape_functions_count", d.FunctionsCount),
		zap.Int("shape_input_count", d.InputCount),
	}
}

func logOpenAIChatCompletionsShapeDiagnostics(ctx context.Context, body, responsesBody []byte, isResponsesShape bool) {
	diag := diagnoseOpenAIChatCompletionsShape(body, responsesBody)
	if !diag.hasFindings() {
		return
	}
	fields := diag.zapFields()
	fields = append(fields, zap.Bool("responses_shape", isResponsesShape))
	logger.FromContext(ctx).Warn("openai_chat_completions.request_shape_diagnostics", fields...)
}

func diagnoseOpenAIChatCompletionsShape(body, responsesBody []byte) requestShapeDiagnostics {
	diag := requestShapeDiagnostics{}
	diag.MessagesCount = arrayLen(gjson.GetBytes(body, "messages"))
	diag.ToolsCount = arrayLen(gjson.GetBytes(body, "tools"))
	diag.FunctionsCount = arrayLen(gjson.GetBytes(body, "functions"))
	diag.InputCount = arrayLen(gjson.GetBytes(responsesBody, "input"))
	diagnoseChatToolCalls(body, &diag)
	diagnoseResponsesInputItems(responsesBody, &diag)
	return diag
}

func logAnthropicMessagesShapeDiagnostics(ctx context.Context, body []byte, stage string) {
	diag := diagnoseAnthropicMessagesShape(body)
	if !diag.hasFindings() {
		return
	}
	fields := diag.zapFields()
	fields = append(fields, zap.String("stage", stage))
	logger.FromContext(ctx).Warn("gateway.anthropic_request_shape_diagnostics", fields...)
}

func diagnoseAnthropicMessagesShape(body []byte) requestShapeDiagnostics {
	diag := requestShapeDiagnostics{
		MessagesCount: arrayLen(gjson.GetBytes(body, "messages")),
		ToolsCount:    arrayLen(gjson.GetBytes(body, "tools")),
	}
	for _, path := range []string{"stream_options", "group"} {
		if gjson.GetBytes(body, path).Exists() {
			addShapeFinding(&diag, "anthropic."+path+":present")
		}
	}
	if strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "output_config.effort").String())) == "xhigh" {
		addShapeFinding(&diag, "anthropic.output_config.effort:xhigh")
	}
	diagnoseAnthropicMessages(body, &diag)
	diagnoseAnthropicTools(body, &diag)
	diagnoseAnthropicThinkingBudget(body, &diag)
	return diag
}

func diagnoseChatToolCalls(body []byte, diag *requestShapeDiagnostics) {
	gjson.GetBytes(body, "messages").ForEach(func(i, msg gjson.Result) bool {
		msg.Get("tool_calls").ForEach(func(j, tc gjson.Result) bool {
			base := fmt.Sprintf("chat.messages.%d.tool_calls.%d", i.Int(), j.Int())
			if !nonEmptyString(tc.Get("id")) {
				addShapeFinding(diag, base+".id:missing")
			}
			if !nonEmptyString(tc.Get("function.name")) {
				addShapeFinding(diag, base+".function.name:missing")
			}
			return true
		})
		return true
	})
}

func diagnoseResponsesInputItems(body []byte, diag *requestShapeDiagnostics) {
	gjson.GetBytes(body, "input").ForEach(func(i, item gjson.Result) bool {
		base := fmt.Sprintf("responses.input.%d", i.Int())
		switch item.Get("type").String() {
		case "function_call":
			requireShapeString(diag, item, base, "call_id")
			requireShapeString(diag, item, base, "name")
			requireShapeString(diag, item, base, "arguments")
		case "function_call_output":
			requireShapeString(diag, item, base, "call_id")
			requireShapeString(diag, item, base, "output")
		}
		return true
	})
}

func diagnoseAnthropicMessages(body []byte, diag *requestShapeDiagnostics) {
	gjson.GetBytes(body, "messages").ForEach(func(i, msg gjson.Result) bool {
		if msg.Get("cache_control").Exists() {
			addShapeFinding(diag, fmt.Sprintf("anthropic.messages.%d.cache_control:present", i.Int()))
		}
		return true
	})
}

func diagnoseAnthropicTools(body []byte, diag *requestShapeDiagnostics) {
	gjson.GetBytes(body, "tools").ForEach(func(i, tool gjson.Result) bool {
		base := fmt.Sprintf("anthropic.tools.%d", i.Int())
		if tool.Get("function").Exists() {
			addShapeFinding(diag, base+".function:wrapped")
		}
		if tool.Get("custom").Exists() {
			addShapeFinding(diag, base+".custom:wrapped")
		}
		if tool.Get("parameters").Exists() && !tool.Get("input_schema").Exists() {
			addShapeFinding(diag, base+".parameters:needs_input_schema")
		}
		if hasInvalidAnthropicToolName(tool.Get("name")) {
			addShapeFinding(diag, base+".name:invalid")
		}
		if hasInvalidAnthropicToolName(tool.Get("custom.name")) {
			addShapeFinding(diag, base+".custom.name:invalid")
		}
		if hasInvalidAnthropicToolName(tool.Get("function.name")) {
			addShapeFinding(diag, base+".function.name:invalid")
		}
		if tool.Get("defer_loading").Bool() && tool.Get("cache_control").Exists() {
			addShapeFinding(diag, base+".defer_loading_cache_control:conflict")
		}
		return true
	})
}

func hasInvalidAnthropicToolName(name gjson.Result) bool {
	if !name.Exists() || name.Type != gjson.String {
		return false
	}
	return sanitizeAnthropicToolName(name.String()) != name.String()
}

func diagnoseAnthropicThinkingBudget(body []byte, diag *requestShapeDiagnostics) {
	maxTokens := gjson.GetBytes(body, "max_tokens")
	budget := gjson.GetBytes(body, "thinking.budget_tokens")
	if maxTokens.Exists() && budget.Exists() && maxTokens.Int() <= budget.Int() {
		addShapeFinding(diag, "anthropic.thinking.budget_tokens:gte_max_tokens")
	}
}

func requireShapeString(diag *requestShapeDiagnostics, item gjson.Result, base, field string) {
	if !nonEmptyString(item.Get(field)) {
		addShapeFinding(diag, base+"."+field+":missing")
	}
}

func nonEmptyString(v gjson.Result) bool {
	return v.Exists() && strings.TrimSpace(v.String()) != ""
}

func arrayLen(v gjson.Result) int {
	if !v.IsArray() {
		return 0
	}
	return len(v.Array())
}

func addShapeFinding(diag *requestShapeDiagnostics, finding string) {
	if len(diag.Findings) >= requestShapeFindingLimit {
		return
	}
	diag.Findings = append(diag.Findings, finding)
}
