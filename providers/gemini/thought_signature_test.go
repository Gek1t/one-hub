package gemini_test

import (
	"encoding/json"
	"one-api/providers/gemini"
	"one-api/types"
	"strings"
	"testing"
)

func TestThoughtSignatureRoundTrip(t *testing.T) {
	rawSignature := "test_encrypted_thought_signature_blob_12345"
	quotedSig, _ := json.Marshal(rawSignature)

	// 1. 模拟 Gemini/Vertex 返回带有 thought_signature 的 candidate
	candidate := gemini.GeminiChatCandidate{
		Content: gemini.GeminiChatContent{
			Role: "model",
			Parts: []gemini.GeminiPart{
				{
					FunctionCall: &gemini.GeminiFunctionCall{
						Name: "default_api:bash",
						Args: map[string]interface{}{
							"command": "ls -la",
						},
					},
					ThoughtSignatureSnake: json.RawMessage(quotedSig),
				},
			},
		},
	}

	// 2. 出包测试：转换为 OpenAI Choice
	req := &types.ChatCompletionRequest{Model: "gemini-3.7-flash"}
	choice := candidate.ToOpenAIChoice(req)

	if len(choice.Message.ToolCalls) == 0 {
		t.Fatalf("Expected tool calls, got 0")
	}
	toolCall := choice.Message.ToolCalls[0]
	t.Logf("Generated ToolCall ID: %s", toolCall.Id)

	if !strings.Contains(toolCall.Id, "___TS___") {
		t.Fatalf("Expected ToolCall ID to contain '___TS___', got %s", toolCall.Id)
	}

	// 3. 入包测试：模拟 OpenCode 回传带有该 ID 的 ToolCalls
	assistantMsg := types.ChatCompletionMessage{
		Role: "assistant",
		ToolCalls: []*types.ChatCompletionToolCalls{
			toolCall,
		},
	}

	contents, _, err := gemini.OpenAIToGeminiChatContent([]types.ChatCompletionMessage{assistantMsg})
	if err != nil {
		t.Fatalf("OpenAIToGeminiChatContent failed: %v", err)
	}

	if len(contents) == 0 || len(contents[0].Parts) == 0 {
		t.Fatalf("Expected gemini content parts, got empty")
	}

	part := contents[0].Parts[0]
	recoveredSig := part.GetThoughtSignature()
	t.Logf("Recovered ThoughtSignature: %s", recoveredSig)

	if recoveredSig != rawSignature {
		t.Fatalf("Expected signature %s, got %s", rawSignature, recoveredSig)
	}
}
