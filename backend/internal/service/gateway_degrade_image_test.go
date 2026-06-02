package service

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/png"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// makePNGBase64 生成指定尺寸 PNG 的 base64 数据。
func makePNGBase64(t *testing.T, w, h int) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func TestRemoveOversizedImages_Dimension(t *testing.T) {
	big := makePNGBase64(t, 9000, 10) // 宽超 8000
	body := `{"messages":[{"role":"user","content":[
		{"type":"text","text":"看图"},
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + big + `"}}
	]}]}`
	out, ok := removeOversizedImages([]byte(body))
	if !ok {
		t.Fatal("expected oversized image removed")
	}
	blocks := gjson.GetBytes(out, "messages.0.content").Array()
	if len(blocks) != 1 {
		t.Fatalf("content blocks=%d, want 1(图片应被删,文本保留)", len(blocks))
	}
	if blocks[0].Get("type").String() != "text" {
		t.Fatal("remaining block should be text")
	}
}

func TestRemoveOversizedImages_Size(t *testing.T) {
	// data 长度 > 5MB*4/3，触发大小判断(无需真图，解码前即 return)
	huge := strings.Repeat("A", 7*1024*1024)
	body := `{"messages":[{"role":"user","content":[
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + huge + `"}},
		{"type":"text","text":"hi"}
	]}]}`
	out, ok := removeOversizedImages([]byte(body))
	if !ok {
		t.Fatal("expected oversized(by bytes) removed")
	}
	if n := len(gjson.GetBytes(out, "messages.0.content").Array()); n != 1 {
		t.Fatalf("content blocks=%d, want 1", n)
	}
}

func TestRemoveOversizedImages_KeepNormal(t *testing.T) {
	small := makePNGBase64(t, 100, 100)
	body := `{"messages":[{"role":"user","content":[
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + small + `"}}
	]}]}`
	if _, ok := removeOversizedImages([]byte(body)); ok {
		t.Fatal("normal image should be kept")
	}
}
