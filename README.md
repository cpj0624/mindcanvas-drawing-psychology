# 心畫室 MindCanvas v2.0 — 完整圖形 × 兒童心理學繪畫情緒探索系統

> **線上互動網頁（GitHub Pages）**：https://cpj0624.github.io/mindcanvas-drawing-psychology/  
> **完整操作手冊**：[docs/操作SOP.md](docs/操作SOP.md)

互動式自由繪畫頁面搭配 **A + B 雙引擎兒童心理學分析系統**，優先從使用者的**完整圖形與表情結構**出發，結合九宮格空間構圖、筆觸動力、繪畫歷程與自發口語敘說，溫柔推估當下的心理狀態（傷心、憤怒、焦慮、孤單、愉悅、平靜）及內心可能想表達的想法。

## v2.0 核心升級亮點

- **方案 A（預設・免 API Key）：完整圖形與表情拓樸引擎＋單色鉛筆去偏**
  - 多筆空間拓樸自動辨識：**微笑臉孔、哭泣落淚臉孔、生氣皺眉臉孔、塗黑遮臉（Koppitz）、有／無門窗房屋（Buck HTP）、太陽與光芒、烏雲落雨、愛心符號、打叉否定（✕）、柵欄受困結構、角落微小人物**。
  - **單色鉛筆去偏**：自動偵測單色線條素描（如預設黑筆），歸零黑色負向扣分，以完整圖形與表情主導分析。
  - **畫布圖形辨識標示框**：以非破壞性虛線框標示辨識出的圖形位置，可一鍵切換顯示／隱藏。
- **方案 B（選用・BYOK 安全架構）：Google Gemini 多模態視覺深度解讀**
  - 使用者可選填個人免費 Gemini API Key 深度解讀複雜故事情境、動物與人物互動。
  - 原始碼**零金鑰**；金鑰僅暫存於瀏覽器單一分頁 `sessionStorage`（關閉分頁即自動清除），透過 `x-goog-api-key` Header 加密直連 Google 官方 API，異常時無縫自動降級回方案 A。

## 專案架構

- `index.html`：GitHub Pages 公開互動網頁（零外部後端依賴即可離線／線上運作，並內建 A + B 雙引擎）。
- `docs/操作SOP.md`：完整日常操作、方案 B 金鑰申請與安全清除、親子／教師藝術治療陪伴指引與 FAQ。
- `backend/knowledge_base.py`：兒童發展心理學與藝術治療文獻、六大情緒原型與引導提問庫。
- `backend/analyzer.py`：v2.0 圖形優先多維度心理分析引擎（含 Pillow 影像像素統計與空間拓樸辨識）。
- `backend/server.py`：FastAPI 後端伺服器（`/api/analyze`、`/api/references`、`/api/health`）。
- `frontend/index.html`：串接 FastAPI `/api/analyze` 之完整前端頁面。
- `tests/test_analyzer.py`：13 項自動化單元測試（含單色黑筆笑臉、暖黃筆哭臉、天藍筆怒臉＋打叉、黑筆 HTP 房屋與黑筆愛心等跨色驗證）。

## 本地測試與啟動

```bash
# 執行 13 項自動化單元測試
python3 -m unittest discover -s tests -v

# 啟動 FastAPI 後端服務
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

## 專業倫理提醒

本工具依據兒童發展心理學與藝術治療投射理論建構（Buck HTP 1948、Koppitz HFD 1968、Lowenfeld 1947、Malchiodi 1998、Burkitt 等人 2003、Alschuler & Hattwick 1947），僅作為促進情緒覺察與親子／自我對話之探索輔助，並非臨床心理或醫療診斷。
