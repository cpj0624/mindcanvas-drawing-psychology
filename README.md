# 心畫室 MindCanvas v3.4 — 內在情緒分析 × 圖像互動引導 × 輔導人員 AI 諮詢工作室

> **線上互動網頁（GitHub Pages）**：https://cpj0624.github.io/mindcanvas-drawing-psychology/  
> **完整操作手冊**：[docs/操作SOP.md](docs/操作SOP.md)

互動式自由繪畫頁面搭配 **A + B 雙引擎兒童心理學探索系統**。v3.4 採用**「純圖像涵義實證庫（7 大類別・24 篇國際期刊＋6 大經典架構）＋ 雙方向分頁 ＋ 輔導人員專屬 AI 諮詢與重算閉環」**設計：
1. **分頁 1｜內在情緒分析**：導入 **24 篇國際心理學同儕審查期刊實證（以圖像符號、空間界線、保護物件與敘事為主而非色彩）**、**軟體繪畫介面筆觸感測**與**互動式三層判斷邏輯樹狀圖**。
2. **分頁 2｜圖像互動引導（邊畫邊說）**：依據 Gross & Hayne (1998) 與 Driessnack (2005, `d = 0.95`) 統合分析設計，純粹透過畫作引導兒童表達內心想法，**全程隱藏情緒分數、診斷標籤與情緒足跡條，並自動收合下方大人諮詢區**，可安心與孩子並肩共看。
3. **雙分頁下方｜輔導人員 AI 諮詢與圖像協助辨識（僅限大人・繪畫與分析完成後使用）**：支援輔導人員補充畫中模糊／抽象區塊的實際物件與後續訪談紀錄、自動對應純圖像實證架構（雨中人 DAPR 保護傘、KFD 包覆圈／阻隔物、依附地面支撐、鳥巢畫 BND 安全基地、PPAT 主動性工具、HFD 部位省略發展常模），並一鍵 **「納入補充資訊並重新解析雙分頁」**。

## v3.4 核心升級亮點

- **純圖像涵義實證文獻庫擴充與分類呈現（7 大類別・21 組收錄・24 篇國際期刊＋6 大經典架構）**
  - 新增 7 組以圖像符號、人際界線、因應物件與敘事為核心（完全不依賴色彩）的國際同行評審實證：
    1. **雨中人 DAPR 壓力 vs. 保護因應物件**：Willis, Joy, & Kaiser (2010), *The Arts in Psychotherapy* (`r = .43, p < .01`)。
    2. **家庭畫依附關係與空間支撐**：Fury, Carlson, & Sroufe (1997), *Child Development*；Pace, Muzi, & Vizzino (2022), *Frontiers in Psychology*。
    3. **動態家庭畫 KFD 界線與互動**：Wegmann & Lusebrink (2000), *The Arts in Psychotherapy*（包覆圈、格子分隔、障礙物、視線與照顧動作）。
    4. **鳥巢畫 BND 安全基地與承接容器**：Kaiser (1996)；Kaiser & Deaver (2009), *Art Therapy*。
    5. **摘蘋果的人 PPAT 主動性與問題解決工具**：Bat Or, Ishai, Barkay, & Shalev (2022), *Children*。
    6. **人物畫 HFD 部位省略、誇大姿態與年齡發展常模**：Skybo, Ryan-Wenger, & Su (2007), *Journal of Pediatric Nursing*。
    7. **以圖引談（Draw-and-Tell）跨研究統合分析**：Driessnack (2005), *Journal of Pediatric Nursing*（溝通促進效果量 `d = 0.95`）。
  - 方案 A（本地分析／推理樹／正向資源／本地諮詢）與方案 B（Gemini 視覺系統提示與輔導顧問）已全面同步導入，並於介面「理論與文獻」視窗中依 7 大主題類別分組呈現。
- **輔導人員 AI 諮詢與後設圖像辨識重算閉環（僅限大人・純文字・無麥克風權限）**
  - 擺放於雙分頁正下方且支援一鍵收合；切換至「圖像互動引導」陪孩子共看時自動收合諮詢內容，保護兒童隱私與安全感。
  - **高風險安全與專業轉介護欄**：若補充紀錄或諮詢對話涉及自傷、受暴等危機線索，系統立即攔截並呈現專業求助資源（`1925`、`1995`、`113`、`110/119` 與機構通報轉介流程）。
- **方案 B（選用・BYOK 零留存資安架構）**
  - 原始碼與公開頁面**零金鑰、零追蹤、零伺服器留存**；金鑰僅存單一分頁 `sessionStorage`（關閉即銷毀），嚴格經 `x-goog-api-key` Header 加密直連 Google 官方 API，所有 API 回應皆加註 `Cache-Control: no-store`。

## 專案架構

- `index.html`：GitHub Pages 公開互動網頁（零外部後端依賴即可離線／線上運作，內建 v3.4 21 組分類文獻庫、雙分頁、雙引擎與輔導人員 AI 諮詢重算區）。
- `docs/操作SOP.md`：完整日常操作、雙分頁切換、輔導人員課後／訪談後補充辨識與 AI 諮詢 SOP、方案 B 金鑰安全管理與資安檢查清單。
- `backend/knowledge_base.py`：7 大類別共 21 組國際期刊實證與經典理論架構清單、分齡常模、畫具硬度表、六大情緒原型與引導提問庫。
- `backend/analyzer.py`：v3.4 純圖像實證多維度心理分析、軟體介面筆觸動力、三層判斷邏輯樹狀圖、Draw & Tell 對話引導、輔導員補充重算與本地諮詢引擎（`consult_counselor`）。
- `backend/server.py`：FastAPI 後端伺服器（`/api/analyze`、`/api/consult`、`/api/references`、`/api/health`，具嚴格 Pydantic 邊界驗證、`Cache-Control: no-store` 與完整安全回應標頭）。
- `frontend/index.html`：串接 FastAPI `/api/analyze` 與 `/api/consult` 之完整前端頁面。
- `tests/test_analyzer.py`：29 項自動化單元測試（涵蓋所有情緒原型、雙重色彩去偏、年齡常模、三種裝置筆觸感測、三層邏輯樹、零標籤互動引導、輔導員補充重算、7 組純圖像實證匹配、本地諮詢與危機轉介護欄及資安邊界驗證）。

## 本地測試與啟動

```bash
# 執行 29 項自動化單元測試
python3 -m unittest discover -s tests -v

# 啟動 FastAPI 後端服務（預設僅綁定本機 127.0.0.1 以保護隱私）
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

## 專業倫理提醒

本工具整合 24 篇國際心理學同儕審查期刊實證與 6 大經典架構（HTP／Koppitz／Lowenfeld／Malchiodi／Alschuler & Hattwick／Pulver），嚴格遵循「整體構圖多證據收斂、不以單一線索下診斷定論」原則，僅作為促進情緒覺察與親子／輔導對話之探索輔助，並非臨床心理或醫療診斷。
