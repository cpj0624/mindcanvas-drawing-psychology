# 心畫室 MindCanvas — 兒童心理學繪畫情緒探索系統

互動式自由繪畫頁面搭配兒童心理學分析後端，從使用者的繪畫色彩、線條動力、九宮格構圖比例、繪畫歷程與自發口語敘說，溫柔推估當下的心理狀態（傷心、憤怒、焦慮、孤單、愉悅、平靜）及內心可能想表達的想法。

## 專案架構

- `backend/knowledge_base.py`：兒童發展心理學與藝術治療參考文獻、六大情緒光譜原型、內心獨白與陪伴對話提問庫。
- `backend/analyzer.py`：多維度心理學分析引擎，整合：
  - **色彩投射（HSV 分桶、明度／彩度）** — Burkitt, Barrett & Davis (2003–2005); Alschuler & Hattwick (1947)
  - **筆觸動力學（RDP 轉折角銳利度、流暢度、繪製速度、重疊塗抹密度）** — Lowenfeld (1987); Malchiodi (1998)
  - **空間構圖九宮格重心與邊界比例（含 Pillow 影像像素統計）** — Buck 屋樹人 HTP (1948); Koppitz (1968)
  - **繪畫歷程、幾何象徵辨識與口語敘說／情緒掩飾比對** — Koppitz 情緒指標; Malchiodi 自發性口語敘說
- `backend/server.py`：FastAPI 後端伺服器，提供 `/api/analyze`、`/api/references`、`/api/health` 與靜態前端頁面。
- `frontend/index.html`：支援蠟筆、鉛筆、水彩、麥克筆與橡皮擦之自由繪畫前端，串接 `/api/analyze`（離線時亦自動降級使用本地引擎）。
- `tests/test_analyzer.py`：涵蓋六大情緒原型、Pillow 像素分析與言畫對照（情緒掩飾）之自動化測試。

## 啟動方式

```bash
# 執行自動化單元測試
python3 -m unittest discover -s tests -v

# 啟動 FastAPI 後端服務
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

## 專業倫理提醒

本工具依據兒童發展心理學與藝術治療投射理論建構，僅作為促進情緒覺察與親子／自我對話之探索輔助，並非臨床心理或醫療診斷。
