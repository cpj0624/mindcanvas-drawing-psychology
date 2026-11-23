"""FastAPI backend server for MindCanvas Child Drawing Psychology Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .analyzer import analyze_drawing
from .knowledge_base import PSYCHOLOGY_REFERENCES

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="心畫室 MindCanvas — 兒童心理學繪畫情緒分析後端",
    description=(
        "依據兒童繪畫心理學（Buck HTP 屋樹人、Koppitz 情緒指標、Lowenfeld、"
        "Malchiodi、Burkitt 色彩研究）分析使用者自由繪畫之色彩、筆觸動力、"
        "空間構圖與繪畫歷程，推估情緒狀態與內在想表達之想法。"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StrokePoint(BaseModel):
    x: float
    y: float
    t: float = 0.0
    p: float = 0.5


class StrokeItem(BaseModel):
    id: Optional[str] = None
    tool: str = "crayon"
    color: str = "#222222"
    size: float = 6.0
    points: List[StrokePoint] = Field(default_factory=list)


class CanvasMeta(BaseModel):
    width: float = 800.0
    height: float = 540.0
    backgroundColor: str = "#FDFBF7"


class ProcessMetrics(BaseModel):
    undoCount: int = 0
    redoCount: int = 0
    clearCount: int = 0
    eraserStrokeCount: int = 0
    totalDurationMs: float = 0.0
    pauseCount: int = 0


class UserContext(BaseModel):
    titleOrNote: str = ""
    selectedMotifs: List[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    canvas: CanvasMeta = Field(default_factory=CanvasMeta)
    strokes: List[StrokeItem] = Field(default_factory=list)
    processMetrics: ProcessMetrics = Field(default_factory=ProcessMetrics)
    userContext: UserContext = Field(default_factory=UserContext)
    imageDataUrl: Optional[str] = None
    clientImageStats: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "mindcanvas-child-psychology-backend"}


@app.get("/api/references")
def get_references() -> Dict[str, Any]:
    return {"references": PSYCHOLOGY_REFERENCES}


@app.post("/api/analyze")
def analyze_endpoint(req: AnalyzeRequest) -> JSONResponse:
    result = analyze_drawing(req.model_dump())
    return JSONResponse(content=result)


@app.get("/")
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(index_path)
