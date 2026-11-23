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
    id: Optional[str] = Field(default=None, max_length=64)
    tool: str = Field(default="crayon", max_length=24)
    color: str = Field(default="#222222", max_length=16)
    size: float = Field(default=6.0, ge=1.0, le=80.0)
    points: List[StrokePoint] = Field(default_factory=list, max_length=4000)


class CanvasMeta(BaseModel):
    width: float = Field(default=800.0, ge=50.0, le=4096.0)
    height: float = Field(default=540.0, ge=50.0, le=4096.0)
    backgroundColor: str = Field(default="#FDFBF7", max_length=16)


class ProcessMetrics(BaseModel):
    undoCount: int = Field(default=0, ge=0, le=10000)
    redoCount: int = Field(default=0, ge=0, le=10000)
    clearCount: int = Field(default=0, ge=0, le=1000)
    eraserStrokeCount: int = Field(default=0, ge=0, le=10000)
    totalDurationMs: float = Field(default=0.0, ge=0.0, le=86_400_000.0)
    pauseCount: int = Field(default=0, ge=0, le=10000)


class UserContext(BaseModel):
    titleOrNote: str = Field(default="", max_length=300)
    selectedMotifs: List[str] = Field(default_factory=list, max_length=20)


class AnalyzeRequest(BaseModel):
    canvas: CanvasMeta = Field(default_factory=CanvasMeta)
    strokes: List[StrokeItem] = Field(default_factory=list, max_length=2000)
    processMetrics: ProcessMetrics = Field(default_factory=ProcessMetrics)
    userContext: UserContext = Field(default_factory=UserContext)
    imageDataUrl: Optional[str] = Field(default=None, max_length=6_000_000)
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
