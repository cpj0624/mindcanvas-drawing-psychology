"""FastAPI backend server for MindCanvas Child Drawing Psychology Studio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .analyzer import analyze_drawing
from .knowledge_base import PSYCHOLOGY_REFERENCES

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_TOTAL_STROKE_POINTS = 50_000

_env_origins = os.getenv("MINDCANVAS_ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else ["http://localhost:8000", "http://127.0.0.1:8000"]
)

app = FastAPI(
    title="心畫室 MindCanvas — 兒童心理學繪畫情緒分析後端",
    description=(
        "依據兒童繪畫心理學（Buck HTP 屋樹人、Koppitz 情緒指標、Lowenfeld、"
        "Malchiodi、Burkitt 色彩研究）分析使用者自由繪畫之色彩、筆觸動力、"
        "空間構圖與繪畫歷程，推估情緒狀態與內在想表達之想法。"
    ),
    version="2.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class StrokePoint(BaseModel):
    x: float = Field(..., ge=-5000.0, le=10000.0, allow_inf_nan=False)
    y: float = Field(..., ge=-5000.0, le=10000.0, allow_inf_nan=False)
    t: float = Field(default=0.0, ge=0.0, le=86_400_000.0, allow_inf_nan=False)
    p: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)


class StrokeItem(BaseModel):
    id: Optional[str] = Field(default=None, max_length=64)
    tool: str = Field(default="crayon", max_length=24)
    color: str = Field(default="#222222", max_length=16)
    size: float = Field(default=6.0, ge=1.0, le=80.0, allow_inf_nan=False)
    points: List[StrokePoint] = Field(default_factory=list, max_length=4000)


class CanvasMeta(BaseModel):
    width: float = Field(default=800.0, ge=50.0, le=4096.0, allow_inf_nan=False)
    height: float = Field(default=540.0, ge=50.0, le=4096.0, allow_inf_nan=False)
    backgroundColor: str = Field(default="#FDFBF7", max_length=16)


class ProcessMetrics(BaseModel):
    undoCount: int = Field(default=0, ge=0, le=10000)
    redoCount: int = Field(default=0, ge=0, le=10000)
    clearCount: int = Field(default=0, ge=0, le=1000)
    eraserStrokeCount: int = Field(default=0, ge=0, le=10000)
    totalDurationMs: float = Field(default=0.0, ge=0.0, le=86_400_000.0, allow_inf_nan=False)
    pauseCount: int = Field(default=0, ge=0, le=10000)


class UserContext(BaseModel):
    titleOrNote: str = Field(default="", max_length=300)
    selectedMotifs: List[Annotated[str, Field(max_length=48)]] = Field(
        default_factory=list, max_length=20
    )


class ClientImageStats(BaseModel):
    inkCoverageRatio: UnitFloat
    grid3x3: List[UnitFloat] = Field(..., min_length=9, max_length=9)
    colorHistogram: Dict[Annotated[str, Field(max_length=32)], UnitFloat]
    meanSaturation: UnitFloat
    meanLuminance: UnitFloat
    pixelCenter: Optional[Tuple[UnitFloat, UnitFloat]] = None

    @field_validator("colorHistogram")
    @classmethod
    def limit_histogram_keys(cls, value: Dict[str, float]) -> Dict[str, float]:
        if len(value) > 16:
            raise ValueError("colorHistogram exceeds allowed key count")
        return value


class AnalyzeRequest(BaseModel):
    canvas: CanvasMeta = Field(default_factory=CanvasMeta)
    strokes: List[StrokeItem] = Field(default_factory=list, max_length=2000)
    processMetrics: ProcessMetrics = Field(default_factory=ProcessMetrics)
    userContext: UserContext = Field(default_factory=UserContext)
    imageDataUrl: Optional[str] = Field(default=None, max_length=6_000_000)
    clientImageStats: Optional[ClientImageStats] = None

    @model_validator(mode="after")
    def limit_total_stroke_points(self) -> "AnalyzeRequest":
        total_pts = sum(len(s.points) for s in self.strokes)
        if total_pts > MAX_TOTAL_STROKE_POINTS:
            raise ValueError(
                f"Total stroke points ({total_pts}) exceeds maximum limit ({MAX_TOTAL_STROKE_POINTS})"
            )
        return self


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
