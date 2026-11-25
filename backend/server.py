"""FastAPI backend server for MindCanvas Child Drawing Psychology Studio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .analyzer import analyze_drawing, consult_counselor
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
        "針對 3–13 歲兒童，依據 29 篇國際心理學同儕審查期刊實證與 6 部受限經典架構（共 35 項來源），"
        "以純圖像涵義為優先分析情緒線索、提供 3C 媒體與神經多樣性去偏護欄、作畫歷程亮點對話引導，"
        "並支援輔導人員後設圖像補充重算與實證諮詢。"
    ),
    version="3.5.1",
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
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store, no-cache, must-revalidate, private")
        response.headers.setdefault("Pragma", "no-cache")
    return response


UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class StrokePoint(BaseModel):
    x: float = Field(..., ge=-5000.0, le=10000.0, allow_inf_nan=False)
    y: float = Field(..., ge=-5000.0, le=10000.0, allow_inf_nan=False)
    t: float = Field(default=0.0, ge=0.0, le=86_400_000.0, allow_inf_nan=False)
    p: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    tx: float = Field(default=0.0, ge=-90.0, le=90.0, allow_inf_nan=False)
    ty: float = Field(default=0.0, ge=-90.0, le=90.0, allow_inf_nan=False)
    w: float = Field(default=0.0, ge=0.0, le=200.0, allow_inf_nan=False)


class StrokeItem(BaseModel):
    id: Optional[str] = Field(default=None, max_length=64)
    tool: str = Field(default="crayon", max_length=24)
    color: str = Field(default="#222222", max_length=16)
    size: float = Field(default=6.0, ge=1.0, le=80.0, allow_inf_nan=False)
    pointerType: str = Field(default="mouse", max_length=16)
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
    activeDrawingMs: float = Field(default=0.0, ge=0.0, le=86_400_000.0, allow_inf_nan=False)
    pauseCount: int = Field(default=0, ge=0, le=10000)
    colorChangeCount: int = Field(default=0, ge=0, le=10000)
    toolSwitchCount: int = Field(default=0, ge=0, le=10000)
    sizeChangeCount: int = Field(default=0, ge=0, le=10000)
    paperChangeCount: int = Field(default=0, ge=0, le=1000)
    inputDevices: List[Annotated[str, Field(max_length=16)]] = Field(
        default_factory=list, max_length=8
    )


class UserContext(BaseModel):
    titleOrNote: str = Field(default="", max_length=300)
    ageGroup: str = Field(default="unspecified", max_length=16)
    selectedMotifs: List[Annotated[str, Field(max_length=48)]] = Field(
        default_factory=list, max_length=20
    )


class CounselorContext(BaseModel):
    verifiedObjects: List[Annotated[str, Field(max_length=80)]] = Field(
        default_factory=list, max_length=15
    )
    followUpNotes: str = Field(default="", max_length=600)


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


class HotspotBox(BaseModel):
    x: float = Field(default=0.0, ge=-5000.0, le=10000.0, allow_inf_nan=False)
    y: float = Field(default=0.0, ge=-5000.0, le=10000.0, allow_inf_nan=False)
    w: float = Field(default=0.0, ge=0.0, le=10000.0, allow_inf_nan=False)
    h: float = Field(default=0.0, ge=0.0, le=10000.0, allow_inf_nan=False)


class RepeatHotspot(BaseModel):
    zone: str = Field(default="", max_length=24)
    strokeCount: int = Field(default=0, ge=0, le=5000)
    bbox: Optional[HotspotBox] = None


class RevisionHotspot(BaseModel):
    zone: str = Field(default="", max_length=24)
    count: int = Field(default=0, ge=0, le=5000)
    bbox: Optional[HotspotBox] = None


class DrawingSequenceHighlight(BaseModel):
    firstZone: str = Field(default="", max_length=24)
    lastZone: str = Field(default="", max_length=24)
    totalDrawStrokes: int = Field(default=0, ge=0, le=5000)


class TempoShiftHighlight(BaseModel):
    type: str = Field(default="", max_length=32)
    zone: str = Field(default="", max_length=24)
    pauseSec: Optional[float] = Field(default=None, ge=0.0, le=3600.0, allow_inf_nan=False)
    detail: str = Field(default="", max_length=80)
    strokeIdx: int = Field(default=0, ge=0, le=5000)


class ProcessHighlights(BaseModel):
    repeatHotspot: Optional[RepeatHotspot] = None
    revisionHotspot: Optional[RevisionHotspot] = None
    sequence: Optional[DrawingSequenceHighlight] = None
    tempoShift: Optional[TempoShiftHighlight] = None


class AnalyzeRequest(BaseModel):
    canvas: CanvasMeta = Field(default_factory=CanvasMeta)
    strokes: List[StrokeItem] = Field(default_factory=list, max_length=2000)
    processMetrics: ProcessMetrics = Field(default_factory=ProcessMetrics)
    processHighlights: Optional[ProcessHighlights] = None
    userContext: UserContext = Field(default_factory=UserContext)
    counselorContext: CounselorContext = Field(default_factory=CounselorContext)
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


class ConsultAnalysisSummary(BaseModel):
    primaryLabel: str = Field(default="", max_length=48)
    primaryScore: int = Field(default=0, ge=0, le=100)
    secondaryLabel: str = Field(default="", max_length=48)
    secondaryScore: int = Field(default=0, ge=0, le=100)
    convergenceLabel: str = Field(default="", max_length=64)
    ageNormLabel: str = Field(default="", max_length=48)
    reasoningSummary: str = Field(default="", max_length=500)
    alternativeNote: str = Field(default="", max_length=320)
    strengthsNote: str = Field(default="", max_length=320)
    strokeInsight: str = Field(default="", max_length=320)
    recognizedFigures: List[Annotated[str, Field(max_length=64)]] = Field(
        default_factory=list, max_length=12
    )


class ConsultRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    analysisSummary: ConsultAnalysisSummary = Field(default_factory=ConsultAnalysisSummary)
    counselorContext: CounselorContext = Field(default_factory=CounselorContext)


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "mindcanvas-child-psychology-backend", "version": "3.5.1"}


@app.get("/api/references")
def get_references() -> Dict[str, Any]:
    return {"references": PSYCHOLOGY_REFERENCES}


@app.post("/api/analyze")
def analyze_endpoint(req: AnalyzeRequest) -> JSONResponse:
    result = analyze_drawing(req.model_dump())
    return JSONResponse(content=result)


@app.post("/api/consult")
def consult_endpoint(req: ConsultRequest) -> JSONResponse:
    result = consult_counselor(req.model_dump())
    return JSONResponse(content=result)


@app.get("/")
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(index_path)
