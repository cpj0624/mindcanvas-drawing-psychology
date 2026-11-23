"""Unit and scenario verification tests for the child drawing psychology engine."""

from __future__ import annotations

import base64
import io
import math
import unittest

from PIL import Image, ImageDraw

from backend.analyzer import analyze_drawing


def _make_zigzag_stroke(
    x0: float, y0: float, x1: float, y1: float, zigzags: int, amp: float,
    color: str, size: float = 12.0, dt_ms: float = 14.0
):
    pts = []
    t = 0.0
    for i in range(zigzags + 1):
        u = i / zigzags
        bx = x0 + (x1 - x0) * u
        by = y0 + (y1 - y0) * u
        offset = amp if i % 2 == 0 else -amp
        pts.append({"x": bx, "y": by + offset, "t": t, "p": 0.9})
        t += dt_ms
    return {"tool": "marker", "color": color, "size": size, "points": pts}


def _make_circle_stroke(
    cx: float, cy: float, r: float, color: str, size: float = 8.0,
    steps: int = 36, dt_ms: float = 35.0
):
    pts = []
    for i in range(steps + 1):
        ang = (2 * math.pi * i) / steps
        pts.append({
            "x": cx + r * math.cos(ang),
            "y": cy + r * math.sin(ang),
            "t": i * dt_ms,
            "p": 0.55,
        })
    return {"tool": "crayon", "color": color, "size": size, "points": pts}


def _make_line_stroke(
    x0: float, y0: float, x1: float, y1: float, color: str,
    size: float = 5.0, steps: int = 12, dt_ms: float = 55.0, tool: str = "pencil"
):
    pts = []
    for i in range(steps + 1):
        u = i / steps
        pts.append({
            "x": x0 + (x1 - x0) * u,
            "y": y0 + (y1 - y0) * u,
            "t": i * dt_ms,
            "p": 0.4,
        })
    return {"tool": tool, "color": color, "size": size, "points": pts}


class TestChildPsychologyAnalyzer(unittest.TestCase):
    def test_empty_canvas(self):
        res = analyze_drawing({"canvas": {"width": 800, "height": 540}, "strokes": []})
        self.assertEqual(res["status"], "empty")

    def test_anger_storm_archetype(self):
        strokes = [
            _make_zigzag_stroke(40, 80, 760, 440, 14, 65, "#D92B2B", size=15, dt_ms=10),
            _make_zigzag_stroke(750, 90, 50, 460, 14, 70, "#1E1E22", size=14, dt_ms=10),
            _make_zigzag_stroke(120, 260, 720, 240, 16, 80, "#B81D24", size=16, dt_ms=9),
            _make_zigzag_stroke(200, 140, 620, 400, 12, 55, "#232328", size=12, dt_ms=11),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 540},
            "strokes": strokes,
            "processMetrics": {"undoCount": 0, "eraserStrokeCount": 0, "totalDurationMs": 4200},
        })
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["primaryEmotion"]["key"], "anger", f"Got {res['emotions']}")
        self.assertGreater(res["pillars"]["strokes"]["sharpnessIndex"], 0.4)

    def test_sadness_rain_archetype(self):
        # Cool dark/blue downward rain streaks + drooping arc in lower canvas
        strokes = [
            _make_line_stroke(210, 240, 200, 470, "#3A5F8A", size=5, steps=16, dt_ms=70),
            _make_line_stroke(260, 250, 250, 480, "#4B6D94", size=5, steps=16, dt_ms=75),
            _make_line_stroke(310, 230, 302, 475, "#5A6A7A", size=5, steps=16, dt_ms=80),
            _make_line_stroke(360, 260, 355, 490, "#2F4866", size=6, steps=16, dt_ms=75),
            _make_line_stroke(410, 250, 405, 480, "#6E7B8B", size=5, steps=16, dt_ms=80),
            _make_line_stroke(460, 270, 452, 490, "#3A5F8A", size=5, steps=16, dt_ms=85),
            _make_line_stroke(510, 260, 505, 470, "#2A3240", size=6, steps=16, dt_ms=70),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 540},
            "strokes": strokes,
            "processMetrics": {"undoCount": 1, "eraserStrokeCount": 0, "totalDurationMs": 14000},
        })
        self.assertEqual(res["primaryEmotion"]["key"], "sadness", f"Got {res['emotions']}")
        motif_codes = {m["code"] for m in res["detectedMotifs"]}
        self.assertIn("downward_streaks", motif_codes)

    def test_anxiety_revision_and_shading_archetype(self):
        # Fragmented short dark/grey scribbles with heavy over-drawing + high undo/eraser counts
        strokes = []
        for i in range(14):
            cx = 340 + (i % 4) * 28
            cy = 230 + (i // 4) * 24
            strokes.append(
                _make_zigzag_stroke(cx, cy, cx + 36, cy + 16, 6, 12, "#4C4E54" if i % 2 else "#222428", size=6, dt_ms=28)
            )
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 540},
            "strokes": strokes,
            "processMetrics": {
                "undoCount": 6,
                "eraserStrokeCount": 5,
                "totalDurationMs": 22000,
                "pauseCount": 4,
            },
        })
        self.assertEqual(res["primaryEmotion"]["key"], "anxiety", f"Got {res['emotions']}")
        motif_codes = {m["code"] for m in res["detectedMotifs"]}
        self.assertIn("heavy_revision", motif_codes)

    def test_loneliness_tiny_corner_archetype(self):
        # Tiny cool/grey circle + tiny stick strokes tucked into bottom-left corner
        strokes = [
            _make_circle_stroke(95, 430, 16, "#5C738D", size=4, steps=24, dt_ms=45),
            _make_line_stroke(95, 446, 95, 482, "#5C738D", size=4, steps=8, dt_ms=55),
            _make_line_stroke(82, 460, 108, 460, "#7C8794", size=4, steps=6, dt_ms=55),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 540},
            "strokes": strokes,
            "processMetrics": {"undoCount": 0, "eraserStrokeCount": 0, "totalDurationMs": 9000},
        })
        self.assertEqual(res["primaryEmotion"]["key"], "loneliness", f"Got {res['emotions']}")
        motif_codes = {m["code"] for m in res["detectedMotifs"]}
        self.assertIn("tiny_corner_cluster", motif_codes)

    def test_joy_sunny_archetype(self):
        # Warm yellow sun with radiating rays + bright green hill + pink flower across balanced center
        strokes = [
            _make_circle_stroke(240, 150, 44, "#F5B82E", size=10, steps=36, dt_ms=26),
            _make_line_stroke(240, 92, 240, 56, "#F59E0B", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_line_stroke(284, 110, 315, 82, "#F59E0B", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_line_stroke(296, 150, 336, 150, "#F5B82E", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_line_stroke(196, 112, 166, 84, "#F59E0B", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_line_stroke(184, 150, 146, 150, "#F5B82E", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_circle_stroke(480, 280, 56, "#F472B6", size=10, steps=36, dt_ms=28),
            _make_circle_stroke(480, 280, 22, "#FBBF24", size=10, steps=28, dt_ms=28),
            # Rolling green hill curve
            {
                "tool": "watercolor",
                "color": "#45B36B",
                "size": 14,
                "points": [
                    {"x": 140 + i * 24, "y": 410 - math.sin(i / 7.0) * 45, "t": i * 32, "p": 0.6}
                    for i in range(23)
                ],
            },
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 540},
            "strokes": strokes,
            "processMetrics": {"undoCount": 0, "eraserStrokeCount": 0, "totalDurationMs": 12000},
        })
        self.assertEqual(res["primaryEmotion"]["key"], "joy", f"Got {res['emotions']}")

    def test_calm_harmonious_archetype(self):
        # Centered smooth waves & circles in tranquil green/sky-blue with steady pace & zero revisions
        strokes = [
            {
                "tool": "watercolor",
                "color": "#3C9D78",
                "size": 10,
                "points": [
                    {"x": 200 + i * 16, "y": 320 + math.sin(i * 0.35) * 26, "t": i * 42, "p": 0.5}
                    for i in range(26)
                ],
            },
            {
                "tool": "watercolor",
                "color": "#5CA4C4",
                "size": 10,
                "points": [
                    {"x": 210 + i * 15, "y": 250 + math.sin(i * 0.32) * 22, "t": i * 44, "p": 0.5}
                    for i in range(26)
                ],
            },
            _make_circle_stroke(400, 210, 52, "#4EA885", size=8, steps=40, dt_ms=40),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 540},
            "strokes": strokes,
            "processMetrics": {"undoCount": 0, "eraserStrokeCount": 0, "totalDurationMs": 15000},
        })
        self.assertEqual(res["primaryEmotion"]["key"], "calm", f"Got {res['emotions']}")

    def test_verbal_visual_incongruence_and_pillow_png(self):
        # Draw angry red/black jagged strokes on a real PIL image while writing "我很開心沒事"
        img = Image.new("RGB", (400, 270), (253, 251, 247))
        draw = ImageDraw.Draw(img)
        draw.line([(20, 40), (380, 230), (40, 240), (370, 50)], fill=(217, 43, 43), width=10)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        strokes = [
            _make_zigzag_stroke(40, 80, 760, 460, 14, 70, "#D92B2B", size=16, dt_ms=10),
            _make_zigzag_stroke(760, 90, 50, 460, 14, 70, "#1E1E22", size=14, dt_ms=10),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 540, "backgroundColor": "#FDFBF7"},
            "strokes": strokes,
            "imageDataUrl": data_url,
            "userContext": {"titleOrNote": "今天很開心沒事"},
        })
        self.assertEqual(res["status"], "ok")
        self.assertIsNotNone(res["incongruenceNote"])
        self.assertGreater(res["pillars"]["spatial"]["inkCoverage"], 0.02)


if __name__ == "__main__":
    unittest.main()
