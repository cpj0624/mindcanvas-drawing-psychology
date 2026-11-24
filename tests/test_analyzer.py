"""Unit and scenario verification tests for the v2.0 Shape-First child drawing psychology engine."""

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
    steps: int = 36, dt_ms: float = 35.0, tool: str = "crayon"
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
    return {"tool": tool, "color": color, "size": size, "points": pts}


def _make_arc_stroke(
    cx: float, cy: float, r: float, a0: float, a1: float, color: str,
    size: float = 6.0, steps: int = 18, dt_ms: float = 38.0, tool: str = "pencil"
):
    pts = []
    for i in range(steps + 1):
        ang = a0 + (a1 - a0) * (i / steps)
        pts.append({
            "x": cx + r * math.cos(ang),
            "y": cy + r * math.sin(ang),
            "t": i * dt_ms,
            "p": 0.5,
        })
    return {"tool": tool, "color": color, "size": size, "points": pts}


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


def _make_heart_stroke(cx: float, cy: float, scale: float, color: str, size: float = 7.0):
    pts = []
    steps = 48
    for i in range(steps + 1):
        t = (2 * math.pi * i) / steps
        hx = 16 * (math.sin(t) ** 3)
        hy = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append({"x": cx + hx * scale, "y": cy + hy * scale, "t": i * 32.0, "p": 0.55})
    return {"tool": "crayon", "color": color, "size": size, "points": pts}


class TestChildPsychologyAnalyzer(unittest.TestCase):
    def test_empty_canvas(self):
        res = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": []})
        self.assertEqual(res["status"], "empty")

    def test_anger_storm_archetype(self):
        strokes = [
            _make_zigzag_stroke(40, 80, 760, 440, 14, 65, "#D92B2B", size=15, dt_ms=10),
            _make_zigzag_stroke(750, 90, 50, 460, 14, 70, "#1E1E22", size=14, dt_ms=10),
            _make_zigzag_stroke(120, 260, 720, 240, 16, 80, "#B81D24", size=16, dt_ms=9),
            _make_zigzag_stroke(200, 140, 620, 400, 12, 55, "#232328", size=12, dt_ms=11),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "processMetrics": {"undoCount": 0, "eraserStrokeCount": 0, "totalDurationMs": 4200},
        })
        self.assertEqual(res["primaryEmotion"]["key"], "anger", f"Got {res['emotions']}")

    def test_sadness_rain_archetype(self):
        strokes = [
            {"tool": "watercolor", "color": "#6E7B8B", "size": 14, "points": [
                {"x": 190 + i * 14, "y": 195 + math.sin(i * 0.6) * 14, "t": i * 55, "p": 0.45} for i in range(28)
            ]},
            _make_line_stroke(210, 240, 200, 470, "#3A5F8A", size=5, steps=16, dt_ms=70),
            _make_line_stroke(260, 250, 250, 480, "#4B6D94", size=5, steps=16, dt_ms=75),
            _make_line_stroke(310, 230, 302, 475, "#5A6A7A", size=5, steps=16, dt_ms=80),
            _make_line_stroke(360, 260, 355, 490, "#2F4866", size=6, steps=16, dt_ms=75),
            _make_line_stroke(410, 250, 405, 480, "#6E7B8B", size=5, steps=16, dt_ms=80),
            _make_line_stroke(460, 270, 452, 490, "#3A5F8A", size=5, steps=16, dt_ms=85),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "processMetrics": {"undoCount": 1, "eraserStrokeCount": 0, "totalDurationMs": 14000},
        })
        self.assertEqual(res["primaryEmotion"]["key"], "sadness", f"Got {res['emotions']}")
        fig_codes = {f["code"] for f in res["recognizedFigures"]}
        self.assertIn("rain_cloud_figure", fig_codes)

    def test_anxiety_revision_and_shading_archetype(self):
        strokes = []
        for i in range(14):
            cx = 340 + (i % 4) * 28
            cy = 230 + (i // 4) * 24
            strokes.append(
                _make_zigzag_stroke(cx, cy, cx + 36, cy + 16, 6, 12, "#4C4E54" if i % 2 else "#222428", size=6, dt_ms=28)
            )
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "processMetrics": {"undoCount": 6, "eraserStrokeCount": 5, "totalDurationMs": 22000, "pauseCount": 4},
        })
        self.assertEqual(res["primaryEmotion"]["key"], "anxiety", f"Got {res['emotions']}")

    def test_loneliness_tiny_corner_archetype(self):
        # Black pencil tiny stick figure tucked into bottom-left corner
        strokes = [
            _make_circle_stroke(95, 430, 16, "#2A2B30", size=4, steps=24, dt_ms=45, tool="pencil"),
            _make_line_stroke(95, 446, 95, 482, "#2A2B30", size=4, steps=8, dt_ms=55),
            _make_line_stroke(82, 460, 108, 460, "#2A2B30", size=4, steps=6, dt_ms=55),
            _make_line_stroke(95, 482, 82, 505, "#2A2B30", size=4, steps=6, dt_ms=55),
            _make_line_stroke(95, 482, 108, 505, "#2A2B30", size=4, steps=6, dt_ms=55),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "processMetrics": {"undoCount": 0, "eraserStrokeCount": 0, "totalDurationMs": 9000},
        })
        self.assertEqual(res["primaryEmotion"]["key"], "loneliness", f"Got {res['emotions']}")
        fig_codes = {f["code"] for f in res["recognizedFigures"]}
        self.assertIn("person_tiny_isolated", fig_codes)

    def test_joy_sunny_archetype(self):
        strokes = [
            _make_circle_stroke(240, 150, 44, "#F5B82E", size=10, steps=36, dt_ms=26),
            _make_line_stroke(240, 96, 240, 56, "#F59E0B", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_line_stroke(284, 110, 315, 82, "#F59E0B", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_line_stroke(296, 150, 336, 150, "#F5B82E", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_line_stroke(196, 112, 166, 84, "#F59E0B", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_line_stroke(184, 150, 146, 150, "#F5B82E", size=8, steps=8, dt_ms=24, tool="crayon"),
            _make_circle_stroke(480, 280, 56, "#F472B6", size=10, steps=36, dt_ms=28),
            {
                "tool": "watercolor", "color": "#45B36B", "size": 14,
                "points": [{"x": 140 + i * 24, "y": 410 - math.sin(i / 7.0) * 45, "t": i * 32, "p": 0.6} for i in range(23)],
            },
        ]
        res = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": strokes})
        self.assertEqual(res["primaryEmotion"]["key"], "joy", f"Got {res['emotions']}")
        fig_codes = {f["code"] for f in res["recognizedFigures"]}
        self.assertIn("sun_figure", fig_codes)

    def test_calm_harmonious_archetype(self):
        strokes = [
            {"tool": "watercolor", "color": "#3C9D78", "size": 10, "points": [
                {"x": 200 + i * 16, "y": 320 + math.sin(i * 0.35) * 26, "t": i * 42, "p": 0.5} for i in range(26)
            ]},
            {"tool": "watercolor", "color": "#5CA4C4", "size": 10, "points": [
                {"x": 210 + i * 15, "y": 250 + math.sin(i * 0.32) * 22, "t": i * 44, "p": 0.5} for i in range(26)
            ]},
            _make_circle_stroke(400, 210, 52, "#4EA885", size=8, steps=40, dt_ms=40),
        ]
        res = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": strokes})
        self.assertEqual(res["primaryEmotion"]["key"], "calm", f"Got {res['emotions']}")

    # --- NEW v2.0 SHAPE-FIRST & MONOCHROME DEBIASING TESTS ---

    def test_monochrome_black_pencil_happy_face_and_sun_is_joy(self):
        """100% black charcoal pencil smiling face + sun MUST classify as JOY, not sad/angry/anxious!"""
        c = "#2A2B30"
        strokes = [
            # Smiling face in center
            _make_circle_stroke(400, 300, 95, c, size=6, steps=40, dt_ms=32, tool="pencil"),
            _make_circle_stroke(365, 275, 8, c, size=5, steps=16, dt_ms=25, tool="pencil"),
            _make_circle_stroke(435, 275, 8, c, size=5, steps=16, dt_ms=25, tool="pencil"),
            _make_arc_stroke(400, 305, 48, math.pi * 0.15, math.pi * 0.85, c, size=6, steps=20, dt_ms=30, tool="pencil"),
            # Black pencil sun in top-left
            _make_circle_stroke(150, 120, 34, c, size=5, steps=30, dt_ms=30, tool="pencil"),
            _make_line_stroke(150, 78, 150, 48, c, size=5, steps=6, dt_ms=25),
            _make_line_stroke(185, 92, 210, 68, c, size=5, steps=6, dt_ms=25),
            _make_line_stroke(194, 120, 224, 120, c, size=5, steps=6, dt_ms=25),
            _make_line_stroke(114, 92, 90, 68, c, size=5, steps=6, dt_ms=25),
            _make_line_stroke(106, 120, 76, 120, c, size=5, steps=6, dt_ms=25),
        ]
        res = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": strokes})
        self.assertTrue(res["isMonochromeSketch"])
        self.assertEqual(res["primaryEmotion"]["key"], "joy", f"Got {res['emotions']}")
        fig_codes = {f["code"] for f in res["recognizedFigures"]}
        self.assertIn("face_happy", fig_codes)
        self.assertIn("sun_figure", fig_codes)

    def test_warm_yellow_crying_face_is_sadness(self):
        """100% bright warm yellow crying face with tears MUST classify as SADNESS, overriding warm color!"""
        c = "#F5C43D"
        strokes = [
            _make_circle_stroke(400, 280, 100, c, size=7, steps=40, dt_ms=48, tool="crayon"),
            _make_circle_stroke(360, 250, 8, c, size=6, steps=16, dt_ms=45, tool="crayon"),
            _make_circle_stroke(440, 250, 8, c, size=6, steps=16, dt_ms=45, tool="crayon"),
            # Tears falling below left & right eyes
            _make_line_stroke(360, 272, 360, 304, c, size=5, steps=8, dt_ms=55, tool="crayon"),
            _make_line_stroke(440, 272, 440, 304, c, size=5, steps=8, dt_ms=55, tool="crayon"),
            # Inverted-U frown mouth in lower half
            _make_arc_stroke(400, 355, 42, math.pi * 1.15, math.pi * 1.85, c, size=6, steps=18, dt_ms=50, tool="crayon"),
        ]
        res = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": strokes})
        self.assertEqual(res["primaryEmotion"]["key"], "sadness", f"Got {res['emotions']}")
        fig_codes = {f["code"] for f in res["recognizedFigures"]}
        self.assertIn("face_sad_crying", fig_codes)

    def test_sky_blue_angry_face_and_cross_out_is_anger(self):
        """100% sky-blue angry furrowed-brow face + cross-out X MUST classify as ANGER!"""
        c = "#54A9D6"
        strokes = [
            _make_circle_stroke(390, 270, 95, c, size=7, steps=40, dt_ms=22, tool="marker"),
            # Angry inward-sloping eyebrows (\ on left, / on right)
            _make_line_stroke(335, 225, 378, 248, c, size=7, steps=8, dt_ms=16, tool="marker"),
            _make_line_stroke(445, 225, 402, 248, c, size=7, steps=8, dt_ms=16, tool="marker"),
            _make_circle_stroke(358, 255, 6, c, size=5, steps=12, dt_ms=18, tool="marker"),
            _make_circle_stroke(422, 255, 6, c, size=5, steps=12, dt_ms=18, tool="marker"),
            # Tense jagged mouth
            _make_zigzag_stroke(352, 315, 428, 315, 6, 8, c, size=6, dt_ms=14),
            # Large cross-out X on the side
            _make_line_stroke(540, 170, 660, 310, c, size=9, steps=8, dt_ms=14, tool="marker"),
            _make_line_stroke(660, 170, 540, 310, c, size=9, steps=8, dt_ms=14, tool="marker"),
        ]
        res = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": strokes})
        self.assertEqual(res["primaryEmotion"]["key"], "anger", f"Got {res['emotions']}")
        fig_codes = {f["code"] for f in res["recognizedFigures"]}
        self.assertIn("face_angry", fig_codes)
        self.assertIn("cross_out_figure", fig_codes)

    def test_black_pencil_open_house_is_calm(self):
        """Monochrome black pencil HTP house with roof, door & window -> CALM."""
        c = "#2A2B30"
        strokes = [
            # Wall box (4 lines)
            _make_line_stroke(280, 230, 520, 230, c, size=6, steps=10, dt_ms=42),
            _make_line_stroke(520, 230, 520, 410, c, size=6, steps=10, dt_ms=42),
            _make_line_stroke(520, 410, 280, 410, c, size=6, steps=10, dt_ms=42),
            _make_line_stroke(280, 410, 280, 230, c, size=6, steps=10, dt_ms=42),
            # Triangular roof above wall
            _make_line_stroke(260, 235, 400, 120, c, size=6, steps=10, dt_ms=42),
            _make_line_stroke(400, 120, 540, 235, c, size=6, steps=10, dt_ms=42),
            # Door + window inside walls
            _make_line_stroke(375, 410, 375, 325, c, size=5, steps=8, dt_ms=44),
            _make_line_stroke(375, 325, 425, 325, c, size=5, steps=8, dt_ms=44),
            _make_line_stroke(425, 325, 425, 410, c, size=5, steps=8, dt_ms=44),
            _make_circle_stroke(325, 290, 20, c, size=5, steps=24, dt_ms=40, tool="pencil"),
        ]
        res = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": strokes})
        self.assertTrue(res["isMonochromeSketch"])
        self.assertEqual(res["primaryEmotion"]["key"], "calm", f"Got {res['emotions']}")
        fig_codes = {f["code"] for f in res["recognizedFigures"]}
        self.assertIn("house_open", fig_codes)

    def test_black_pencil_heart_is_joy(self):
        """Monochrome black pencil heart MUST classify as JOY, not dark sadness!"""
        strokes = [_make_heart_stroke(400, 275, 8.5, "#2A2B30", size=8)]
        res = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": strokes})
        self.assertEqual(res["primaryEmotion"]["key"], "joy", f"Got {res['emotions']}")
        fig_codes = {f["code"] for f in res["recognizedFigures"]}
        self.assertIn("heart_figure", fig_codes)

    def test_verbal_visual_incongruence_and_pillow_png(self):
        img = Image.new("RGB", (400, 275), (253, 251, 247))
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
            "canvas": {"width": 800, "height": 550, "backgroundColor": "#FDFBF7"},
            "strokes": strokes,
            "imageDataUrl": data_url,
            "userContext": {"titleOrNote": "今天很開心沒事"},
        })
        self.assertEqual(res["status"], "ok")
        self.assertIsNotNone(res["incongruenceNote"])
        self.assertGreater(res["pillars"]["spatial"]["inkCoverage"], 0.02)

    def test_security_extreme_coordinate_dos_protection(self):
        """Extreme coordinate jump must not trigger CPU/RAM exhaustion in _resample_points."""
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [{
                "tool": "pencil",
                "color": "#222222",
                "size": 5,
                "points": [
                    {"x": 0.0, "y": 0.0, "t": 0.0, "p": 0.5},
                    {"x": 1e12, "y": 1e12, "t": 100.0, "p": 0.5},
                ],
            }],
        })
        self.assertEqual(res["status"], "ok")

    def test_security_malformed_client_image_stats_and_nan_fallback(self):
        """Malformed clientImageStats and non-finite coordinates must safely fall back without 500."""
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [{
                "tool": "pencil",
                "color": "#222222",
                "size": 5,
                "points": [
                    {"x": float("nan"), "y": 10.0, "t": 0.0, "p": 0.5},
                    {"x": 100.0, "y": 100.0, "t": 10.0, "p": 0.5},
                    {"x": 220.0, "y": 180.0, "t": 50.0, "p": 0.5},
                ],
            }],
            "imageDataUrl": "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
            "clientImageStats": {"inkCoverageRatio": 0.5, "grid3x3": [0.1]},
        })
        self.assertEqual(res["status"], "ok")

    def test_security_pydantic_request_validation(self):
        """FastAPI AnalyzeRequest must reject NaN, malformed clientImageStats, and oversized total points."""
        from pydantic import ValidationError
        from backend.server import AnalyzeRequest

        with self.assertRaises(ValidationError):
            AnalyzeRequest(strokes=[{"points": [{"x": float("nan"), "y": 10.0}]}])

        with self.assertRaises(ValidationError):
            AnalyzeRequest(strokes=[{"points": [{"x": 1e9, "y": 10.0}]}])

        with self.assertRaises(ValidationError):
            AnalyzeRequest(clientImageStats={"inkCoverageRatio": 0.4})

        # 20 strokes * 3000 points = 60,000 > MAX_TOTAL_STROKE_POINTS (50,000)
        heavy_pts = [{"x": float(i % 800), "y": 100.0} for i in range(3000)]
        with self.assertRaises(ValidationError):
            AnalyzeRequest(strokes=[{"points": heavy_pts} for _ in range(20)])


if __name__ == "__main__":
    unittest.main()
