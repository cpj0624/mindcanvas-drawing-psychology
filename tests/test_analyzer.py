"""Unit and scenario verification tests for the v2.0 Shape-First child drawing psychology engine."""

from __future__ import annotations

import base64
import io
import math
import unittest

from PIL import Image, ImageDraw

from backend.analyzer import analyze_drawing, consult_counselor


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

        # New v3.0 telemetry bounds validation
        with self.assertRaises(ValidationError):
            AnalyzeRequest(strokes=[{"points": [{"x": 10.0, "y": 10.0, "tx": 180.0}]}])

        valid_v3 = AnalyzeRequest(
            strokes=[{
                "tool": "pencil",
                "color": "#2A2B30",
                "size": 6,
                "pointerType": "pen",
                "points": [{"x": 100.0, "y": 100.0, "t": 0.0, "p": 0.72, "tx": 18.0, "ty": -12.0, "w": 4.0}],
            }],
            processMetrics={
                "activeDrawingMs": 3200.0,
                "totalDurationMs": 9500.0,
                "colorChangeCount": 2,
                "toolSwitchCount": 1,
                "sizeChangeCount": 1,
                "inputDevices": ["pen"],
            },
            userContext={"titleOrNote": "測試", "ageGroup": "7-10"},
        )
        self.assertEqual(valid_v3.userContext.ageGroup, "7-10")

    def test_v3_face_no_mouth_and_age_group_attenuation(self):
        """Omitted mouth detects face_no_mouth; age 3-6 attenuates omitted-detail penalty (Picard 2012; Nanayama Tanaka 2020)."""
        c = "#2A2B30"
        strokes = [
            _make_circle_stroke(400, 270, 95, c, size=6, steps=36, dt_ms=35, tool="pencil"),
            _make_circle_stroke(365, 245, 9, c, size=5, steps=16, dt_ms=30, tool="pencil"),
            _make_circle_stroke(435, 245, 9, c, size=5, steps=16, dt_ms=30, tool="pencil"),
        ]
        res_young = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "userContext": {"ageGroup": "3-6"},
        })
        res_older = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "userContext": {"ageGroup": "11+"},
        })
        self.assertIn("face_no_mouth", {f["code"] for f in res_young["recognizedFigures"]})
        self.assertIn("face_no_mouth", {f["code"] for f in res_older["recognizedFigures"]})
        self.assertEqual(res_young["ageNorm"]["omittedDetailScale"], 0.0)
        self.assertEqual(res_older["ageGroup"], "11-13")
        self.assertEqual(res_older["ageNorm"]["omittedDetailScale"], 1.0)
        young_anx = next(e["score"] for e in res_young["emotions"] if e["key"] == "anxiety")
        older_anx = next(e["score"] for e in res_older["emotions"] if e["key"] == "anxiety")
        self.assertLessEqual(young_anx, older_anx)

    def test_v3_open_rounded_smile_nanayama_tanaka(self):
        """Enclosed wide rounded open smile -> face_happy & joy (Nanayama Tanaka et al., 2020)."""
        c = "#2A2B30"
        # Oval open mouth inside lower face
        mouth_pts = [
            {"x": 400 + 38 * math.cos(2 * math.pi * i / 28), "y": 318 + 20 * math.sin(2 * math.pi * i / 28), "t": i * 25.0, "p": 0.5}
            for i in range(29)
        ]
        strokes = [
            _make_circle_stroke(400, 270, 100, c, size=6, steps=36, dt_ms=30, tool="pencil"),
            _make_circle_stroke(365, 240, 8, c, size=5, steps=16, dt_ms=25, tool="pencil"),
            _make_circle_stroke(435, 240, 8, c, size=5, steps=16, dt_ms=25, tool="pencil"),
            {"tool": "pencil", "color": c, "size": 5, "points": mouth_pts},
        ]
        res = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": strokes})
        self.assertIn("face_happy", {f["code"] for f in res["recognizedFigures"]})
        self.assertEqual(res["primaryEmotion"]["key"], "joy", f"Got {res['emotions']}")

    def test_v3_canonical_nature_palette_debiasing_crawford(self):
        """Blue sky + green grass + warm sun must trigger canonical nature palette debiasing (Crawford et al., 2012)."""
        strokes = [
            # Blue sky waves at top
            _make_line_stroke(60, 75, 740, 75, "#3B82F6", size=14, steps=18, dt_ms=32, tool="watercolor"),
            _make_line_stroke(60, 110, 740, 110, "#60A5FA", size=14, steps=18, dt_ms=32, tool="watercolor"),
            # Green grass hills at bottom
            _make_line_stroke(50, 440, 750, 440, "#43A047", size=14, steps=18, dt_ms=35, tool="crayon"),
            _make_line_stroke(50, 470, 750, 470, "#2E7D32", size=14, steps=18, dt_ms=35, tool="crayon"),
            # Warm yellow sun
            _make_circle_stroke(640, 130, 38, "#FFB300", size=10, steps=28, dt_ms=30, tool="crayon"),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "processMetrics": {"colorChangeCount": 3, "toolSwitchCount": 1},
        })
        self.assertTrue(res["isCanonicalNaturePalette"])
        self.assertIn(res["primaryEmotion"]["key"], ("joy", "calm"), f"Got {res['emotions']}")

    def test_v3_software_interface_stroke_sensing_modes(self):
        """Software interface sensing differentiates mouse/pen/touch, force proxies, and inAirPauseRatio."""
        # 1) Mouse: heavy marker vs light watercolor (both flat p=0.5)
        heavy_mouse = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [
                {**_make_line_stroke(100, 150, 700, 150, "#D92B2B", size=22, steps=16, dt_ms=10, tool="marker"), "pointerType": "mouse"},
                {**_make_line_stroke(100, 220, 700, 220, "#D92B2B", size=22, steps=16, dt_ms=10, tool="marker"), "pointerType": "mouse"},
            ],
            "processMetrics": {"totalDurationMs": 20000, "activeDrawingMs": 4000, "colorChangeCount": 1, "toolSwitchCount": 1},
        })
        light_mouse = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [
                {**_make_line_stroke(100, 150, 400, 150, "#60A5FA", size=4, steps=16, dt_ms=65, tool="watercolor"), "pointerType": "mouse"},
                {**_make_line_stroke(100, 220, 400, 220, "#60A5FA", size=4, steps=16, dt_ms=65, tool="watercolor"), "pointerType": "mouse"},
            ],
        })
        self.assertEqual(heavy_mouse["pillars"]["strokes"]["sensingMode"], "mouse_interface")
        self.assertGreater(
            heavy_mouse["pillars"]["strokes"]["effectiveForceIndex"],
            light_mouse["pillars"]["strokes"]["effectiveForceIndex"],
        )
        self.assertAlmostEqual(heavy_mouse["pillars"]["strokes"]["inAirPauseRatio"], 0.8, delta=0.05)

        # 2) Stylus pen hardware mode with real varying pressure + tilt
        pen_pts = [
            {"x": 150 + i * 22.0, "y": 260.0, "t": i * 25.0, "p": 0.25 + 0.06 * i, "tx": 24.0, "ty": 14.0, "w": 2.0}
            for i in range(12)
        ]
        pen_res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [{"tool": "pencil", "color": "#2A2B30", "size": 6, "pointerType": "pen", "points": pen_pts}],
        })
        self.assertEqual(pen_res["pillars"]["strokes"]["sensingMode"], "pen_hardware")

        # 3) Touch contact width mode
        touch_pts = [
            {"x": 150 + i * 22.0, "y": 260.0, "t": i * 25.0, "p": 0.5, "w": 36.0}
            for i in range(10)
        ]
        touch_res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [{"tool": "crayon", "color": "#2A2B30", "size": 8, "pointerType": "touch", "points": touch_pts}],
        })
        self.assertEqual(touch_res["pillars"]["strokes"]["sensingMode"], "touch_contact")

    def test_v3_low_evidence_convergence_score_cap(self):
        """Minimal single-stroke input must stay in low convergence and cap score <= 66 (Joiner 1996; Catte & Cox 1999)."""
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [_make_line_stroke(300, 260, 480, 260, "#2A2B30", size=6, steps=6, dt_ms=40, tool="pencil")],
        })
        self.assertEqual(res["evidenceConvergence"]["level"], "low")
        self.assertLessEqual(res["primaryEmotion"]["score"], 66)
        self.assertTrue(res["evidenceConvergence"]["alternativeNote"])

    def test_v3_references_and_penzes_structure_balance(self):
        """Output includes all 14+ verified peer-reviewed references plus Pénzes structureBalance & strengthsNote."""
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [
                _make_circle_stroke(400, 275, 90, "#43A047", size=8, steps=32, dt_ms=35, tool="crayon"),
                _make_line_stroke(180, 390, 620, 390, "#FFB300", size=10, steps=16, dt_ms=35, tool="crayon"),
            ],
        })
        self.assertEqual(res["engineMode"], "local_empirical_v3")
        self.assertGreaterEqual(len(res["references"]), 13)
        self.assertIn("code", res["structureBalance"])
        self.assertTrue(res["strengthsNote"])
        self.assertTrue(res["expressiveChannels"])

    def test_v3_explainable_reasoning_tree_structure(self):
        """Analyzer must return a complete 3-level reasoningTree explaining judgment logic and evidence."""
        c = "#2A2B30"
        strokes = [
            _make_circle_stroke(400, 280, 95, c, size=6, steps=36, dt_ms=32, tool="pencil"),
            _make_circle_stroke(365, 255, 8, c, size=5, steps=16, dt_ms=25, tool="pencil"),
            _make_circle_stroke(435, 255, 8, c, size=5, steps=16, dt_ms=25, tool="pencil"),
            _make_arc_stroke(400, 290, 46, math.pi * 0.15, math.pi * 0.85, c, size=6, steps=20, dt_ms=30, tool="pencil"),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "userContext": {"titleOrNote": "開心畫畫", "ageGroup": "7-10"},
        })
        tree = res.get("reasoningTree")
        self.assertIsInstance(tree, dict)
        self.assertIn("結論總結", tree["rootTitle"])
        self.assertTrue(tree["rootSummary"])
        branches = tree["branches"]
        self.assertEqual([b["id"] for b in branches], ["figures", "strokes", "spatial", "color", "context_guardrail"])
        # Figure branch must be primary and list face_happy with impact delta
        fig_branch = branches[0]
        self.assertEqual(fig_branch["status"], "primary")
        self.assertTrue(any("face_happy" in ch["id"] and "+" in ch["tag"] for ch in fig_branch["children"]))
        # Color branch must reflect monochrome debiasing
        color_branch = branches[3]
        self.assertEqual(color_branch["status"], "attenuated")
        self.assertIn("單色", color_branch["badge"])
        # Every child node must contain title, summary, detail, citation
        for b in branches:
            self.assertGreaterEqual(len(b["children"]), 1)
            for ch in b["children"]:
                for field in ("id", "title", "tag", "summary", "detail", "citation"):
                    self.assertIn(field, ch)
                    self.assertTrue(ch[field], f"Empty {field} in {ch}")

    def test_v3_dialogue_guide_draw_and_tell_non_diagnostic(self):
        """Analyzer must return a 4-stage non-diagnostic Draw & Tell dialogueGuide adapted to drawing and age."""
        c = "#2A2B30"
        strokes = [
            _make_circle_stroke(105, 420, 17, c, size=4, steps=24, dt_ms=50, tool="pencil"),
            _make_line_stroke(105, 438, 105, 480, c, size=4, steps=10, dt_ms=52, tool="pencil"),
            _make_line_stroke(86, 456, 124, 456, c, size=4, steps=8, dt_ms=52, tool="pencil"),
            _make_line_stroke(105, 480, 91, 506, c, size=4, steps=8, dt_ms=52, tool="pencil"),
            _make_line_stroke(105, 480, 119, 506, c, size=4, steps=8, dt_ms=52, tool="pencil"),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "userContext": {"ageGroup": "3-6"},
        })
        dg = res.get("dialogueGuide")
        self.assertIsInstance(dg, dict)
        self.assertIn("3–6 歲", dg["ageToneLabel"])
        self.assertTrue(dg["openingCue"])
        self.assertEqual(len(dg["stages"]), 4)
        for st in dg["stages"]:
            self.assertGreaterEqual(len(st["prompts"]), 2)
            self.assertTrue(st["followUpTip"])
        self.assertIn("title", dg["coDrawIdea"])
        self.assertGreaterEqual(len(dg["avoidPhrases"]), 2)
        # Monochrome tiny corner drawing should trigger anti-black-bias & anti-size-labeling avoid tips
        joined_avoid = " ".join(item["avoid"] for item in dg["avoidPhrases"])
        self.assertIn("黑色", joined_avoid)
        self.assertIn("小", joined_avoid)

    def test_v33_counselor_context_reanalysis_fusion(self):
        """Counselor verifiedObjects & followUpNotes must enrich tree Branch 1/5, motifs, and dialogueGuide."""
        strokes = [
            _make_circle_stroke(140, 410, 28, "#2A2B30", size=5, steps=24, dt_ms=42, tool="pencil"),
            _make_line_stroke(100, 380, 180, 380, "#2A2B30", size=5, steps=10, dt_ms=42, tool="pencil"),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "userContext": {"ageGroup": "7-10"},
            "counselorContext": {
                "verifiedObjects": ["左下角＝躲在桌下的小狗", "橫線＝保護罩"],
                "followUpNotes": "孩子訪談表示小狗因為換新班級有點害怕不敢出來說話",
            },
        })
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["counselorContext"]["hasContext"])
        self.assertIsNone(res["safetyAlert"])
        self.assertTrue(any("輔導員確認：左下角＝躲在桌下的小狗" in m["label"] for m in res["detectedMotifs"]))
        fig_branch = res["reasoningTree"]["branches"][0]
        self.assertEqual(fig_branch["status"], "primary")
        self.assertTrue(any(ch["id"] == "fig_counselor_verified" for ch in fig_branch["children"]))
        ctx_branch = res["reasoningTree"]["branches"][4]
        self.assertTrue(any(ch["id"] == "ctx_counselor" for ch in ctx_branch["children"]))
        self.assertIn("左下角＝躲在桌下的小狗", res["dialogueGuide"]["openingCue"])

    def test_v33_counselor_consultation_and_crisis_guardrail(self):
        """consult_counselor handles clarification, reasoning Q&A, resistance coaching, and urgent crisis referral."""
        summary = {
            "primaryLabel": "緊張與焦慮",
            "primaryScore": 74,
            "secondaryLabel": "孤單與渴望連結",
            "secondaryScore": 68,
            "convergenceLabel": "中度收斂（探索性參考）",
            "ageNormLabel": "7–10 歲（國小中低）",
            "reasoningSummary": "筆觸修正較多且主體縮聚於一隅。",
            "alternativeNote": "也可能與剛接觸數位畫布不熟悉有關。",
            "strengthsNote": "仍保有清晰的邊界整理資源。",
            "recognizedFigures": ["微小人物"],
        }
        r_clarify = consult_counselor({
            "question": "左下角那一塊其實是躲起來的烏龜",
            "analysisSummary": summary,
            "counselorContext": {},
        })
        self.assertEqual(r_clarify["status"], "ok")
        self.assertIsNone(r_clarify["safetyAlert"])
        self.assertIn("烏龜", r_clarify["suggestedObject"] or "")

        r_why = consult_counselor({
            "question": "為什麼焦慮和孤單分數很接近？有哪些替代解釋？",
            "analysisSummary": summary,
            "counselorContext": {},
        })
        self.assertIn("替代解釋", r_why["reply"])
        self.assertIn("6 分", r_why["reply"])

        r_silent = consult_counselor({
            "question": "孩子回答不知道也不想說，建議怎麼接話？",
            "analysisSummary": summary,
            "counselorContext": {},
        })
        self.assertIn("第三人稱", r_silent["reply"])

        r_crisis = consult_counselor({
            "question": "孩子訪談中提到在家受暴且有自傷想法",
            "analysisSummary": summary,
            "counselorContext": {},
        })
        self.assertIsNotNone(r_crisis["safetyAlert"])
        self.assertEqual(r_crisis["safetyAlert"]["level"], "urgent")
        self.assertIn("1925", r_crisis["reply"])
        self.assertIn("113", r_crisis["reply"])

    def test_v33_security_consult_request_validation(self):
        """FastAPI ConsultRequest & CounselorContext must enforce strict bounds against DoS/overflow."""
        from pydantic import ValidationError
        from backend.server import AnalyzeRequest, ConsultRequest

        with self.assertRaises(ValidationError):
            ConsultRequest(question="")
        with self.assertRaises(ValidationError):
            ConsultRequest(question="x" * 501)
        with self.assertRaises(ValidationError):
            AnalyzeRequest(counselorContext={"verifiedObjects": [f"obj{i}" for i in range(16)]})
        with self.assertRaises(ValidationError):
            AnalyzeRequest(counselorContext={"followUpNotes": "a" * 601})

    def test_v34_image_first_literature_catalog_and_groups(self):
        """PSYCHOLOGY_REFERENCES must include all 26 categorized entries across 8 groups (29 studies + 6 classics)."""
        from backend.knowledge_base import PSYCHOLOGY_REFERENCES

        ids = {r["id"] for r in PSYCHOLOGY_REFERENCES}
        self.assertEqual(len(PSYCHOLOGY_REFERENCES), 26)
        for req_id in (
            "willis_2010_dapr",
            "fury_1997_pace_2022",
            "wegmann_lusebrink_2000",
            "kaiser_1996_2009_bnd",
            "bat_or_2022_ppat",
            "skybo_2007_hfd",
            "driessnack_2005_meta",
        ):
            self.assertIn(req_id, ids)
        for r in PSYCHOLOGY_REFERENCES:
            self.assertTrue(r.get("group"), f"Missing group for {r['id']}")

    def test_v34_image_first_frameworks_in_analysis_and_consultation(self):
        """Counselor verified image motifs (umbrella, encapsulation, nest, ladder, missing hands) trigger exact empirical citations."""
        strokes = [
            _make_circle_stroke(360, 260, 48, "#2A2B30", size=5, steps=30, dt_ms=40, tool="pencil"),
            _make_line_stroke(310, 200, 410, 200, "#2A2B30", size=5, steps=10, dt_ms=40, tool="pencil"),
        ]
        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "userContext": {"ageGroup": "7-10"},
            "counselorContext": {
                "verifiedObjects": ["主角拿著大雨傘", "外面畫了一個大框框", "旁邊有梯子和小幫手"],
                "followUpNotes": "孩子說撐雨傘躲雨，梯子是用來拿樹上的果實",
            },
        })
        self.assertEqual(res["status"], "ok")
        self.assertIn("Driessnack (2005)", res["dialogueGuide"]["theoryBasis"])
        self.assertIn("Willis", res["dialogueGuide"]["theoryBasis"])
        fig_branch = res["reasoningTree"]["branches"][0]
        child_ids = {c["id"] for c in fig_branch["children"]}
        self.assertIn("fig_fw_dapr_coping", child_ids)
        self.assertIn("fig_fw_kfd_boundaries", child_ids)
        self.assertIn("fig_fw_ppat_agency", child_ids)
        self.assertIn("雨中人", res["strengthsNote"])

        summary = {
            "primaryLabel": "平靜與安全感",
            "primaryScore": 72,
            "secondaryLabel": "焦慮與壓力",
            "secondaryScore": 58,
            "convergenceLabel": "中度收斂",
            "ageNormLabel": "7–10 歲",
            "recognizedFigures": [],
        }
        r_dapr = consult_counselor({"question": "主角在雨中撐著雨傘代表什麼？", "analysisSummary": summary})
        self.assertIn("Willis", r_dapr["reply"])

        r_bnd = consult_counselor({"question": "畫面裡有一個很穩固的鳥巢和秘密基地", "analysisSummary": summary})
        self.assertIn("Kaiser", r_bnd["reply"])

        r_hfd = consult_counselor({"question": "人物沒畫手腳而且有大牙齒怎麼解讀？", "analysisSummary": summary})
        self.assertIn("Skybo", r_hfd["reply"])

        r_silent = consult_counselor({"question": "孩子說不知道不想講，怎麼接話？", "analysisSummary": summary})
        self.assertIn("Driessnack", r_silent["reply"])

    def test_v35_five_new_studies_and_8_groups_no_stale_14(self):
        """v3.5 must include all 5 new 3C/ASD studies across 8 groups and zero stale 14 篇 text."""
        from backend.knowledge_base import PSYCHOLOGY_REFERENCES
        from backend.server import health_check

        self.assertEqual(health_check()["version"], "3.5.0")
        by_id = {r["id"]: r for r in PSYCHOLOGY_REFERENCES}
        groups = {r["group"] for r in PSYCHOLOGY_REFERENCES}
        self.assertEqual(len(PSYCHOLOGY_REFERENCES), 26)
        self.assertEqual(len(groups), 8)

        expected_dois = {
            "ivashkevich_2009_popculture": "10.1080/00393541.2009.11518790",
            "price_2015_ipads": "10.1016/j.compedu.2015.04.003",
            "mohd_shukri_2014_tablet": "10.1145/2559206.2581278",
            "papangelo_2020_asd_hfd": "10.3390/brainsci10060398",
            "kotroni_2018_asd_emotions": "10.1080/20473869.2018.1434855",
        }
        for ref_id, doi in expected_dois.items():
            self.assertIn(ref_id, by_id)
            self.assertEqual(by_id[ref_id]["doi"], doi)

        res = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [_make_circle_stroke(400, 270, 80, "#2A2B30", size=6)],
        })
        self.assertEqual(res["engineVersion"], "3.5.0")
        self.assertIn("29 篇", res["ethicalNotice"])
        self.assertNotIn("14 篇", res["ethicalNotice"])

    def test_v35_pulver_graphology_zeroed_objective_coords_only(self):
        """Pulver left/right/top/bottom psychological symbolism must be completely removed (weight = 0)."""
        top_strokes = [_make_circle_stroke(160, 120, 45, "#2A2B30", size=6, steps=28, dt_ms=35, tool="pencil")]
        bot_strokes = [_make_circle_stroke(640, 430, 45, "#2A2B30", size=6, steps=28, dt_ms=35, tool="pencil")]
        res_top = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": top_strokes})
        res_bot = analyze_drawing({"canvas": {"width": 800, "height": 550}, "strokes": bot_strokes})

        banned = ("內向回顧", "尋求依附", "面向外界", "探索環境", "想像思緒", "渴望抽離", "尋求踏實", "情緒沉重")
        for res in (res_top, res_bot):
            spatial_txt = res["pillars"]["spatial"]["insight"]
            spatial_branch = next(b for b in res["reasoningTree"]["branches"] if b["id"] == "spatial")
            center_node = next(c for c in spatial_branch["children"] if c["id"] == "spatial_center")
            combined = spatial_txt + center_node["summary"] + center_node["detail"] + center_node["tag"]
            for phrase in banned:
                self.assertNotIn(phrase, combined)

        self.assertIn("水平偏左", res_top["pillars"]["spatial"]["insight"])
        self.assertIn("垂直偏上", res_top["pillars"]["spatial"]["insight"])
        self.assertIn("水平偏右", res_bot["pillars"]["spatial"]["insight"])
        self.assertIn("垂直偏下", res_bot["pillars"]["spatial"]["insight"])

        # Bottom placement must NOT receive Pulver's old +0.18 (~+12~15 pt) sadness boost over top placement
        sad_top = next(e["score"] for e in res_top["emotions"] if e["key"] == "sadness")
        sad_bot = next(e["score"] for e in res_bot["emotions"] if e["key"] == "sadness")
        self.assertLessEqual(abs(sad_bot - sad_top), 1)
        self.assertLessEqual(sad_bot, sad_top + 1)

    def test_v35_koppitz_3_6_zero_omission_and_house_closed_parity(self):
        """At age 3-6, house_closed, face_no_mouth, and person_blank_face must all have 0.0 omission penalty."""
        c = "#2A2B30"
        # Closed rectangle wall + triangle roof (no doors/windows) -> house_closed
        house_strokes = [
            {
                "tool": "pencil", "color": c, "size": 6,
                "points": [
                    {"x": 300, "y": 280, "t": 0, "p": 0.5},
                    {"x": 500, "y": 280, "t": 100, "p": 0.5},
                    {"x": 500, "y": 430, "t": 200, "p": 0.5},
                    {"x": 300, "y": 430, "t": 300, "p": 0.5},
                    {"x": 300, "y": 280, "t": 400, "p": 0.5},
                ],
            },
            _make_line_stroke(285, 280, 400, 185, c, size=6, steps=12, dt_ms=30, tool="pencil"),
            _make_line_stroke(400, 185, 515, 280, c, size=6, steps=12, dt_ms=30, tool="pencil"),
        ]
        res_young = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": house_strokes,
            "userContext": {"ageGroup": "3-6"},
        })
        res_older = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": house_strokes,
            "userContext": {"ageGroup": "11-13"},
        })
        hc_young = next((f for f in res_young["recognizedFigures"] if f["code"] == "house_closed"), None)
        hc_older = next((f for f in res_older["recognizedFigures"] if f["code"] == "house_closed"), None)
        self.assertIsNotNone(hc_young)
        self.assertIsNotNone(hc_older)
        self.assertEqual(hc_young["impact"]["anxiety"], 0.0)
        self.assertEqual(hc_young["impact"]["loneliness"], 0.0)
        self.assertAlmostEqual(hc_older["impact"]["anxiety"], 0.10, places=2)
        self.assertAlmostEqual(hc_older["impact"]["loneliness"], 0.10, places=2)
        # Blank face person at age 3-6 must also be zeroed
        blank_person = [
            _make_circle_stroke(400, 200, 36, c, size=5, steps=28, dt_ms=30, tool="pencil"),
            _make_line_stroke(400, 236, 400, 360, c, size=5, steps=10, dt_ms=30, tool="pencil"),
        ]
        res_bp = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": blank_person,
            "userContext": {"ageGroup": "3-6"},
        })
        bp_fig = next((f for f in res_bp["recognizedFigures"] if f["code"] == "person_blank_face"), None)
        self.assertIsNotNone(bp_fig)
        self.assertEqual(bp_fig["impact"]["loneliness"], 0.0)
        self.assertEqual(bp_fig["impact"]["anxiety"], 0.0)

    def test_v35_dynamic_attenuation_and_alschuler_device_split(self):
        """Tablet multi-stroke + undos must not misclassify a happy face as anxiety; pressure only activates on pen_hardware."""
        c = "#2A2B30"
        smile_pts = [
            {"x": 400 + 46 * math.cos(math.radians(a)), "y": 295 + 32 * math.sin(math.radians(a)), "t": i * 25.0, "p": 0.5, "w": 22.0}
            for i, a in enumerate(range(20, 165, 10))
        ]
        happy_tablet_strokes = [
            {**_make_circle_stroke(400, 270, 95, c, size=6, steps=36, dt_ms=30, tool="crayon"), "pointerType": "touch"},
            {**_make_circle_stroke(365, 240, 8, c, size=5, steps=16, dt_ms=25, tool="crayon"), "pointerType": "touch"},
            {**_make_circle_stroke(435, 240, 8, c, size=5, steps=16, dt_ms=25, tool="crayon"), "pointerType": "touch"},
            {"tool": "crayon", "color": c, "size": 6, "pointerType": "touch", "points": smile_pts},
            # Add 5 short multi-stroke decorative segments typical of tablet drawing (Mohd Shukri, 2014)
            {**_make_line_stroke(200, 140, 224, 146, "#F5C43D", size=7, steps=5, dt_ms=25), "pointerType": "touch"},
            {**_make_line_stroke(230, 148, 254, 152, "#F5C43D", size=7, steps=5, dt_ms=25), "pointerType": "touch"},
            {**_make_line_stroke(560, 140, 584, 146, "#F5C43D", size=7, steps=5, dt_ms=25), "pointerType": "touch"},
            {**_make_line_stroke(590, 148, 612, 154, "#F5C43D", size=7, steps=5, dt_ms=25), "pointerType": "touch"},
        ]
        res_touch = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": happy_tablet_strokes,
            "processMetrics": {"undoCount": 3, "eraserStrokeCount": 1, "pauseCount": 2, "totalDurationMs": 18000, "inputDevices": ["touch"]},
            "userContext": {"ageGroup": "3-6"},
        })
        self.assertEqual(res_touch["primaryEmotion"]["key"], "joy", f"Got {res_touch['emotions']}")
        self.assertFalse(res_touch["pillars"]["strokes"]["penPressureEmotionActive"])

        # Hardware stylus with real pressure variance activates low-weight pressure flag
        pen_strokes = [{
            "tool": "crayon",
            "color": "#D62828",
            "size": 10,
            "pointerType": "pen",
            "points": [
                {"x": 150 + i * 25, "y": 240 + (i % 2) * 50, "t": i * 20.0, "p": 0.25 + (i % 5) * 0.16, "tx": 15, "ty": 10}
                for i in range(18)
            ],
        }]
        res_pen = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": pen_strokes,
            "processMetrics": {"inputDevices": ["pen"]},
        })
        self.assertTrue(res_pen["pillars"]["strokes"]["penPressureEmotionActive"])
        self.assertEqual(res_pen["pillars"]["strokes"]["sensingMode"], "pen_hardware")

    def test_v35_process_highlights_dialogue_and_pydantic_validation(self):
        """processHighlights must generate concrete Tab 2 open-ended questions and validate via FastAPI AnalyzeRequest."""
        from pydantic import ValidationError
        from backend.server import AnalyzeRequest

        strokes = [
            _make_circle_stroke(180, 150, 45, "#2A2B30", size=6, steps=24, dt_ms=30, tool="pencil"),
            _make_circle_stroke(620, 410, 45, "#2A2B30", size=14, steps=24, dt_ms=30, tool="marker"),
        ]
        highlights_in = {
            "repeatHotspot": {"zone": "左上", "strokeCount": 7, "bbox": {"x": 120, "y": 90, "w": 110, "h": 110}},
            "revisionHotspot": {"zone": "中央", "count": 4, "bbox": {"x": 340, "y": 220, "w": 120, "h": 110}},
            "sequence": {"firstZone": "左上", "lastZone": "右下", "totalDrawStrokes": 9},
            "tempoShift": {"type": "long_pause", "zone": "右下", "pauseSec": 4.2, "detail": "", "strokeIdx": 8},
        }
        req = AnalyzeRequest(strokes=strokes, processHighlights=highlights_in)
        res = analyze_drawing(req.model_dump())

        s1 = res["dialogueGuide"]["stages"][0]["prompts"]
        s3 = res["dialogueGuide"]["stages"][2]["prompts"]
        self.assertTrue(any("左上" in q and "重複畫" in q for q in s1), f"Stage 1 prompts: {s1}")
        self.assertTrue(any("中央" in q and "修改" in q for q in s1), f"Stage 1 prompts: {s1}")
        self.assertTrue(any("左上" in q and "右下" in q for q in s3), f"Stage 3 prompts: {s3}")
        self.assertEqual(res["processHighlights"]["repeatHotspot"]["zone"], "左上")

        # Invalid negative count or huge zone string rejected by Pydantic
        with self.assertRaises(ValidationError):
            AnalyzeRequest(processHighlights={"revisionHotspot": {"zone": "x" * 40, "count": -1}})

    def test_v35_age_7_10_pop_culture_debias_and_11_13_preadolescent(self):
        """Age 7-10 pop-culture cues attenuate anger/anxiety and trigger media-source check; 11-13 adds realism-anxiety note."""
        strokes = [
            _make_zigzag_stroke(150, 180, 620, 360, 10, 45, "#2A2B30", size=10, dt_ms=22),
            _make_circle_stroke(400, 260, 70, "#2A2B30", size=6, steps=30, dt_ms=30),
        ]
        res_7_10 = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": strokes,
            "userContext": {"ageGroup": "7-10", "titleOrNote": "動漫裡有尖牙和武器的怪獸角色", "selectedMotifs": ["monster_storm"]},
        })
        self.assertTrue(len(res_7_10["popCultureHits"]) >= 2)
        s2_prompts = res_7_10["dialogueGuide"]["stages"][1]["prompts"]
        self.assertTrue(any("影片" in q or "漫畫" in q or "遊戲" in q for q in s2_prompts), f"s2={s2_prompts}")
        ctx_branch = res_7_10["reasoningTree"]["branches"][4]
        self.assertIn("ctx_pop_culture", {c["id"] for c in ctx_branch["children"]})

        res_11_13 = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": [_make_circle_stroke(400, 260, 50, "#2A2B30", size=4, steps=24, dt_ms=35, tool="pencil")],
            "processMetrics": {"undoCount": 4},
            "userContext": {"ageGroup": "11-13"},
        })
        self.assertEqual(res_11_13["ageGroup"], "11-13")
        self.assertIn("怕畫不像", res_11_13["evidenceConvergence"]["alternativeNote"])

    def test_v35_modern_symbols_neurodiversity_asd_and_tablet_consult(self):
        """Contemporary symbols (考卷/耳機/棉被), ASD guardrail (Papangelo 2020; Kotroni 2018), and tablet consultation."""
        # Geometric repetitive straight strokes without emotional face -> triggers ASD guardrail
        geo_strokes = [
            _make_line_stroke(160 + i * 50, 150, 160 + i * 50, 390, "#2A2B30", size=4, steps=10, dt_ms=35, tool="pencil")
            for i in range(9)
        ]
        res_asd = analyze_drawing({
            "canvas": {"width": 800, "height": 550},
            "strokes": geo_strokes,
            "userContext": {"ageGroup": "7-10", "titleOrNote": "火車軌道和機械齒輪，戴耳機看考試卷"},
        })
        self.assertIsNotNone(res_asd["neurodiversityNote"])
        self.assertIn("Papangelo", res_asd["neurodiversityNote"])
        self.assertIn("Kotroni", res_asd["neurodiversityNote"])
        ctx_ids = {c["id"] for c in res_asd["reasoningTree"]["branches"][4]["children"]}
        self.assertIn("ctx_neurodiversity", ctx_ids)
        # Modern symbols matched in VERBAL_LEXICON (考卷/考試卷 -> anxiety, 耳機 -> calm)
        matched_v = res_asd["pillars"]["process"]["matchedVerbalCues"]
        self.assertIn("考試卷", matched_v.get("anxiety", []))
        self.assertIn("耳機", matched_v.get("calm", []))

        summary = {"primaryLabel": "平靜與安全感", "primaryScore": 70, "secondaryLabel": "焦慮與壓力", "secondaryScore": 54}
        r_tab = consult_counselor({"question": "孩子在平板上一直按復原又畫多段短筆畫，會誤判焦慮嗎？", "analysisSummary": summary})
        self.assertIn("Price", r_tab["reply"])
        self.assertIn("Mohd Shukri", r_tab["reply"])

        r_asd = consult_counselor({"question": "畫中都是重複幾何和機械火車軌道細節，表情很少", "analysisSummary": summary})
        self.assertIn("Papangelo", r_asd["reply"])
        self.assertIn("Kotroni", r_asd["reply"])


if __name__ == "__main__":
    unittest.main()



