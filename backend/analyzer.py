"""Multi-dimensional child drawing psychology analysis engine (v2.0 Shape-First).

Extracts objective composite figures, facial expressions, HTP structures,
stroke kinematics, spatial composition, process telemetry, and optional
Pillow pixel statistics from a freehand drawing. Prioritizes complete
graphic structures and facial expressions over raw color, with automatic
monochrome/pencil-test debiasing aligned with Buck HTP (1948), Koppitz
HFD (1968), Lowenfeld (1987), Malchiodi (1998), and Burkitt et al. (2003).
"""

from __future__ import annotations

import base64
import colorsys
import io
import math
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .knowledge_base import (
    EMOTION_KEYS,
    EMOTION_PROFILES,
    PSYCHOLOGY_REFERENCES,
    USER_MOTIF_WEIGHTS,
    VERBAL_LEXICON,
)


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    cleaned = (hex_color or "#222222").strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(c * 2 for c in cleaned)
    if len(cleaned) != 6:
        return (34, 34, 34)
    try:
        return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))
    except ValueError:
        return (34, 34, 34)


def classify_rgb_bucket(r: int, g: int, b: int) -> Tuple[str, float, float]:
    """Classify an RGB pixel into psychological color affect buckets."""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    hue_deg = h * 360.0
    lum = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf

    if v < 0.24 or (lum < 0.20 and s < 0.45):
        return ("darkMuted", s, lum)
    if s < 0.14:
        return ("darkMuted" if v < 0.48 else "neutralGrey", s, lum)
    if v < 0.40 and (hue_deg < 48 or hue_deg >= 255):
        return ("darkMuted", s, lum)
    if (hue_deg <= 14 or hue_deg >= 342) and s >= 0.48 and v >= 0.35:
        if v >= 0.72 and s < 0.65:
            return ("gentlePinkPurple", s, lum)
        return ("intenseRed", s, lum)
    if 14 < hue_deg <= 68:
        if v < 0.48 and s < 0.65:
            return ("darkMuted", s, lum)
        return ("warmBright", s, lum)
    if 68 < hue_deg <= 165:
        if v < 0.34:
            return ("darkMuted", s, lum)
        return ("freshGreen", s, lum)
    if 165 < hue_deg <= 255:
        return ("coolBlue", s, lum)
    if 255 < hue_deg < 342:
        if v < 0.42:
            return ("darkMuted", s, lum)
        return ("gentlePinkPurple", s, lum)
    return ("neutralGrey", s, lum)


def _resample_points(
    points: List[Dict[str, Any]], step: float = 12.0
) -> List[Tuple[float, float]]:
    if not points:
        return []
    out: List[Tuple[float, float]] = [(float(points[0]["x"]), float(points[0]["y"]))]
    acc = 0.0
    prev_x, prev_y = out[0]
    for pt in points[1:]:
        cx, cy = float(pt["x"]), float(pt["y"])
        dx, dy = cx - prev_x, cy - prev_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        while acc + dist >= step:
            rem = step - acc
            t = rem / dist
            nx = prev_x + dx * t
            ny = prev_y + dy * t
            out.append((nx, ny))
            prev_x, prev_y = nx, ny
            dx, dy = cx - prev_x, cy - prev_y
            dist = math.hypot(dx, dy)
            acc = 0.0
        acc += dist
        prev_x, prev_y = cx, cy
    last = (float(points[-1]["x"]), float(points[-1]["y"]))
    if math.hypot(last[0] - out[-1][0], last[1] - out[-1][1]) > step * 0.35:
        out.append(last)
    return out


def _rdp_simplify(
    points: List[Tuple[float, float]], epsilon: float = 5.5
) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return points
    x1, y1 = points[0]
    x2, y2 = points[-1]
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.hypot(dx, dy)
    max_dist = -1.0
    max_idx = 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = (
            math.hypot(px - x1, py - y1)
            if seg_len < 1e-6
            else abs(dy * px - dx * py + x2 * y1 - y2 * x1) / seg_len
        )
        if d > max_dist:
            max_dist = d
            max_idx = i
    if max_dist > epsilon:
        left = _rdp_simplify(points[: max_idx + 1], epsilon)
        right = _rdp_simplify(points[max_idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _segments_intersect(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
) -> bool:
    def ccw(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = ccw(a1, a2, b1)
    d2 = ccw(a1, a2, b2)
    d3 = ccw(b1, b2, a1)
    d4 = ccw(b1, b2, a2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _analyze_image_buffer(
    image_data_url: str, bg_hex: str
) -> Optional[Dict[str, Any]]:
    if not image_data_url or "," not in image_data_url:
        return None
    try:
        _, b64_part = image_data_url.split(",", 1)
        raw = base64.b64decode(b64_part)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        target_w = 180
        ratio = target_w / max(1, img.width)
        target_h = max(60, int(round(img.height * ratio)))
        img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    except Exception:
        return None

    bg_r, bg_g, bg_b = _hex_to_rgb(bg_hex)
    pixels = list(img.get_flattened_data()) if hasattr(img, "get_flattened_data") else list(img.getdata())
    w, h = img.size
    total_px = w * h
    ink_count = 0
    grid = [0.0] * 9
    buckets = {
        "warmBright": 0,
        "intenseRed": 0,
        "gentlePinkPurple": 0,
        "freshGreen": 0,
        "coolBlue": 0,
        "darkMuted": 0,
        "neutralGrey": 0,
    }
    sat_sum = 0.0
    lum_sum = 0.0
    sum_x = 0.0
    sum_y = 0.0

    for idx, (r, g, b) in enumerate(pixels):
        if abs(r - bg_r) + abs(g - bg_g) + abs(b - bg_b) < 26:
            continue
        x = idx % w
        y = idx // w
        ink_count += 1
        sum_x += x / max(1, w - 1)
        sum_y += y / max(1, h - 1)
        col = min(2, int((x / w) * 3))
        row = min(2, int((y / h) * 3))
        grid[row * 3 + col] += 1.0
        bucket, sat, lum = classify_rgb_bucket(r, g, b)
        buckets[bucket] += 1
        sat_sum += sat
        lum_sum += lum

    if ink_count == 0:
        return None

    return {
        "inkCoverageRatio": round(ink_count / total_px, 4),
        "grid3x3": [round(c / ink_count, 4) for c in grid],
        "colorHistogram": {k: round(v / ink_count, 4) for k, v in buckets.items()},
        "meanSaturation": round(sat_sum / ink_count, 4),
        "meanLuminance": round(lum_sum / ink_count, 4),
        "pixelCenter": (round(sum_x / ink_count, 4), round(sum_y / ink_count, 4)),
    }


def _detect_composite_figures(
    prims: List[Dict[str, Any]], width: float, height: float
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Assemble multi-stroke spatial topology into complete figures & expressions.

    Returns (recognized_figures, emotion_boosts_dict).
    Each figure includes a bounding box [x, y, w, h] for optional canvas overlay.
    """
    figures: List[Dict[str, Any]] = []
    boosts: Dict[str, float] = {k: 0.0 for k in EMOTION_KEYS}
    found_codes = set()
    used_face_ids = set()
    canvas_area = max(1.0, width * height)

    def add_fig(
        code: str,
        label: str,
        category: str,
        theory: str,
        meaning: str,
        bbox: Tuple[float, float, float, float],
        emo_delta: Dict[str, float],
    ) -> None:
        found_codes.add(code)
        x0, y0, x1, y1 = bbox
        figures.append({
            "code": code,
            "label": label,
            "category": category,
            "theory": theory,
            "meaning": meaning,
            "bbox": {
                "x": round(max(0.0, x0), 1),
                "y": round(max(0.0, y0), 1),
                "w": round(max(8.0, x1 - x0), 1),
                "h": round(max(8.0, y1 - y0), 1),
            },
        })
        for emo, delta in emo_delta.items():
            boosts[emo] = boosts.get(emo, 0.0) + delta

    # Helper: check if primitive `inner` is spatially inside `outer`
    def is_inside(inner: Dict[str, Any], outer: Dict[str, Any], margin: float = 0.10) -> bool:
        mw = outer["w"] * margin
        mh = outer["h"] * margin
        return (
            inner["cx"] >= outer["minX"] - mw
            and inner["cx"] <= outer["maxX"] + mw
            and inner["cy"] >= outer["minY"] - mh
            and inner["cy"] <= outer["maxY"] + mh
            and inner["diag"] < outer["diag"] * 0.92
        )

    # -------------------------------------------------------------------------
    # 1. HEART SYMBOL DETECTION (Double-lobe top cleft + tapered bottom apex)
    # -------------------------------------------------------------------------
    for p in prims:
        if not p["isClosed"] or p["diag"] < 34.0 or p["isScribble"]:
            continue
        aspect = p["w"] / max(1.0, p["h"])
        if not (0.65 <= aspect <= 1.55):
            continue
        pts = p["resamp"]
        if len(pts) < 12:
            continue
        top_band_y = p["minY"] + p["h"] * 0.38
        bot_band_y = p["minY"] + p["h"] * 0.72
        top_left = [pt for pt in pts if pt[1] <= top_band_y and pt[0] < p["cx"] - p["w"] * 0.08]
        top_right = [pt for pt in pts if pt[1] <= top_band_y and pt[0] > p["cx"] + p["w"] * 0.08]
        top_mid = [pt for pt in pts if abs(pt[0] - p["cx"]) <= p["w"] * 0.14 and pt[1] < p["cy"]]
        bot_pts = [pt for pt in pts if pt[1] >= bot_band_y]
        if top_left and top_right and top_mid and bot_pts:
            min_tl_y = min(pt[1] for pt in top_left)
            min_tr_y = min(pt[1] for pt in top_right)
            cleft_y = max(pt[1] for pt in top_mid)
            lobe_top = max(min_tl_y, min_tr_y)
            bot_xs = [pt[0] for pt in bot_pts]
            bot_span = max(bot_xs) - min(bot_xs)
            if (cleft_y - lobe_top) > p["h"] * 0.12 and bot_span < p["w"] * 0.56:
                add_fig(
                    "heart_figure",
                    "愛心符號",
                    "正向心靈符號",
                    "兒童情感依附與正向象徵",
                    "畫面描繪了愛心圖案，直接表達善意、被愛的需求、珍惜或溫暖的情感連結。",
                    (p["minX"], p["minY"], p["maxX"], p["maxY"]),
                    {"joy": 0.68, "calm": -0.06, "anger": -0.14, "sadness": -0.08},
                )
                used_face_ids.add(p["idx"])
                break

    # -------------------------------------------------------------------------
    # 2. SUN DETECTION (Circle + >= 4 outward radiating rays — color-independent!)
    # -------------------------------------------------------------------------
    for p in prims:
        if p["idx"] in used_face_ids or not p["isSmoothLoop"] or p["diag"] < 26.0:
            continue
        r_est = (p["w"] + p["h"]) * 0.25
        rays = []
        ray_angles = set()
        for q in prims:
            if q["idx"] == p["idx"] or not q["isStraight"] or q["len"] < 12.0 or q["len"] > r_est * 3.8:
                continue
            d_start = math.hypot(q["x0"] - p["cx"], q["y0"] - p["cy"])
            d_end = math.hypot(q["x1"] - p["cx"], q["y1"] - p["cy"])
            d_near = min(d_start, d_end)
            d_far = max(d_start, d_end)
            if r_est * 0.55 <= d_near <= r_est * 2.1 and d_far > d_near + 8.0:
                # Check radial alignment
                vx = (q["x1"] - q["x0"]) if d_start < d_end else (q["x0"] - q["x1"])
                vy = (q["y1"] - q["y0"]) if d_start < d_end else (q["y0"] - q["y1"])
                rx = q["cx"] - p["cx"]
                ry = q["cy"] - p["cy"]
                denom = math.hypot(vx, vy) * math.hypot(rx, ry)
                if denom > 1e-4 and (vx * rx + vy * ry) / denom > 0.62:
                    rays.append(q)
                    ang_bin = int((math.degrees(math.atan2(ry, rx)) + 180) // 45)
                    ray_angles.add(ang_bin)
        if len(rays) >= 4 and len(ray_angles) >= 3:
            xs = [p["minX"], p["maxX"]] + [q["minX"] for q in rays] + [q["maxX"] for q in rays]
            ys = [p["minY"], p["maxY"]] + [q["minY"] for q in rays] + [q["maxY"] for q in rays]
            add_fig(
                "sun_figure",
                "太陽與光芒",
                "自然正向象徵",
                "Buck HTP 與 Malchiodi 正向依附象徵",
                "辨識出圓體與周圍放射光線構成的太陽（不限色彩），象徵內心的溫暖、希望感、正向能量或渴望被照亮。",
                (min(xs), min(ys), max(xs), max(ys)),
                {"joy": 0.36, "calm": 0.14, "sadness": -0.14},
            )
            used_face_ids.add(p["idx"])
            break

    # -------------------------------------------------------------------------
    # 3. FACE & FACIAL EXPRESSION RECOGNITION (Smile / Frown / Tears / Angry Brows / Shaded Face)
    # -------------------------------------------------------------------------
    face_candidates = [
        p
        for p in prims
        if p["idx"] not in used_face_ids
        and p["isClosed"]
        and not p["isScribble"]
        and p["diag"] >= 38.0
        and 0.58 <= (p["w"] / max(1.0, p["h"])) <= 1.65
    ]
    # Sort larger faces first
    face_candidates.sort(key=lambda x: x["diag"], reverse=True)

    detected_faces: List[Dict[str, Any]] = []
    for face in face_candidates:
        if any(is_inside(face, existing, margin=0.02) for existing in detected_faces):
            continue
        inner = [q for q in prims if q["idx"] != face["idx"] and is_inside(q, face, margin=0.06)]
        if not inner:
            continue

        # Check heavy face blackout/shading (Koppitz Face Shading)
        inner_scribbles = [q for q in inner if q["isScribble"] or q["len"] > face["diag"] * 2.2]
        inner_total_len = sum(q["len"] for q in inner)
        if inner_scribbles and inner_total_len > face["diag"] * 3.0:
            add_fig(
                "face_shaded",
                "被塗抹遮蓋的臉孔／圓體",
                "Koppitz 情緒指標",
                "Koppitz (1968) 臉部塗黑陰影指標（Shading of Face）",
                "圓形臉孔內部出現密集塗黑或反覆遮蓋，在兒童人物畫中常對應強烈的焦慮、自我否定、羞愧或想把真實情緒藏起來。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"anxiety": 0.38, "sadness": 0.20, "anger": 0.16, "joy": -0.25},
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
            continue

        # Classify facial components inside the head circle
        eyes_left = 0
        eyes_right = 0
        angry_left_brow = False
        angry_right_brow = False
        has_smile = False
        has_frown = False
        has_jagged_mouth = False
        has_tense_flat_mouth = False
        tear_count = 0

        for q in inner:
            rel_x = (q["cx"] - face["minX"]) / max(1.0, face["w"])
            rel_y = (q["cy"] - face["minY"]) / max(1.0, face["h"])

            # Upper-half features: eyes or angry furrowed brows
            if 0.10 <= rel_y <= 0.58:
                # Check angry inward-sloping eyebrow (\ on left, / on right)
                if q["isStraight"] and q["w"] >= face["w"] * 0.08 and q["w"] <= face["w"] * 0.45:
                    dx = q["x1"] - q["x0"]
                    dy = q["y1"] - q["y0"]
                    if abs(dx) > 4.0:
                        slope = dy / dx
                        if rel_x < 0.48 and slope > 0.22:
                            angry_left_brow = True
                        elif rel_x > 0.52 and slope < -0.22:
                            angry_right_brow = True

                # Small eye dot / tiny circle / small mark
                if q["diag"] <= face["diag"] * 0.24:
                    if rel_x < 0.48:
                        eyes_left += 1
                    elif rel_x > 0.52:
                        eyes_right += 1

            # Tears: vertical short strokes or droplets below eye level on left/right cheek zones
            if 0.42 <= rel_y <= 0.92 and (rel_x < 0.44 or rel_x > 0.56):
                is_vert_drop = (
                    (q["h"] >= max(6.0, q["w"] * 1.25) and q["h"] <= face["h"] * 0.42)
                    or (q["diag"] <= face["diag"] * 0.11 and rel_y >= 0.48)
                )
                if is_vert_drop:
                    tear_count += 1

            # Lower-half mouth zone (centered horizontally)
            if 0.48 <= rel_y <= 0.92 and 0.22 <= rel_x <= 0.78 and q["w"] >= face["w"] * 0.14:
                if q["sharpTurns"] >= 2 and q["sharpRatio"] > 0.35:
                    has_jagged_mouth = True
                elif q["arcSagRatio"] >= 0.13:
                    # +y is downward -> center dips below endpoints -> SMILE U-arc!
                    has_smile = True
                elif q["arcSagRatio"] <= -0.13:
                    # center arches above endpoints -> FROWN inverted-U arc!
                    has_frown = True
                elif q["isStraight"] and q["w"] > q["h"] * 2.2:
                    has_tense_flat_mouth = True

        # Check tears slightly BELOW the face boundary too (falling tears off the chin/cheeks)
        for q in prims:
            if q["idx"] == face["idx"] or q in inner:
                continue
            if (
                face["minX"] <= q["cx"] <= face["maxX"]
                and face["maxY"] - face["h"] * 0.1 <= q["minY"] <= face["maxY"] + face["h"] * 0.35
                and q["h"] > q["w"] * 1.4
                and 8.0 <= q["h"] <= face["h"] * 0.45
                and abs(q["cx"] - face["cx"]) > face["w"] * 0.10
            ):
                tear_count += 1

        has_eyes = (eyes_left >= 1 and eyes_right >= 1) or (eyes_left + eyes_right >= 2)
        angry_brows = angry_left_brow and angry_right_brow

        if angry_brows or (has_jagged_mouth and (angry_left_brow or angry_right_brow or has_eyes)):
            add_fig(
                "face_angry",
                "生氣皺眉臉孔",
                "臉部表情辨識",
                "Koppitz HFD 表情與攻擊張力特徵",
                "辨識出圓形臉孔內帶有向內斜壓的八字眉或緊繃鋸齒嘴型，直接呈現憤怒、不滿、委屈抗議或內在高度防衛情緒。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"anger": 0.56, "anxiety": 0.16, "joy": -0.32, "calm": -0.24},
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
        elif (has_frown and tear_count >= 1) or (has_eyes and tear_count >= 2 and not has_smile):
            add_fig(
                "face_sad_crying",
                "哭泣落淚臉孔",
                "臉部表情辨識",
                "Koppitz HFD 與 Malchiodi 悲傷表情指標",
                "辨識出下垂嘴角與眼下淚痕／淚滴，明確指向傷心、難過、委屈想哭或亟需被溫柔安慰的心理狀態。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"sadness": 0.72, "loneliness": 0.12, "joy": -0.36, "calm": -0.26},
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
        elif has_frown and (has_eyes or len(inner) >= 2):
            add_fig(
                "face_sad",
                "嘴角下垂的難過臉孔",
                "臉部表情辨識",
                "Koppitz HFD 與兒童表情投射",
                "辨識出臉孔下半部呈現明顯下垂的倒弧線嘴型，透露失落、不開心、沮喪或提不起勁的內在心情。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"sadness": 0.48, "loneliness": 0.14, "joy": -0.30, "calm": -0.20},
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
        elif has_smile:
            add_fig(
                "face_happy",
                "微笑開朗臉孔",
                "臉部表情辨識",
                "Malchiodi 正向臉部情感投射",
                "辨識出臉孔內帶有明顯上揚的微笑弧線，顯示當下心情正向開朗、保有希望感或願意友善互動（不受單色畫筆影響）。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"joy": 0.54, "calm": 0.16, "sadness": -0.28, "anger": -0.24},
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
        elif has_eyes and has_tense_flat_mouth:
            add_fig(
                "face_tense_neutral",
                "抿嘴緊繃／沉默臉孔",
                "臉部表情辨識",
                "Koppitz HFD 情緒壓抑特徵",
                "臉孔呈現平直緊閉的嘴型，可能反映壓抑想法、無奈、緊張或一時不知如何說出口的狀態。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"anxiety": 0.24, "sadness": 0.16, "loneliness": 0.14},
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])

    # -------------------------------------------------------------------------
    # 4. HTP HOUSE DETECTION (Rectangular wall + roof above + door/window check)
    # -------------------------------------------------------------------------
    # Support both 1-stroke closed box OR 4-line assembled box
    box_candidates: List[Dict[str, Any]] = []
    for p in prims:
        if p["idx"] in used_face_ids:
            continue
        if p["isClosed"] and p["sharpTurns"] >= 3 and p["w"] >= 55.0 and p["h"] >= 45.0:
            box_candidates.append(p)

    # Also detect 4-line assembled rectangles (2 horizontal + 2 vertical lines forming a box)
    h_lines = [p for p in prims if p["isStraight"] and p["w"] > p["h"] * 2.5 and p["w"] >= 50.0]
    v_lines = [p for p in prims if p["isStraight"] and p["h"] > p["w"] * 2.5 and p["h"] >= 40.0]
    if not box_candidates and len(h_lines) >= 2 and len(v_lines) >= 2:
        h_sorted = sorted(h_lines, key=lambda x: x["cy"])
        v_sorted = sorted(v_lines, key=lambda x: x["cx"])
        top_l, bot_l = h_sorted[0], h_sorted[-1]
        left_l, right_l = v_sorted[0], v_sorted[-1]
        if (
            bot_l["cy"] - top_l["cy"] >= 45.0
            and right_l["cx"] - left_l["cx"] >= 50.0
            and abs(top_l["minX"] - left_l["cx"]) < 38.0
            and abs(top_l["maxX"] - right_l["cx"]) < 38.0
            and abs(bot_l["minX"] - left_l["cx"]) < 38.0
            and abs(bot_l["maxX"] - right_l["cx"]) < 38.0
        ):
            bw = right_l["cx"] - left_l["cx"]
            bh = bot_l["cy"] - top_l["cy"]
            box_candidates.append({
                "idx": -100,
                "minX": left_l["cx"],
                "maxX": right_l["cx"],
                "minY": top_l["cy"],
                "maxY": bot_l["cy"],
                "cx": (left_l["cx"] + right_l["cx"]) * 0.5,
                "cy": (top_l["cy"] + bot_l["cy"]) * 0.5,
                "w": bw,
                "h": bh,
                "diag": math.hypot(bw, bh),
                "component_ids": {top_l["idx"], bot_l["idx"], left_l["idx"], right_l["idx"]},
            })

    for box in box_candidates:
        comp_ids = box.get("component_ids") or {box["idx"]}
        # Look for a roof above the box top edge (sloped strokes or triangle/trapezoid above minY)
        roof_strokes = [
            q
            for q in prims
            if q["idx"] not in comp_ids
            and q["maxY"] <= box["minY"] + box["h"] * 0.28
            and q["minY"] < box["minY"] - 14.0
            and q["maxX"] >= box["minX"] - 30.0
            and q["minX"] <= box["maxX"] + 30.0
        ]
        if roof_strokes:
            roof_ids = {q["idx"] for q in roof_strokes}
            # Check for doors / windows inside the wall box
            inner_house = [
                q
                for q in prims
                if q["idx"] not in comp_ids
                and q["idx"] not in roof_ids
                and is_inside(q, box, margin=-0.02)
            ]
            inner_scribbles = [q for q in inner_house if q["isScribble"]]
            xs = [box["minX"], box["maxX"]] + [q["minX"] for q in roof_strokes] + [q["maxX"] for q in roof_strokes]
            ys = [box["minY"], box["maxY"]] + [q["minY"] for q in roof_strokes] + [q["maxY"] for q in roof_strokes]
            house_bbox = (min(xs), min(ys), max(xs), max(ys))
            if inner_house and not inner_scribbles:
                add_fig(
                    "house_open",
                    "有門窗的完整房屋（HTP）",
                    "屋樹人 HTP 結構",
                    "Buck (1948) 屋樹人測驗：門窗象徵對外溝通與安全感",
                    "辨識出具備屋頂、牆體及內部門窗的房屋結構，象徵家庭或內在安全基地穩固，對外界維持開放與聯繫意願。",
                    house_bbox,
                    {"calm": 0.36, "joy": 0.18, "loneliness": -0.14},
                )
            else:
                add_fig(
                    "house_closed",
                    "無門窗／緊閉的房屋（HTP）",
                    "屋樹人 HTP 結構",
                    "Buck (1948) 屋樹人測驗：省略門窗反映防衛與孤立感",
                    "辨識出房屋外框但內部缺少門窗（或充滿塗抹），常反映較高的心理防衛、緊繃、害怕外界干擾或覺得難以被人走近。",
                    house_bbox,
                    {"anxiety": 0.28, "loneliness": 0.26, "calm": -0.14},
                )
            break

    # -------------------------------------------------------------------------
    # 5. CLOUD + FALLING RAIN COMPOSITE (Color-independent!)
    # -------------------------------------------------------------------------
    down_lines = [p for p in prims if p["isDownward"] and p["h"] >= 28.0]
    if len(down_lines) >= 3:
        min_dy = min(p["minY"] for p in down_lines)
        min_dx = min(p["minX"] for p in down_lines)
        max_dx = max(p["maxX"] for p in down_lines)
        max_dy = max(p["maxY"] for p in down_lines)
        upper_cloud = [
            p
            for p in prims
            if not p["isDownward"]
            and p["idx"] not in used_face_ids
            and p["w"] >= 70.0
            and p["cy"] <= min_dy + 35.0
            and p["maxX"] >= min_dx - 30.0
            and p["minX"] <= max_dx + 30.0
        ]
        if upper_cloud:
            xs = [min_dx, max_dx] + [c["minX"] for c in upper_cloud] + [c["maxX"] for c in upper_cloud]
            ys = [min_dy, max_dy] + [c["minY"] for c in upper_cloud] + [c["maxY"] for c in upper_cloud]
            add_fig(
                "rain_cloud_figure",
                "烏雲與落雨景象",
                "天氣情境象徵",
                "Malchiodi (1998) 天氣隱喻與悲傷投射",
                "辨識出上方雲團結合正下方多道雨絲（不論色彩），在兒童畫中常隱喻低落、委屈、心事沉重或想哭釋放情緒。",
                (min(xs), min(ys), max(xs), max(ys)),
                {"sadness": 0.42, "loneliness": 0.14, "joy": -0.22, "calm": -0.20},
            )

    # -------------------------------------------------------------------------
    # 6. CROSS-OUT SYMBOL (✕) — Two opposing crossing diagonals
    # -------------------------------------------------------------------------
    diag_pos = [
        p
        for p in prims
        if p["isStraight"]
        and p["len"] >= 48.0
        and abs(p["x1"] - p["x0"]) >= 20.0
        and abs(p["y1"] - p["y0"]) >= 20.0
        and (p["x1"] - p["x0"]) * (p["y1"] - p["y0"]) > 0
    ]
    diag_neg = [
        p
        for p in prims
        if p["isStraight"]
        and p["len"] >= 48.0
        and abs(p["x1"] - p["x0"]) >= 20.0
        and abs(p["y1"] - p["y0"]) >= 20.0
        and (p["x1"] - p["x0"]) * (p["y1"] - p["y0"]) < 0
    ]
    cross_found = False
    for a in diag_pos:
        for b in diag_neg:
            if _segments_intersect((a["x0"], a["y0"]), (a["x1"], a["y1"]), (b["x0"], b["y0"]), (b["x1"], b["y1"])):
                dist_c = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
                if dist_c < max(a["len"], b["len"]) * 0.42:
                    xb = (min(a["minX"], b["minX"]), min(a["minY"], b["minY"]), max(a["maxX"], b["maxX"]), max(a["maxY"], b["maxY"]))
                    add_fig(
                        "cross_out_figure",
                        "打叉否定標記（✕）",
                        "衝突與否定符號",
                        "藝術治療塗銷與拒絕指標",
                        "辨識出交叉的大打叉（✕），常代表強烈抗拒、對某件事感到氣憤挫折、自我否定或想把煩心的事情劃掉。",
                        xb,
                        {"anger": 0.40, "anxiety": 0.22, "sadness": 0.12, "joy": -0.24, "calm": -0.22},
                    )
                    cross_found = True
                    break
        if cross_found:
            break

    # -------------------------------------------------------------------------
    # 7. CAGE / PARALLEL BARS (Vertical or horizontal enclosing bars >= 4)
    # -------------------------------------------------------------------------
    vert_bars = [p for p in prims if p["isStraight"] and p["h"] > p["w"] * 3.0 and p["h"] >= 65.0]
    if len(vert_bars) >= 4 and "rain_cloud_figure" not in found_codes:
        # Check if vertical bars align closely in top/bottom Y (like cage/prison bars rather than scattered rain)
        ys_top = [b["minY"] for b in vert_bars]
        ys_bot = [b["maxY"] for b in vert_bars]
        top_spread = max(ys_top) - min(ys_top)
        bot_spread = max(ys_bot) - min(ys_bot)
        has_horiz_frame = any(
            p["isStraight"] and p["w"] > p["h"] * 2.5 and p["w"] >= 70.0 for p in prims
        )
        if has_horiz_frame and top_spread < 45.0 and bot_spread < 45.0:
            xb = (min(b["minX"] for b in vert_bars), min(ys_top), max(b["maxX"] for b in vert_bars), max(ys_bot))
            add_fig(
                "cage_bars_figure",
                "柵欄／受困格線結構",
                "空間界線與受困指標",
                "Buck HTP 與藝術治療空間束縛象徵",
                "辨識出整齊排列的柵欄或格柵框線，常呼應內在感到受限、壓力沉重、需要自我保護或覺得被困住的心情。",
                xb,
                {"anxiety": 0.36, "loneliness": 0.24, "calm": -0.20, "joy": -0.18},
            )

    # -------------------------------------------------------------------------
    # 8. HUMAN STICK/CONTOUR PERSON & ISOLATED CORNER FIGURE
    # -------------------------------------------------------------------------
    head_candidates = [
        p
        for p in prims
        if p["isClosed"]
        and not p["isScribble"]
        and 16.0 <= p["diag"] <= 190.0
        and 0.6 <= (p["w"] / max(1.0, p["h"])) <= 1.5
    ]
    for head in head_candidates:
        # Look for vertical torso below head
        torso = [
            q
            for q in prims
            if q["idx"] != head["idx"]
            and abs(q["cx"] - head["cx"]) <= head["w"] * 0.65
            and q["minY"] >= head["cy"]
            and q["minY"] <= head["maxY"] + head["h"] * 0.6
            and q["maxY"] > head["maxY"] + head["h"] * 0.4
        ]
        if torso:
            parts = [head] + torso
            # Gather nearby limbs around torso
            t_min_y = min(t["minY"] for t in torso)
            t_max_y = max(t["maxY"] for t in torso)
            for q in prims:
                if q in parts:
                    continue
                if (
                    abs(q["cx"] - head["cx"]) <= head["w"] * 2.4
                    and t_min_y - head["h"] * 0.4 <= q["cy"] <= t_max_y + head["h"] * 1.4
                    and q["diag"] <= head["diag"] * 2.8
                ):
                    parts.append(q)
            px0 = min(x["minX"] for x in parts)
            py0 = min(x["minY"] for x in parts)
            px1 = max(x["maxX"] for x in parts)
            py1 = max(x["maxY"] for x in parts)
            p_area_ratio = ((px1 - px0) * (py1 - py0)) / canvas_area
            pcx = ((px0 + px1) * 0.5) / width
            pcy = ((py0 + py1) * 0.5) / height
            p_center_dist = math.hypot(pcx - 0.5, pcy - 0.5) / 0.7071

            if p_area_ratio < 0.11 and p_center_dist > 0.30:
                add_fig(
                    "person_tiny_isolated",
                    "角落中的微小人物",
                    "Koppitz 人物畫指標",
                    "Koppitz (1968) 微小人物指標（Tiny Figure）",
                    "辨識出縮在畫紙角落的微小人物，周圍伴隨大片留白，在兒童人物畫中常反映孤單感、害羞退縮或希望被看見。",
                    (px0, py0, px1, py1),
                    {"loneliness": 0.46, "sadness": 0.18, "joy": -0.18},
                )
            elif "face_sad" not in found_codes and "face_sad_crying" not in found_codes and "face_angry" not in found_codes:
                # Check if head has zero internal strokes (blank omitted face on a larger figure)
                head_inner = [q for q in prims if q["idx"] != head["idx"] and is_inside(q, head, margin=0.05)]
                if len(head_inner) == 0 and head["diag"] >= 38.0:
                    add_fig(
                        "person_blank_face",
                        "五官留白的人物輪廓",
                        "Koppitz 人物畫指標",
                        "Koppitz (1968) 五官省略指標",
                        "人物頭部保留空白未畫五官，可能呈現逃避對視、不知如何表達心情、防備心較高或單純勾勒遠景輪廓。",
                        (px0, py0, px1, py1),
                        {"loneliness": 0.22, "anxiety": 0.16},
                    )
                else:
                    add_fig(
                        "person_figure",
                        "人物輪廓",
                        "Koppitz 人物畫結構",
                        "Koppitz HFD 與 Buck HTP 人物投射",
                        "辨識出頭身完整的人物圖形，反映創作者以自我或重要他人作為畫面的情感聚焦中心。",
                        (px0, py0, px1, py1),
                        {"calm": 0.10, "joy": 0.08},
                    )
            break

    return figures, boosts


def analyze_drawing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run v2.0 Shape-First child psychology analysis on drawing telemetry."""
    canvas_info = payload.get("canvas") or {}
    width = float(canvas_info.get("width") or 800.0)
    height = float(canvas_info.get("height") or 550.0)
    bg_hex = str(canvas_info.get("backgroundColor") or "#FDFBF7")
    canvas_area = max(1.0, width * height)

    all_strokes: List[Dict[str, Any]] = payload.get("strokes") or []
    draw_strokes = [
        s for s in all_strokes if s.get("tool") != "eraser" and len(s.get("points") or []) >= 1
    ]
    eraser_strokes = [s for s in all_strokes if s.get("tool") == "eraser"]

    process = payload.get("processMetrics") or {}
    undo_count = int(process.get("undoCount") or 0)
    eraser_count = max(int(process.get("eraserStrokeCount") or 0), len(eraser_strokes))
    total_duration_ms = float(process.get("totalDurationMs") or 0.0)
    pause_count = int(process.get("pauseCount") or 0)

    user_context = payload.get("userContext") or {}
    title_or_note = str(user_context.get("titleOrNote") or "").strip()
    selected_motifs: List[str] = list(user_context.get("selectedMotifs") or [])

    if not draw_strokes:
        return {
            "status": "empty",
            "message": "畫布目前還是空白的，請自由畫上幾筆線條或圖案後再進行心理探索分析。",
        }

    # --- 1. Extract Per-Stroke Primitives, Kinematics, Color & Grid ---
    total_drawn_ink_area = 0.0
    velocities: List[float] = []
    pressures: List[float] = []
    widths: List[float] = []
    sharp_turns = 0
    smooth_turns = 0
    total_turns = 0
    downward_strokes = 0
    upward_strokes = 0
    closed_loops = 0
    smooth_loops = 0
    angular_loops = 0
    edge_touch_strokes = 0
    short_fragment_strokes = 0
    scribble_strokes = 0

    min_x, max_x = width, 0.0
    min_y, max_y = height, 0.0
    weighted_x_sum = 0.0
    weighted_y_sum = 0.0
    total_weight = 0.0

    vec_grid = [0.0] * 9
    cell_occupancy: Dict[Tuple[int, int], int] = {}
    vec_buckets = {
        "warmBright": 0.0,
        "intenseRed": 0.0,
        "gentlePinkPurple": 0.0,
        "freshGreen": 0.0,
        "coolBlue": 0.0,
        "darkMuted": 0.0,
        "neutralGrey": 0.0,
    }
    color_ink_by_hex: Dict[str, float] = {}
    sat_weighted = 0.0
    lum_weighted = 0.0
    radiating_candidates: List[Tuple[float, float, float, float, str]] = []
    stroke_prims: List[Dict[str, Any]] = []

    for s_idx, stroke in enumerate(draw_strokes):
        pts = stroke.get("points") or []
        color_hex = str(stroke.get("color") or "#2A2B30").lower()
        tool = str(stroke.get("tool") or "crayon")
        size = float(stroke.get("size") or 7.0)
        eff_width = size * {"pencil": 0.65, "crayon": 1.15, "marker": 1.3, "watercolor": 1.65}.get(tool, 1.0)
        widths.append(eff_width)

        r, g, b = _hex_to_rgb(color_hex)
        bucket, sat, lum = classify_rgb_bucket(r, g, b)

        s_len = 0.0
        s_min_x, s_max_x = width, 0.0
        s_min_y, s_max_y = height, 0.0
        touches_edge = False

        for idx, pt in enumerate(pts):
            x = float(pt.get("x", 0.0))
            y = float(pt.get("y", 0.0))
            pressures.append(_clamp(float(pt.get("p", 0.5)), 0.05, 1.0))
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            s_min_x, s_max_x = min(s_min_x, x), max(s_max_x, x)
            s_min_y, s_max_y = min(s_min_y, y), max(s_max_y, y)
            if x < width * 0.055 or x > width * 0.945 or y < height * 0.055 or y > height * 0.945:
                touches_edge = True
            if idx > 0:
                px = float(pts[idx - 1].get("x", x))
                py = float(pts[idx - 1].get("y", y))
                dt = max(4.0, float(pt.get("t", 0.0)) - float(pts[idx - 1].get("t", 0.0)))
                seg_d = math.hypot(x - px, y - py)
                s_len += seg_d
                if seg_d > 0.5 and dt < 1500.0:
                    velocities.append(min(4.5, seg_d / dt))

        if s_len < 1.0:
            s_len = eff_width * 1.2

        ink_area = s_len * max(2.0, eff_width)
        total_drawn_ink_area += ink_area
        color_ink_by_hex[color_hex] = color_ink_by_hex.get(color_hex, 0.0) + ink_area
        if touches_edge:
            edge_touch_strokes += 1
        if s_len < width * 0.065:
            short_fragment_strokes += 1

        resamp = _resample_points(pts, step=10.0)
        pt_weight = max(1.0, eff_width) * (s_len / max(1, len(resamp)))
        for rx, ry in resamp:
            col = min(2, max(0, int((rx / width) * 3)))
            row = min(2, max(0, int((ry / height) * 3)))
            vec_grid[row * 3 + col] += pt_weight
            weighted_x_sum += (rx / width) * pt_weight
            weighted_y_sum += (ry / height) * pt_weight
            total_weight += pt_weight
            gx, gy = int(rx / 14.0), int(ry / 14.0)
            cell_occupancy[(gx, gy)] = cell_occupancy.get((gx, gy), 0) + 1

        vec_buckets[bucket] += ink_area
        sat_weighted += sat * ink_area
        lum_weighted += lum * ink_area

        sw = max(1.0, s_max_x - s_min_x)
        sh = max(1.0, s_max_y - s_min_y)
        stroke_bbox_diag = math.hypot(sw, sh)
        rdp_eps = min(5.5, max(1.6, stroke_bbox_diag * 0.055))
        simp = _rdp_simplify(resamp, epsilon=rdp_eps)

        s_sharp = 0
        s_smooth = 0
        s_turns = 0
        for i in range(1, len(simp) - 1):
            v1x, v1y = simp[i][0] - simp[i - 1][0], simp[i][1] - simp[i - 1][1]
            v2x, v2y = simp[i + 1][0] - simp[i][0], simp[i + 1][1] - simp[i][1]
            m1, m2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
            if m1 < 1e-4 or m2 < 1e-4:
                continue
            cos_a = _clamp((v1x * v2x + v1y * v2y) / (m1 * m2), -1.0, 1.0)
            angle_deg = math.degrees(math.acos(cos_a))
            if angle_deg < 6.0:
                continue
            s_turns += 1
            total_turns += 1
            if angle_deg >= 62.0:
                s_sharp += 1
                sharp_turns += 1
            elif 8.0 <= angle_deg <= 54.0:
                s_smooth += 1
                smooth_turns += 1

        start_pt, end_pt = pts[0], pts[-1]
        x0, y0 = float(start_pt["x"]), float(start_pt["y"])
        x1, y1 = float(end_pt["x"]), float(end_pt["y"])
        dx_total, dy_total = x1 - x0, y1 - y0
        chord = math.hypot(dx_total, dy_total)
        straightness = chord / max(1.0, s_len)

        is_straight = s_len > 10.0 and straightness > 0.74
        is_downward = is_straight and dy_total > abs(dx_total) * 1.15 and dy_total > 36.0
        is_upward = is_straight and -dy_total > abs(dx_total) * 1.15 and -dy_total > 36.0
        is_closed = (
            stroke_bbox_diag > 16.0
            and chord < stroke_bbox_diag * 0.45
            and s_len > stroke_bbox_diag * 1.85
        )
        sharp_ratio = s_sharp / max(1, s_turns)
        is_smooth_loop = is_closed and (s_turns == 0 or sharp_ratio < 0.28)
        is_scribble = s_turns >= 4 and sharp_ratio > 0.35 and s_len > stroke_bbox_diag * 2.2

        if is_downward:
            downward_strokes += 1
        elif is_upward:
            upward_strokes += 1
        if is_straight and s_len > 18.0:
            radiating_candidates.append((x0, y0, x1, y1, bucket))
        if is_closed and stroke_bbox_diag > 24.0:
            closed_loops += 1
            if is_smooth_loop:
                smooth_loops += 1
            elif s_sharp >= 2:
                angular_loops += 1
        if is_scribble:
            scribble_strokes += 1

        # Compute open horizontal arc vertical sag ratio (+ = smile U-curve, - = frown ∩-curve)
        arc_sag_ratio = 0.0
        if not is_closed and len(resamp) >= 3 and sw >= sh * 0.75 and abs(x1 - x0) >= sw * 0.45:
            y_chord_mid = (y0 + y1) * 0.5
            inner_slice = resamp[1:-1]
            y_inner_mean = sum(p[1] for p in inner_slice) / max(1, len(inner_slice))
            arc_sag_ratio = (y_inner_mean - y_chord_mid) / max(1.0, sw)

        stroke_prims.append({
            "idx": s_idx,
            "minX": s_min_x,
            "maxX": s_max_x,
            "minY": s_min_y,
            "maxY": s_max_y,
            "cx": (s_min_x + s_max_x) * 0.5,
            "cy": (s_min_y + s_max_y) * 0.5,
            "w": sw,
            "h": sh,
            "diag": stroke_bbox_diag,
            "len": s_len,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "bucket": bucket,
            "isStraight": is_straight,
            "isDownward": is_downward,
            "isClosed": is_closed,
            "isSmoothLoop": is_smooth_loop,
            "isScribble": is_scribble,
            "sharpTurns": s_sharp,
            "sharpRatio": sharp_ratio,
            "arcSagRatio": arc_sag_ratio,
            "resamp": resamp,
        })

    # --- 2. Combine Vector Telemetry with Pixel Stats & Detect Monochrome Sketch Mode ---
    pixel_stats = _analyze_image_buffer(str(payload.get("imageDataUrl") or ""), bg_hex)
    if not pixel_stats and payload.get("clientImageStats"):
        pixel_stats = payload.get("clientImageStats")

    vec_total_area = max(1.0, sum(vec_buckets.values()))
    vec_hist = {k: round(v / vec_total_area, 4) for k, v in vec_buckets.items()}
    vec_grid_sum = max(1e-6, sum(vec_grid))
    norm_vec_grid = [round(v / vec_grid_sum, 4) for v in vec_grid]

    if pixel_stats and pixel_stats.get("inkCoverageRatio", 0.0) > 0.0005:
        color_hist = pixel_stats["colorHistogram"]
        grid_3x3 = pixel_stats["grid3x3"]
        ink_coverage = float(pixel_stats["inkCoverageRatio"])
        mean_sat = float(pixel_stats["meanSaturation"])
        mean_lum = float(pixel_stats["meanLuminance"])
        cx, cy = pixel_stats.get(
            "pixelCenter",
            (weighted_x_sum / max(1e-6, total_weight), weighted_y_sum / max(1e-6, total_weight)),
        )
    else:
        color_hist = vec_hist
        grid_3x3 = norm_vec_grid
        ink_coverage = _clamp((len(cell_occupancy) * 196.0) / canvas_area, 0.002, 0.95)
        mean_sat = sat_weighted / vec_total_area
        mean_lum = lum_weighted / vec_total_area
        cx = weighted_x_sum / max(1e-6, total_weight)
        cy = weighted_y_sum / max(1e-6, total_weight)

    # Monochrome / Single-Pencil Debiasing check
    meaningful_colors = [
        hx for hx, area in color_ink_by_hex.items() if area / max(1.0, total_drawn_ink_area) >= 0.05
    ]
    is_monochrome_sketch = len(meaningful_colors) <= 1
    # Color factor: 0.20 in monochrome pencil mode, 0.52 in multi-color mode (Shape-First!)
    color_scale = 0.20 if is_monochrome_sketch else 0.52

    # --- 3. Composite Complete-Figure & Expression Recognition (Shape-First!) ---
    recognized_figures, fig_boosts = _detect_composite_figures(stroke_prims, width, height)
    fig_codes = {f["code"] for f in recognized_figures}

    # Kinematic & Spatial Indices
    covered_px_est = max(150.0, ink_coverage * canvas_area)
    raw_overlap_ratio = total_drawn_ink_area / covered_px_est
    multi_hit_cells = sum(1 for c in cell_occupancy.values() if c >= 3)
    cell_overlap_ratio = multi_hit_cells / max(1, len(cell_occupancy))
    overlap_index = _clamp((raw_overlap_ratio - 1.0) / 3.2 * 0.65 + cell_overlap_ratio * 0.55)

    sharpness_index = _clamp((sharp_turns / max(1, total_turns)) * 1.15)
    smoothness_index = _clamp((smooth_turns / max(1, total_turns)) * 1.15)
    mean_vel = sum(velocities) / max(1, len(velocities)) if velocities else 0.55
    velocity_index = _clamp(mean_vel / 1.8)
    mean_width = sum(widths) / max(1, len(widths))
    mean_pressure = sum(pressures) / max(1, len(pressures))
    weight_index = _clamp((mean_width / 22.0) * 0.65 + mean_pressure * 0.35)
    fragmentation_index = _clamp(
        (short_fragment_strokes / max(1, len(draw_strokes))) * 0.75 + _clamp(pause_count / 6.0) * 0.25
    )

    revision_events = undo_count + eraser_count
    revision_ratio = revision_events / max(1.0, len(draw_strokes) + revision_events)
    revision_index = _clamp(revision_ratio * 1.8 + _clamp(revision_events / 8.0) * 0.45)

    bbox_w_ratio = _clamp((max_x - min_x) / width)
    bbox_h_ratio = _clamp((max_y - min_y) / height)
    bbox_area_ratio = _clamp(bbox_w_ratio * bbox_h_ratio)
    center_dist = math.hypot(cx - 0.5, cy - 0.5) / 0.7071
    edge_touch_ratio = edge_touch_strokes / max(1, len(draw_strokes))

    warm_bright = color_hist.get("warmBright", 0.0)
    intense_red = color_hist.get("intenseRed", 0.0)
    pink_purple = color_hist.get("gentlePinkPurple", 0.0)
    fresh_green = color_hist.get("freshGreen", 0.0)
    cool_blue = color_hist.get("coolBlue", 0.0)
    dark_muted = color_hist.get("darkMuted", 0.0)
    neutral_grey = color_hist.get("neutralGrey", 0.0)

    positive_warm_palette = warm_bright + fresh_green * 0.75 + pink_purple * 0.65
    dark_heavy_palette = dark_muted + neutral_grey * 0.5
    active_buckets = sum(1 for v in color_hist.values() if v >= 0.06)

    # --- 4. Combine Recognized Complete Figures + Secondary Geometric Motifs ---
    detected_motifs: List[Dict[str, str]] = [
        {
            "code": f["code"],
            "label": f["label"],
            "theory": f["theory"],
            "meaning": f["meaning"],
        }
        for f in recognized_figures
    ]
    motif_codes = set(fig_codes)

    if scribble_strokes >= 2 or (overlap_index > 0.48 and sharpness_index > 0.35):
        motif_codes.add("dense_scribble_knot")
        detected_motifs.append({
            "code": "dense_scribble_knot",
            "label": "密集交織／情緒塗抹筆觸",
            "theory": "Koppitz (1968) 重度陰影與反覆塗抹指標",
            "meaning": "重複來回交疊的強烈筆觸，常代表內心積聚張力、煩躁、焦慮或試圖覆蓋某種不安感受。",
        })

    if "rain_cloud_figure" not in motif_codes and (
        downward_strokes >= 4 or (downward_strokes >= 3 and (cool_blue + dark_muted + neutral_grey) > 0.45)
    ):
        motif_codes.add("downward_streaks")
        detected_motifs.append({
            "code": "downward_streaks",
            "label": "向下垂落／雨絲般的線條",
            "theory": "Malchiodi (1998) 動態線條方向象徵",
            "meaning": "多道朝下延展的線條宛如雨滴或垂落感，常見於情緒低落、委屈、悲傷或內在沉重時。",
        })

    radiating_hub_warm = "sun_figure" in motif_codes
    radiating_hub_tense = False
    if "sun_figure" not in motif_codes and len(radiating_candidates) >= 5:
        angles = {int((math.degrees(math.atan2(ey - sy, ex - sx)) + 180) // 45) for sx, sy, ex, ey, _ in radiating_candidates}
        warm_rad = sum(1 for *_, bkt in radiating_candidates if bkt in ("warmBright", "gentlePinkPurple"))
        tense_rad = sum(1 for *_, bkt in radiating_candidates if bkt in ("intenseRed", "darkMuted"))
        if len(angles) >= 4:
            if warm_rad >= 3:
                radiating_hub_warm = True
                motif_codes.add("radiating_warm_burst")
                detected_motifs.append({
                    "code": "radiating_warm_burst",
                    "label": "向外綻放的暖色光芒",
                    "theory": "Buck HTP 與兒童正向象徵（太陽／光線）",
                    "meaning": "以暖色朝多方向開展的放射筆觸，常呼應對溫暖、希望、活力與被關愛的感受。",
                })
            elif tense_rad >= 3:
                radiating_hub_tense = True
                motif_codes.add("radiating_tense_burst")
                detected_motifs.append({
                    "code": "radiating_tense_burst",
                    "label": "尖銳放射張力線條",
                    "theory": "Alschuler & Hattwick (1947) 高喚起衝動線條",
                    "meaning": "強烈的紅黑放射或急劃線條，呈現內在能量爆發、憤怒或渴望掙脫束縛的張力。",
                })

    if "house_open" not in motif_codes and "house_closed" not in motif_codes and (
        angular_loops >= 1 or (closed_loops >= 2 and sharpness_index > 0.28)
    ):
        motif_codes.add("enclosed_shelter_or_barrier")
        detected_motifs.append({
            "code": "enclosed_shelter_or_barrier",
            "label": "包覆結構／界線框",
            "theory": "Buck (1948) 屋樹人（HTP）邊界與庇護象徵",
            "meaning": "封閉方形或圍欄般的輪廓，反映對安全基地與保護的重視，或感受到某種框架與壓力。",
        })

    if "person_tiny_isolated" not in motif_codes and (
        bbox_area_ratio < 0.065 or (bbox_area_ratio < 0.20 and center_dist > 0.26)
    ):
        motif_codes.add("tiny_corner_cluster")
        detected_motifs.append({
            "code": "tiny_corner_cluster",
            "label": "微小／偏向一隅的構圖比例",
            "theory": "Koppitz (1968) 微小圖形情緒指標（Tiny Figure）",
            "meaning": "主體僅佔畫紙一小角且周圍大片留白，常與退縮、害羞、孤單感或覺得自己聲音微弱有關。",
        })

    if edge_touch_ratio >= 0.30 or bbox_area_ratio > 0.80:
        motif_codes.add("edge_overflow")
        detected_motifs.append({
            "code": "edge_overflow",
            "label": "突破畫紙邊界的擴張構圖",
            "theory": "Buck (1948) 邊界溢出與情緒外放指標",
            "meaning": "筆觸大量延伸至畫布邊緣，顯示高漲的情緒張力、強烈表達欲或難以被框住的衝動能量。",
        })

    if revision_events >= 3 or revision_index > 0.36:
        motif_codes.add("heavy_revision")
        detected_motifs.append({
            "code": "heavy_revision",
            "label": "反覆擦除與修正歷程",
            "theory": "Koppitz (1968) 與 Buck HTP 擦除焦慮指標",
            "meaning": "多次撤銷或塗改可能代表對自我表現的猶豫、要求完美、緊張，或在思考該如何表達內心想法。",
        })

    # --- 5. Spontaneous Verbalization Analysis ---
    verbal_scores = {k: 0.0 for k in EMOTION_KEYS}
    matched_words: Dict[str, List[str]] = {k: [] for k in EMOTION_KEYS}
    if title_or_note:
        lower_note = title_or_note.lower()
        for emo, words in VERBAL_LEXICON.items():
            hits = [w for w in words if w in lower_note]
            if hits:
                matched_words[emo] = hits
                verbal_scores[emo] = _clamp(0.18 + 0.08 * (len(hits) - 1), 0.0, 0.36)

    # --- 6. Shape-First Six-Dimension Psychological Inference ---
    # Start with composite figure & expression weights (45–55% primary driver!)
    scores: Dict[str, float] = {k: 0.10 + fig_boosts.get(k, 0.0) for k in EMOTION_KEYS}

    # In monochrome sketch mode, neutral black/grey pen does NOT penalize positive emotions
    mono_dark_pen = 0.0 if is_monochrome_sketch else 1.0

    # (A) ANGER & FRUSTRATION
    scores["anger"] += (
        sharpness_index * 0.34
        + velocity_index * 0.22
        + weight_index * 0.14
        + overlap_index * (0.20 if (sharpness_index > 0.3 or intense_red > 0.15) else 0.06)
        + edge_touch_ratio * 0.16
        + (0.22 if "dense_scribble_knot" in motif_codes and (velocity_index > 0.48 or intense_red > 0.15) else 0.0)
        + (0.14 if radiating_hub_tense else 0.0)
        + color_scale * (
            intense_red * 0.50
            + dark_muted * (0.24 if (sharpness_index > 0.28 or velocity_index > 0.55) else 0.04) * mono_dark_pen
            - (warm_bright + fresh_green) * 0.12
        )
        - smoothness_index * 0.12
    )

    # (B) SADNESS & MELANCHOLY
    bottom_weight = grid_3x3[6] + grid_3x3[7] + grid_3x3[8]
    downward_ratio = downward_strokes / max(1, len(draw_strokes))
    low_speed = 1.0 - velocity_index
    melancholic_cool = (cool_blue + neutral_grey * 0.9) * (1.25 if mean_lum < 0.52 else 0.85)
    scores["sadness"] += (
        downward_ratio * 0.34
        + low_speed * 0.16
        + _clamp((bottom_weight - 0.33) * 1.4) * 0.18
        + (0.22 if "downward_streaks" in motif_codes else 0.0)
        + (0.12 if "tiny_corner_cluster" in motif_codes else 0.0)
        + color_scale * (
            melancholic_cool * 0.44
            + dark_muted * (0.24 if velocity_index < 0.55 else 0.06) * mono_dark_pen
            + (1.0 - mean_lum) * 0.16 * mono_dark_pen
            - warm_bright * 0.20
            - fresh_green * 0.14
        )
    )

    # (C) ANXIETY & TENSION
    scores["anxiety"] += (
        revision_index * 0.38
        + fragmentation_index * 0.28
        + overlap_index * 0.28
        + sharpness_index * 0.16
        + (0.18 if "heavy_revision" in motif_codes else 0.0)
        + (0.14 if "enclosed_shelter_or_barrier" in motif_codes and (overlap_index > 0.30 or revision_events >= 2) else 0.0)
        + (0.14 if "dense_scribble_knot" in motif_codes and intense_red < 0.25 else 0.0)
        + _clamp(pause_count / 5.0) * 0.10
        + color_scale * ((dark_muted + neutral_grey) * 0.24 * mono_dark_pen - warm_bright * 0.10)
        - smoothness_index * 0.10
    )

    # (D) LONELINESS & SEEKING CONNECTION
    sparsity = _clamp(1.0 - ink_coverage / 0.10)
    tiny_factor = _clamp(1.0 - bbox_area_ratio / 0.22) * (0.35 + 0.65 * center_dist)
    scores["loneliness"] += (
        tiny_factor * 0.36
        + sparsity * 0.16
        + center_dist * 0.30
        + (0.22 if "tiny_corner_cluster" in motif_codes else 0.0)
        + (0.10 if (len(draw_strokes) <= 7 and bbox_area_ratio < 0.16 and center_dist > 0.24) else 0.0)
        + color_scale * ((cool_blue + neutral_grey) * 0.24 - intense_red * 0.14)
        - (0.14 if bbox_area_ratio > 0.55 else 0.0)
    )

    # (E) JOY & VITALITY
    generous_balanced = 1.0 - abs(bbox_area_ratio - 0.50) / 0.50
    color_richness = _clamp((active_buckets - 1) / 3.0)
    upward_ratio = upward_strokes / max(1, len(draw_strokes))
    scores["joy"] += (
        smoothness_index * 0.20
        + _clamp(generous_balanced) * 0.16
        + upward_ratio * 0.12
        + (0.20 if radiating_hub_warm and "sun_figure" not in motif_codes else 0.0)
        + color_scale * (
            warm_bright * 0.62
            + pink_purple * 0.42
            + fresh_green * 0.22
            + color_richness * 0.24
            + mean_lum * 0.16
            + mean_sat * 0.16
            - dark_heavy_palette * 0.28 * mono_dark_pen
        )
        - sharpness_index * 0.18
        - overlap_index * 0.14
    )

    # (F) CALM & GROUNDEDNESS
    centeredness = 1.0 - center_dist
    steady_tempo = 1.0 - abs(velocity_index - 0.34) / 0.66
    low_tension = (1.0 - sharpness_index) * (1.0 - overlap_index) * (1.0 - revision_index)
    serene_blue = cool_blue * (0.38 if (mean_lum >= 0.44 and "downward_streaks" not in motif_codes and "rain_cloud_figure" not in motif_codes) else 0.08)
    scores["calm"] += (
        smoothness_index * 0.24
        + centeredness * 0.20
        + _clamp(steady_tempo) * 0.14
        + low_tension * 0.20
        + (1.0 - fragmentation_index) * 0.06
        + color_scale * (
            fresh_green * 0.56
            + serene_blue
            + warm_bright * 0.04
            + pink_purple * 0.04
            - intense_red * 0.30
            - dark_muted * 0.24 * mono_dark_pen
            - neutral_grey * 0.18 * mono_dark_pen
        )
        - downward_ratio * 0.24
        - (0.22 if ("downward_streaks" in motif_codes or "rain_cloud_figure" in motif_codes) else 0.0)
        - (0.18 if ("tiny_corner_cluster" in motif_codes or "person_tiny_isolated" in motif_codes) else 0.0)
    )

    visual_scores_snapshot = {k: max(0.02, v) for k, v in scores.items()}
    for tag in selected_motifs:
        for emo, w in USER_MOTIF_WEIGHTS.get(tag, {}).items():
            scores[emo] += w
    for emo, v_score in verbal_scores.items():
        scores[emo] += v_score

    clamped = {k: max(0.03, v) for k, v in scores.items()}
    max_val = max(clamped.values())
    total_val = sum(clamped.values())

    emotion_results = []
    for k in EMOTION_KEYS:
        rel_pct = round((clamped[k] / total_val) * 100)
        dominance = clamped[k] / max(0.45, max_val)
        intensity = _clamp(clamped[k] / 1.35, 0.0, 1.0)
        strength = int(round(_clamp(dominance * 56 + intensity * 38, 6, 95)))
        prof = EMOTION_PROFILES[k]
        emotion_results.append({
            "key": k,
            "label": prof["label"],
            "shortLabel": prof["short_label"],
            "enLabel": prof["en_label"],
            "valence": prof["valence"],
            "accentHex": prof["accent_hex"],
            "score": strength,
            "sharePercent": rel_pct,
        })

    emotion_results.sort(key=lambda item: item["score"], reverse=True)
    primary = emotion_results[0]
    secondary = emotion_results[1]
    primary_key = primary["key"]

    incongruence_note = None
    if title_or_note:
        visual_top = max(visual_scores_snapshot, key=lambda k: visual_scores_snapshot[k])
        verbal_positive = (verbal_scores["joy"] + verbal_scores["calm"]) > 0.14
        visual_distress = visual_top in ("anger", "sadness", "anxiety", "loneliness") and (
            visual_scores_snapshot[visual_top] > visual_scores_snapshot["joy"] + 0.16
        )
        if verbal_positive and visual_distress:
            incongruence_note = (
                f"你寫下的文字帶有正向或安撫語氣，但畫面的圖形、筆觸與構圖仍隱約呈現「{EMOTION_PROFILES[visual_top]['label']}」的張力。"
                "在兒童心理學中這常被稱為『情緒掩飾』——有時嘴上說著「我沒事、我很開心」，畫筆卻誠實替內心的疲憊或委屈留下了線索。"
            )

    # --- 7. Build Four-Pillar Evidence Narratives ---
    dominant_color_bucket = max(color_hist, key=lambda k: color_hist[k])
    color_bucket_labels = {
        "warmBright": "明亮暖色（黃／橘）",
        "intenseRed": "強烈紅色系",
        "gentlePinkPurple": "柔和粉紫色系",
        "freshGreen": "清新自然綠色",
        "coolBlue": "冷色藍色系",
        "darkMuted": "深沉暗色（黑／深棕／暗紫）",
        "neutralGrey": "低飽和灰色系",
    }
    if is_monochrome_sketch:
        color_insight = (
            "偵測到本次為單色線條作畫（類似經典鉛筆測驗）。系統已自動啟動「單色鉛筆去偏」，"
            "不因黑色或單一筆色給予負向扣分，改由完整圖形結構、臉部表情、構圖比例與筆觸動力作為主要判斷依據。"
        )
    elif (intense_red + dark_muted) >= 0.55 and intense_red >= 0.18:
        color_insight = "色彩由高張力的紅與深暗色輔助主導，參考 Alschuler & Hattwick 與 Burkitt 研究，高對比濃烈用色常呼應憤怒、挫敗或迫切的情感宣洩。"
    elif (cool_blue + neutral_grey + dark_muted) >= 0.58:
        color_insight = "色調偏向冷色、低彩度或暗沉色系，依 Burkitt 等人（2003）研究常與低落、難過或深層內省情緒相伴（仍以完整圖形結構為主導）。"
    elif positive_warm_palette >= 0.55:
        color_insight = "暖明色與自然色系帶來溫暖正向的氛圍基調，輔助強化作品的開放與活力感受。"
    else:
        color_insight = "色彩冷暖交錯具多層次變化，與畫面圖形共同映照出豐富細微的情緒層次。"

    if sharpness_index >= 0.34 and velocity_index >= 0.50:
        stroke_insight = "筆觸包含明顯的尖銳折角與較快劃動速度，呈現高度情緒喚起、防衛張力或急切宣洩挫折感的動力特徵（Alschuler & Hattwick）。"
    elif overlap_index >= 0.44:
        stroke_insight = "多處出現反覆重疊與密集塗抹痕跡，對應 Koppitz（1968）『重度陰影／塗黑』指標，常反映焦慮、緊繃或對某種感受的反覆思量。"
    elif fragmentation_index >= 0.45:
        stroke_insight = "線條偏短且較多斷續與停頓，顯示落筆時帶著小心翼翼、猶豫或能量偏低的心境。"
    elif smoothness_index >= 0.42:
        stroke_insight = "整體線條圓潤連貫、轉折柔和，在 Lowenfeld 與 Malchiodi 的觀點中代表情緒流動順暢、內在防禦較低。"
    else:
        stroke_insight = "筆觸兼具直線結構與柔和曲線，節奏平穩可控，反映創作者正以有條理的方式組織畫面。"

    horiz_pos = "偏左（內向回顧／尋求依附）" if cx < 0.38 else ("偏右（面向外界／探索環境）" if cx > 0.62 else "水平居中（平衡穩定）")
    vert_pos = "偏上（想像思緒／渴望抽離）" if cy < 0.38 else ("偏下（尋求踏實／情緒沉重）" if cy > 0.62 else "垂直居中（立足當下）")
    if bbox_area_ratio < 0.18:
        spatial_insight = f"作品主體僅佔畫布約 {round(bbox_area_ratio * 100)}% 且重心{horiz_pos}、{vert_pos}。依 Koppitz 微小圖形與 Buck HTP 觀點，偏小縮聚構圖常暗示退縮、害羞或孤單感。"
    elif edge_touch_ratio >= 0.30 or bbox_area_ratio > 0.78:
        spatial_insight = f"畫面大幅延展至畫布邊緣（覆蓋範圍約 {round(bbox_area_ratio * 100)}%），呈現強烈外放與突破框限的能量。"
    else:
        spatial_insight = f"畫面佔比約 {round(bbox_area_ratio * 100)}%，空間重心位於{horiz_pos}、{vert_pos}，整體留白與主體比例協調（Buck HTP）。"

    if recognized_figures:
        fig_names = "、".join(f["label"] for f in recognized_figures[:3])
        process_insight = f"系統透過多筆空間拓樸成功辨識出完整圖形：「{fig_names}」，並作為本次心理狀態推估的核心依據。"
    elif revision_events >= 3:
        process_insight = f"繪畫過程中進行了 {revision_events} 次復原或擦除動作，反映對自我表現的猶豫、高標準或緊張感（Koppitz 擦除指標）。"
    else:
        process_insight = f"全作共 {len(draw_strokes)} 筆自然完成，未偵測到固定具象圖式，主要綜合筆觸動力、構圖比例與色彩表現進行推估。"

    prof_primary = EMOTION_PROFILES[primary_key]
    voice_idx = (len(draw_strokes) + len(detected_motifs)) % len(prof_primary["inner_voices"])
    primary_voice = prof_primary["inner_voices"][voice_idx]

    inner_thoughts_paragraph = prof_primary["summary"]
    if recognized_figures:
        fig_names = "、".join(f["label"] for f in recognized_figures[:3])
        inner_thoughts_paragraph = f"畫面中辨識出「{fig_names}」等關鍵圖形特徵。{inner_thoughts_paragraph}"
    if secondary["score"] >= primary["score"] - 14:
        inner_thoughts_paragraph += f" 同時隱約流露出「{secondary['label']}」的細微底色，顯示當下感受是立體而富有層次的。"
    if title_or_note:
        inner_thoughts_paragraph += f" 結合你留下的文字「{title_or_note}」，更清楚看見這幅畫正在替此刻的心境發聲。"

    return {
        "status": "ok",
        "engineMode": "local_shape_first_v2",
        "isMonochromeSketch": is_monochrome_sketch,
        "primaryEmotion": primary,
        "secondaryEmotion": secondary,
        "emotions": emotion_results,
        "innerVoice": primary_voice,
        "innerThoughts": inner_thoughts_paragraph,
        "incongruenceNote": incongruence_note,
        "carePrompts": prof_primary["care_prompts"],
        "recognizedFigures": recognized_figures,
        "detectedMotifs": detected_motifs,
        "pillars": {
            "color": {
                "title": "色彩與情感投射",
                "citation": "Burkitt et al. (2003); Alschuler & Hattwick (1947)",
                "dominantBucket": color_bucket_labels.get(dominant_color_bucket, dominant_color_bucket),
                "isMonochromeSketch": is_monochrome_sketch,
                "histogram": color_hist,
                "meanSaturation": round(mean_sat, 3),
                "meanLuminance": round(mean_lum, 3),
                "insight": color_insight,
            },
            "strokes": {
                "title": "線條品質與筆觸動力",
                "citation": "Lowenfeld (1987); Malchiodi (1998)",
                "strokeCount": len(draw_strokes),
                "sharpnessIndex": round(sharpness_index, 3),
                "smoothnessIndex": round(smoothness_index, 3),
                "velocityIndex": round(velocity_index, 3),
                "overlapIndex": round(overlap_index, 3),
                "fragmentationIndex": round(fragmentation_index, 3),
                "insight": stroke_insight,
            },
            "spatial": {
                "title": "空間構圖與九宮格重心",
                "citation": "Buck HTP (1948); Koppitz (1968)",
                "grid3x3": grid_3x3,
                "centerOfMass": {"x": round(cx, 3), "y": round(cy, 3)},
                "bboxCoverage": round(bbox_area_ratio, 3),
                "inkCoverage": round(ink_coverage, 4),
                "edgeTouchRatio": round(edge_touch_ratio, 3),
                "insight": spatial_insight,
            },
            "process": {
                "title": "完整圖形辨識與繪畫歷程",
                "citation": "Buck HTP (1948); Koppitz Emotional Indicators (1968)",
                "undoCount": undo_count,
                "eraserCount": eraser_count,
                "revisionIndex": round(revision_index, 3),
                "durationSeconds": round(total_duration_ms / 1000.0, 1),
                "matchedVerbalCues": {k: v for k, v in matched_words.items() if v},
                "insight": process_insight,
            },
        },
        "references": PSYCHOLOGY_REFERENCES,
        "ethicalNotice": (
            "本分析參考兒童發展心理學與藝術治療投射理論（HTP 屋樹人、Koppitz 情緒指標、Lowenfeld 發展階段及 Burkitt 色彩研究），"
            "旨在提供溫柔的情緒探索與親子／自我對話起點，並非臨床心理或醫療診斷。"
        ),
    }
