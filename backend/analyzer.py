"""Multi-dimensional child drawing psychology analysis engine.

Extracts objective visual, kinematic, spatial, and process telemetry from a
freehand drawing (vector strokes + optional rendered PNG via Pillow + optional
spontaneous verbalization) and maps them onto child psychology and projective
art therapy frameworks (Buck HTP, Koppitz HFD, Lowenfeld, Malchiodi, Burkitt
et al., Alschuler & Hattwick).
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
    """Classify an RGB pixel into psychological color affect buckets.

    Returns (bucket_name, saturation, luminance).
    """
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    hue_deg = h * 360.0
    lum = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf

    # Very dark / charcoal / black / deep shadow
    if v < 0.24 or (lum < 0.20 and s < 0.45):
        return ("darkMuted", s, lum)

    # Low saturation greys / near-whites
    if s < 0.14:
        if v < 0.48:
            return ("darkMuted", s, lum)
        return ("neutralGrey", s, lum)

    # Deep muted browns / dark purples
    if v < 0.40 and (hue_deg < 48 or hue_deg >= 255):
        return ("darkMuted", s, lum)

    # Vivid/intense red & crimson (anger/high arousal energy)
    if (hue_deg <= 14 or hue_deg >= 342) and s >= 0.48 and v >= 0.35:
        if v >= 0.72 and s < 0.65:
            return ("gentlePinkPurple", s, lum)
        return ("intenseRed", s, lum)

    # Warm bright: orange, yellow, golden, warm coral
    if 14 < hue_deg <= 68:
        if v < 0.48 and s < 0.65:
            return ("darkMuted", s, lum)  # muddy dark brown
        return ("warmBright", s, lum)

    # Fresh green / nature / vitality
    if 68 < hue_deg <= 165:
        if v < 0.34:
            return ("darkMuted", s, lum)
        return ("freshGreen", s, lum)

    # Cool blue / indigo / cyan-blue
    if 165 < hue_deg <= 255:
        return ("coolBlue", s, lum)

    # Soft pink / lavender / violet
    if 255 < hue_deg < 342:
        if v < 0.42:
            return ("darkMuted", s, lum)
        return ("gentlePinkPurple", s, lum)

    return ("neutralGrey", s, lum)


def _resample_points(
    points: List[Dict[str, Any]], step: float = 12.0
) -> List[Tuple[float, float]]:
    """Resample polyline at uniform spatial spacing to remove pointer jitter."""
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
    """Ramer-Douglas-Peucker simplification to extract true corners vs smooth arcs."""
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
        if seg_len < 1e-6:
            d = math.hypot(px - x1, py - y1)
        else:
            d = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / seg_len
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > epsilon:
        left = _rdp_simplify(points[: max_idx + 1], epsilon)
        right = _rdp_simplify(points[max_idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _analyze_image_buffer(
    image_data_url: str, bg_hex: str
) -> Optional[Dict[str, Any]]:
    """Use Pillow to compute exact pixel coverage, 3x3 grid density, and color buckets."""
    if not image_data_url or "," not in image_data_url:
        return None
    try:
        _, b64_part = image_data_url.split(",", 1)
        raw = base64.b64decode(b64_part)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        # Downsample cleanly for fast, accurate pixel statistics
        target_w = 180
        ratio = target_w / max(1, img.width)
        target_h = max(60, int(round(img.height * ratio)))
        img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    except Exception:
        return None

    bg_r, bg_g, bg_b = _hex_to_rgb(bg_hex)
    if hasattr(img, "get_flattened_data"):
        pixels = list(img.get_flattened_data())
    else:
        pixels = list(img.getdata())
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
    min_x, max_x = w, 0
    min_y, max_y = h, 0

    for idx, (r, g, b) in enumerate(pixels):
        dist = abs(r - bg_r) + abs(g - bg_g) + abs(b - bg_b)
        if dist < 26:
            continue
        x = idx % w
        y = idx // w
        ink_count += 1
        sum_x += x / max(1, w - 1)
        sum_y += y / max(1, h - 1)
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

        col = min(2, int((x / w) * 3))
        row = min(2, int((y / h) * 3))
        grid[row * 3 + col] += 1.0

        bucket, sat, lum = classify_rgb_bucket(r, g, b)
        buckets[bucket] += 1
        sat_sum += sat
        lum_sum += lum

    if ink_count == 0:
        return {
            "inkCoverageRatio": 0.0,
            "grid3x3": [0.0] * 9,
            "colorHistogram": {k: 0.0 for k in buckets},
            "meanSaturation": 0.0,
            "meanLuminance": 0.5,
            "pixelCenter": (0.5, 0.5),
            "pixelBboxAreaRatio": 0.0,
        }

    cell_px = total_px / 9.0
    norm_grid = [round(c / max(1.0, ink_count), 4) for c in grid]
    hist = {k: round(v / ink_count, 4) for k, v in buckets.items()}
    bbox_area = ((max_x - min_x + 1) * (max_y - min_y + 1)) / max(1, total_px)

    return {
        "inkCoverageRatio": round(ink_count / total_px, 4),
        "grid3x3": norm_grid,
        "gridCellCoverage": [round(c / cell_px, 4) for c in grid],
        "colorHistogram": hist,
        "meanSaturation": round(sat_sum / ink_count, 4),
        "meanLuminance": round(lum_sum / ink_count, 4),
        "pixelCenter": (round(sum_x / ink_count, 4), round(sum_y / ink_count, 4)),
        "pixelBboxAreaRatio": round(bbox_area, 4),
    }


def analyze_drawing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run full child psychology analysis on drawing telemetry and optional PNG."""
    canvas_info = payload.get("canvas") or {}
    width = float(canvas_info.get("width") or 800.0)
    height = float(canvas_info.get("height") or 540.0)
    bg_hex = str(canvas_info.get("backgroundColor") or "#FDFBF7")
    diag = math.hypot(width, height)
    canvas_area = max(1.0, width * height)

    all_strokes: List[Dict[str, Any]] = payload.get("strokes") or []
    draw_strokes = [
        s
        for s in all_strokes
        if s.get("tool") != "eraser" and len(s.get("points") or []) >= 1
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

    # --- 1. Stroke Kinematics, Color Weighting, Spatial Grid & Geometry ---
    total_path_length = 0.0
    total_drawn_ink_area = 0.0
    stroke_lengths: List[float] = []
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

    # Grid & color tracking from vector strokes (fallback or cross-check with Pillow)
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
    sat_weighted = 0.0
    lum_weighted = 0.0
    unique_colors = set()
    stroke_centers: List[Tuple[float, float, str]] = []
    radiating_candidates: List[Tuple[float, float, float, float, str]] = []

    for stroke in draw_strokes:
        pts = stroke.get("points") or []
        color_hex = str(stroke.get("color") or "#222222")
        tool = str(stroke.get("tool") or "crayon")
        size = float(stroke.get("size") or 6.0)
        tool_width_mult = {
            "pencil": 0.65,
            "crayon": 1.15,
            "marker": 1.3,
            "watercolor": 1.65,
        }.get(tool, 1.0)
        eff_width = size * tool_width_mult
        widths.append(eff_width)

        r, g, b = _hex_to_rgb(color_hex)
        bucket, sat, lum = classify_rgb_bucket(r, g, b)
        unique_colors.add(color_hex.lower())

        # Compute raw length, velocity, pressure, bbox
        s_len = 0.0
        s_min_x, s_max_x = width, 0.0
        s_min_y, s_max_y = height, 0.0
        touches_edge = False
        s_cx_sum, s_cy_sum = 0.0, 0.0

        for idx, pt in enumerate(pts):
            x = float(pt.get("x", 0.0))
            y = float(pt.get("y", 0.0))
            p = float(pt.get("p", 0.5))
            pressures.append(_clamp(p, 0.05, 1.0))

            s_cx_sum += x
            s_cy_sum += y
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            s_min_x, s_max_x = min(s_min_x, x), max(s_max_x, x)
            s_min_y, s_max_y = min(s_min_y, y), max(s_max_y, y)

            if (
                x < width * 0.055
                or x > width * 0.945
                or y < height * 0.055
                or y > height * 0.945
            ):
                touches_edge = True

            if idx > 0:
                px = float(pts[idx - 1].get("x", x))
                py = float(pts[idx - 1].get("y", y))
                dt = max(4.0, float(pt.get("t", 0.0)) - float(pts[idx - 1].get("t", 0.0)))
                seg_d = math.hypot(x - px, y - py)
                s_len += seg_d
                if seg_d > 0.5 and dt < 1500.0:
                    velocities.append(min(4.5, seg_d / dt))  # px per ms

        # Single dot stroke fallback length
        if s_len < 1.0:
            s_len = eff_width * 1.2

        stroke_lengths.append(s_len)
        total_path_length += s_len
        ink_area = s_len * max(2.0, eff_width)
        total_drawn_ink_area += ink_area

        if touches_edge:
            edge_touch_strokes += 1
        if s_len < width * 0.065:
            short_fragment_strokes += 1

        # Spatial resampling for clean grid, overlap, and curvature
        resamp = _resample_points(pts, step=12.0)
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

        stroke_centers.append((s_cx_sum / len(pts), s_cy_sum / len(pts), bucket))

        # Angular sharpness vs smooth curvature on RDP-simplified points (eliminates collinear dilution)
        stroke_bbox_diag_pre = math.hypot(s_max_x - s_min_x, s_max_y - s_min_y)
        rdp_eps = min(5.5, max(1.6, stroke_bbox_diag_pre * 0.055))
        simp = _rdp_simplify(resamp, epsilon=rdp_eps)
        s_sharp = 0
        s_smooth = 0
        s_turns = 0
        for i in range(1, len(simp) - 1):
            v1x = simp[i][0] - simp[i - 1][0]
            v1y = simp[i][1] - simp[i - 1][1]
            v2x = simp[i + 1][0] - simp[i][0]
            v2y = simp[i + 1][1] - simp[i][1]
            m1 = math.hypot(v1x, v1y)
            m2 = math.hypot(v2x, v2y)
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

        # Directional trajectory (downward rain/tears vs upward growth)
        start_pt, end_pt = pts[0], pts[-1]
        dx_total = float(end_pt["x"]) - float(start_pt["x"])
        dy_total = float(end_pt["y"]) - float(start_pt["y"])
        chord = math.hypot(dx_total, dy_total)
        stroke_bbox_diag = math.hypot(s_max_x - s_min_x, s_max_y - s_min_y)

        if s_len > 18.0:
            # Mostly straight stroke?
            straightness = chord / max(1.0, s_len)
            if straightness > 0.72:
                # Canvas +y is downward; require meaningful vertical span (>36px) so tiny stick legs aren't mistaken for rain
                if dy_total > abs(dx_total) * 1.15 and dy_total > 36.0:
                    downward_strokes += 1
                elif -dy_total > abs(dx_total) * 1.15 and -dy_total > 36.0:
                    upward_strokes += 1
                radiating_candidates.append(
                    (float(start_pt["x"]), float(start_pt["y"]), float(end_pt["x"]), float(end_pt["y"]), bucket)
                )

            # Closed shape loop?
            if stroke_bbox_diag > 24.0 and chord < stroke_bbox_diag * 0.42 and s_len > stroke_bbox_diag * 1.9:
                closed_loops += 1
                if s_turns > 0 and (s_sharp / s_turns) < 0.25:
                    smooth_loops += 1
                elif s_sharp >= 2:
                    angular_loops += 1

            # Dense back-and-forth scribble stroke?
            if s_turns >= 4 and (s_sharp / max(1, s_turns)) > 0.35 and s_len > stroke_bbox_diag * 2.2:
                scribble_strokes += 1

    # --- 2. Combine Vector Telemetry with Pillow Pixel Analysis (when provided) ---
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
        cx, cy = pixel_stats.get("pixelCenter", (weighted_x_sum / max(1e-6, total_weight), weighted_y_sum / max(1e-6, total_weight)))
    else:
        color_hist = vec_hist
        grid_3x3 = norm_vec_grid
        unique_cells = len(cell_occupancy)
        ink_coverage = _clamp((unique_cells * 14.0 * 14.0) / canvas_area, 0.002, 0.95)
        mean_sat = sat_weighted / vec_total_area
        mean_lum = lum_weighted / vec_total_area
        cx = weighted_x_sum / max(1e-6, total_weight)
        cy = weighted_y_sum / max(1e-6, total_weight)

    # Overlap / shading intensity (classic Koppitz shading/over-drawing measure)
    covered_px_est = max(150.0, ink_coverage * canvas_area)
    raw_overlap_ratio = total_drawn_ink_area / covered_px_est
    multi_hit_cells = sum(1 for c in cell_occupancy.values() if c >= 3)
    cell_overlap_ratio = multi_hit_cells / max(1, len(cell_occupancy))
    overlap_index = _clamp((raw_overlap_ratio - 1.0) / 3.2 * 0.65 + cell_overlap_ratio * 0.55)

    # Kinematic indices
    sharpness_index = _clamp((sharp_turns / max(1, total_turns)) * 1.15)
    smoothness_index = _clamp((smooth_turns / max(1, total_turns)) * 1.15)
    mean_vel = sum(velocities) / max(1, len(velocities)) if velocities else 0.55
    velocity_index = _clamp(mean_vel / 1.8)  # 0 = very slow, 1 = very rapid slashing
    mean_width = sum(widths) / max(1, len(widths))
    mean_pressure = sum(pressures) / max(1, len(pressures))
    weight_index = _clamp((mean_width / 22.0) * 0.65 + mean_pressure * 0.35)

    fragmentation_index = _clamp(
        (short_fragment_strokes / max(1, len(draw_strokes))) * 0.75
        + _clamp(pause_count / 6.0) * 0.25
    )

    # Revision / self-doubt indicator (Koppitz / Buck)
    revision_events = undo_count + eraser_count
    revision_ratio = revision_events / max(1.0, len(draw_strokes) + revision_events)
    revision_index = _clamp(revision_ratio * 1.8 + _clamp(revision_events / 8.0) * 0.45)

    # Spatial indices
    bbox_w_ratio = _clamp((max_x - min_x) / width)
    bbox_h_ratio = _clamp((max_y - min_y) / height)
    bbox_area_ratio = _clamp(bbox_w_ratio * bbox_h_ratio)
    center_dist = math.hypot(cx - 0.5, cy - 0.5) / 0.7071  # 0 = dead center, 1 = corner
    edge_touch_ratio = edge_touch_strokes / max(1, len(draw_strokes))

    # Color aggregates
    warm_bright = color_hist.get("warmBright", 0.0)
    intense_red = color_hist.get("intenseRed", 0.0)
    pink_purple = color_hist.get("gentlePinkPurple", 0.0)
    fresh_green = color_hist.get("freshGreen", 0.0)
    cool_blue = color_hist.get("coolBlue", 0.0)
    dark_muted = color_hist.get("darkMuted", 0.0)
    neutral_grey = color_hist.get("neutralGrey", 0.0)

    positive_warm_palette = warm_bright + fresh_green * 0.75 + pink_purple * 0.65
    cool_muted_palette = cool_blue + neutral_grey * 0.85 + dark_muted * 0.45
    dark_heavy_palette = dark_muted + neutral_grey * 0.5
    active_buckets = sum(1 for v in color_hist.values() if v >= 0.06)

    # --- 3. Algorithmic Geometric Motif Detection ---
    detected_motifs: List[Dict[str, str]] = []
    motif_codes = set()

    if scribble_strokes >= 2 or (overlap_index > 0.48 and sharpness_index > 0.35):
        motif_codes.add("dense_scribble_knot")
        detected_motifs.append({
            "code": "dense_scribble_knot",
            "label": "密集交織／情緒塗抹筆觸",
            "theory": "Koppitz (1968) 重度陰影與反覆塗抹指標",
            "meaning": "重複來回交疊的強烈筆觸，常代表內心積聚張力、煩躁、焦慮或試圖覆蓋某種不安感受。",
        })

    if downward_strokes >= 4 or (downward_strokes >= 3 and (cool_blue + dark_muted + neutral_grey) > 0.45):
        motif_codes.add("downward_streaks")
        detected_motifs.append({
            "code": "downward_streaks",
            "label": "向下垂落／雨絲般的線條",
            "theory": "Malchiodi (1998) 動態線條方向象徵",
            "meaning": "多道朝下延展的線條宛如雨滴或垂落感，常見於情緒低落、委屈、悲傷或內在沉重時。",
        })

    # Detect radiating strokes around a shared hub
    radiating_hub_warm = False
    radiating_hub_tense = False
    if len(radiating_candidates) >= 5:
        # Check if multiple stroke starts or centers cluster together while pointing different directions
        angles = set()
        warm_rad = 0
        tense_rad = 0
        for sx, sy, ex, ey, bkt in radiating_candidates:
            ang_bin = int((math.degrees(math.atan2(ey - sy, ex - sx)) + 180) // 45)
            angles.add(ang_bin)
            if bkt in ("warmBright", "gentlePinkPurple"):
                warm_rad += 1
            elif bkt in ("intenseRed", "darkMuted"):
                tense_rad += 1
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

    if angular_loops >= 1 or (closed_loops >= 2 and sharpness_index > 0.28):
        motif_codes.add("enclosed_shelter_or_barrier")
        detected_motifs.append({
            "code": "enclosed_shelter_or_barrier",
            "label": "包覆結構／屋舍或界線框",
            "theory": "Buck (1948) 屋樹人（HTP）邊界與庇護象徵",
            "meaning": "封閉方形或圍欄般的輪廓，反映對安全基地與保護的重視，或感受到某種框架與壓力。",
        })

    if smooth_loops >= 1:
        motif_codes.add("circular_organic_forms")
        detected_motifs.append({
            "code": "circular_organic_forms",
            "label": "圓潤封閉形體（太陽／臉龐／包覆感）",
            "theory": "Kellogg & Lowenfeld 圓形核心圖式",
            "meaning": "流暢的圓弧封閉線條常見於描繪人臉、太陽或柔和物件，偏向表達完整感、親和力或情感連結。",
        })

    if bbox_area_ratio < 0.16 or (bbox_area_ratio < 0.24 and center_dist > 0.34):
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

    # --- 4. Spontaneous Verbalization Analysis (Malchiodi) ---
    verbal_scores = {k: 0.0 for k in EMOTION_KEYS}
    matched_words: Dict[str, List[str]] = {k: [] for k in EMOTION_KEYS}
    if title_or_note:
        lower_note = title_or_note.lower()
        for emo, words in VERBAL_LEXICON.items():
            hits = [w for w in words if w in lower_note]
            if hits:
                matched_words[emo] = hits
                verbal_scores[emo] = _clamp(0.18 + 0.08 * (len(hits) - 1), 0.0, 0.36)

    # --- 5. Calibrated Six-Dimension Psychological Inference ---
    scores: Dict[str, float] = {k: 0.08 for k in EMOTION_KEYS}

    # (A) ANGER & FRUSTRATION: sharp jaggedness + vivid red/dark + high speed + heavy strokes + overflow/scribble
    scores["anger"] += (
        intense_red * 0.44
        + dark_muted * (0.24 if (sharpness_index > 0.28 or velocity_index > 0.55) else 0.06)
        + sharpness_index * 0.34
        + velocity_index * 0.22
        + weight_index * 0.16
        + overlap_index * (0.18 if (intense_red + dark_muted) > 0.35 else 0.05)
        + edge_touch_ratio * 0.16
        + (0.16 if "dense_scribble_knot" in motif_codes and (intense_red > 0.15 or velocity_index > 0.5) else 0.0)
        + (0.14 if radiating_hub_tense else 0.0)
        - smoothness_index * 0.12
        - (warm_bright + fresh_green) * 0.12
    )

    # (B) SADNESS & MELANCHOLY: cool blue / grey / dark + slow energy + downward flow + bottom weight / small size
    bottom_weight = grid_3x3[6] + grid_3x3[7] + grid_3x3[8]
    downward_ratio = downward_strokes / max(1, len(draw_strokes))
    low_speed = 1.0 - velocity_index
    melancholic_cool = (cool_blue + neutral_grey * 0.9) * (1.25 if mean_lum < 0.52 else 0.85)
    scores["sadness"] += (
        melancholic_cool * 0.38
        + dark_muted * (0.26 if velocity_index < 0.58 else 0.08)
        + downward_ratio * 0.36
        + low_speed * 0.18
        + (1.0 - mean_lum) * 0.18
        + _clamp((bottom_weight - 0.33) * 1.4) * 0.18
        + (0.20 if "downward_streaks" in motif_codes else 0.0)
        + (0.12 if "tiny_corner_cluster" in motif_codes else 0.0)
        - warm_bright * 0.22
        - intense_red * 0.14
        - fresh_green * 0.14
    )

    # (C) ANXIETY & TENSION: heavy over-drawing/shading + revisions/erasures + fragmented short strokes + dark/grey + enclosure
    scores["anxiety"] += (
        revision_index * 0.36
        + fragmentation_index * 0.28
        + overlap_index * 0.28
        + sharpness_index * 0.18
        + (dark_muted + neutral_grey) * 0.24
        + (0.15 if "heavy_revision" in motif_codes else 0.0)
        + (0.14 if "enclosed_shelter_or_barrier" in motif_codes and (dark_muted + neutral_grey + intense_red) > 0.35 else 0.0)
        + (0.12 if "dense_scribble_knot" in motif_codes and intense_red < 0.25 else 0.0)
        + _clamp(pause_count / 5.0) * 0.10
        - smoothness_index * 0.10
        - warm_bright * 0.12
    )

    # (D) LONELINESS & SEEKING CONNECTION: tiny area / sparse ink + off-center corner tuck + high empty space + cool/monochrome
    sparsity = _clamp(1.0 - ink_coverage / 0.12)
    tiny_factor = _clamp(1.0 - bbox_area_ratio / 0.26)
    monochrome_cool = 0.16 if (active_buckets <= 2 and (cool_blue + neutral_grey + dark_muted) > 0.55) else 0.0
    scores["loneliness"] += (
        tiny_factor * 0.34
        + sparsity * 0.22
        + center_dist * 0.28
        + monochrome_cool
        + (cool_blue + neutral_grey) * 0.18
        + (0.18 if "tiny_corner_cluster" in motif_codes else 0.0)
        + (0.10 if (len(draw_strokes) <= 7 and bbox_area_ratio < 0.22) else 0.0)
        - (0.14 if bbox_area_ratio > 0.55 else 0.0)
        - intense_red * 0.14
    )

    # (E) JOY & VITALITY: warm bright / pink / vibrant multi-color + smooth flowing curves + balanced/generous coverage + sun rays
    generous_balanced = 1.0 - abs(bbox_area_ratio - 0.50) / 0.50
    color_richness = _clamp((active_buckets - 1) / 3.0)
    upward_ratio = upward_strokes / max(1, len(draw_strokes))
    scores["joy"] += (
        warm_bright * 0.54
        + pink_purple * 0.36
        + fresh_green * 0.20
        + smoothness_index * 0.20
        + _clamp(generous_balanced) * 0.16
        + color_richness * 0.18
        + mean_lum * 0.14
        + mean_sat * 0.14
        + upward_ratio * 0.12
        + (0.20 if radiating_hub_warm else 0.0)
        + (0.10 if "circular_organic_forms" in motif_codes and positive_warm_palette > 0.35 else 0.0)
        - dark_heavy_palette * 0.28
        - sharpness_index * 0.18
        - overlap_index * 0.14
    )

    # (F) CALM & GROUNDEDNESS: tranquil green/sky-blue harmony + centered stability + steady gentle pace + low tension/revision
    centeredness = 1.0 - center_dist
    serene_blue = cool_blue * (0.34 if (mean_lum >= 0.44 and "downward_streaks" not in motif_codes) else 0.08)
    calm_palette = fresh_green * 0.48 + serene_blue + warm_bright * 0.04 + pink_purple * 0.04
    steady_tempo = 1.0 - abs(velocity_index - 0.34) / 0.66
    low_tension = (1.0 - sharpness_index) * (1.0 - overlap_index) * (1.0 - revision_index)
    scores["calm"] += (
        calm_palette
        + smoothness_index * 0.24
        + centeredness * 0.20
        + _clamp(steady_tempo) * 0.14
        + low_tension * 0.20
        + (1.0 - fragmentation_index) * 0.06
        - intense_red * 0.28
        - dark_muted * 0.24
        - neutral_grey * 0.18
        - downward_ratio * 0.24
        - (0.22 if "downward_streaks" in motif_codes else 0.0)
        - (0.18 if "tiny_corner_cluster" in motif_codes else 0.0)
    )

    # Integrate optional user-selected motifs & spontaneous verbalizations
    visual_scores_snapshot = {k: max(0.02, v) for k, v in scores.items()}
    for tag in selected_motifs:
        for emo, w in USER_MOTIF_WEIGHTS.get(tag, {}).items():
            scores[emo] += w

    for emo, v_score in verbal_scores.items():
        scores[emo] += v_score

    # Normalize scores into intuitive 0–100 emotional strength & relative probabilities
    clamped = {k: max(0.03, v) for k, v in scores.items()}
    max_val = max(clamped.values())
    total_val = sum(clamped.values())

    emotion_results = []
    for k in EMOTION_KEYS:
        rel_pct = round((clamped[k] / total_val) * 100)
        # Visual prominence score (0-100) reflecting both absolute signal intensity & dominance
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
    secondary_key = secondary["key"]

    # Check verbal-visual incongruence (e.g., positive words vs tense/sad visual cues)
    incongruence_note = None
    if title_or_note:
        visual_top = max(visual_scores_snapshot, key=lambda k: visual_scores_snapshot[k])
        verbal_positive = (verbal_scores["joy"] + verbal_scores["calm"]) > 0.14
        visual_distress = visual_top in ("anger", "sadness", "anxiety", "loneliness") and (
            visual_scores_snapshot[visual_top] > visual_scores_snapshot["joy"] + 0.16
        )
        if verbal_positive and visual_distress:
            incongruence_note = (
                f"你寫下的文字帶有正向或安撫語氣，但畫面的筆觸與色彩仍隱約呈現「{EMOTION_PROFILES[visual_top]['label']}」的張力。"
                "在兒童心理學中這常被稱為『情緒掩飾』——孩子有時會說「我沒事、我很開心」，但畫筆卻誠實替內心的疲憊或委屈留下了線索。"
            )

    # --- 6. Build Rich Four-Pillar Psychological Interpretation ---
    # Pillar 1: Color Psychology
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
    if (intense_red + dark_muted) >= 0.55 and intense_red >= 0.18:
        color_insight = "畫面由高張力的紅與深暗色主導，參考 Alschuler & Hattwick 與 Burkitt 等人研究，這類高對比濃烈用色常對應憤怒、強烈挫敗或迫切的情感宣洩。"
    elif (cool_blue + neutral_grey + dark_muted) >= 0.58:
        color_insight = "整體色調偏向冷色、低彩度或暗沉色系，依 Burkitt 等人（2003）色彩情感研究，低明度與冷暗色調常與低落、難過、退縮或深層內省情緒相連。"
    elif positive_warm_palette >= 0.55:
        color_insight = "暖明色與自然色系佔據主導，展現開放、溫暖與正向的情緒基調，顯示創作者當下願意向外探索並表達內在活力。"
    else:
        color_insight = "色彩冷暖交錯且具多層次變化，反映內心同時交織著多種細微感受，既有思考與自我保護，也保留著表達與連結的彈性。"

    # Pillar 2: Stroke Kinematics
    if sharpness_index >= 0.34 and velocity_index >= 0.50:
        stroke_insight = "筆觸包含明顯的尖銳折角與較快劃動速度，呈現高度情緒喚起、防衛張力或想急切宣洩挫折感的動力特徵（Alschuler & Hattwick）。"
    elif overlap_index >= 0.44:
        stroke_insight = "多處出現反覆重疊與密集塗抹痕跡，對應 Koppitz（1968）所指的『重度陰影／塗黑』指標，常反映焦慮、緊繃或對某種感受的反覆思量。"
    elif fragmentation_index >= 0.45:
        stroke_insight = "線條偏短且較多斷續與停頓，顯示落筆時帶著小心翼翼、猶豫或能量偏低的心境，可能伴隨不安或對自我表現的不確定。"
    elif smoothness_index >= 0.42:
        stroke_insight = "整體線條圓潤連貫、轉折柔和，在 Lowenfeld 與 Malchiodi 的觀點中代表情緒流動順暢、內在防禦較低且感到相對自在。"
    else:
        stroke_insight = "筆觸兼具直線結構與柔和曲線，節奏平穩可控，反映創作者正以有條理的方式整理腦海中的想法與畫面。"

    # Pillar 3: Spatial Composition
    horiz_pos = "偏左（內向回顧／尋求依附）" if cx < 0.38 else ("偏右（面向外界／探索環境）" if cx > 0.62 else "水平居中（平衡穩定）")
    vert_pos = "偏上（想像思緒／渴望抽離）" if cy < 0.38 else ("偏下（尋求踏實／情緒沉重）" if cy > 0.62 else "垂直居中（立足當下）")
    if bbox_area_ratio < 0.18:
        spatial_insight = f"作品主體僅佔畫布約 {round(bbox_area_ratio * 100)}% 且重心{horiz_pos}、{vert_pos}。依 Koppitz 微小圖形與 Buck HTP 空間觀點，偏小縮聚的構圖常暗示退縮、害羞、低落或孤單感。"
    elif edge_touch_ratio >= 0.30 or bbox_area_ratio > 0.78:
        spatial_insight = f"畫面大幅延展至畫布邊緣（覆蓋範圍約 {round(bbox_area_ratio * 100)}%），呈現強烈外放與突破框限的能量，常出現於高張力情緒宣洩或高度興奮狀態。"
    else:
        spatial_insight = f"畫面佔比約 {round(bbox_area_ratio * 100)}%，空間重心位於{horiz_pos}、{vert_pos}，整體留白與主體比例協調，反映較穩定的自我空間感（Buck HTP）。"

    # Pillar 4: Process & Symbolic Cues
    if revision_events >= 3:
        process_insight = f"繪畫過程中進行了 {revision_events} 次復原或擦除動作。在兒童繪畫評估中，頻繁修正往往反映自我要求高、擔心畫不好或對內心真實想法仍有一點猶豫。"
    elif len(detected_motifs) > 0:
        motif_names = "、".join(m["label"] for m in detected_motifs[:2])
        process_insight = f"系統從筆觸幾何中辨識出「{motif_names}」等象徵特徵，搭配整體繪畫節奏，為理解未說出口的內心狀態提供重要線索。"
    else:
        process_insight = f"全作共 {len(draw_strokes)} 筆自然完成、幾乎沒有反覆塗改，展現直覺且真誠的情感投射流動。"

    # --- 7. Contextualized Inner Voice Synthesis ---
    prof_primary = EMOTION_PROFILES[primary_key]
    voice_idx = (len(draw_strokes) + len(detected_motifs)) % len(prof_primary["inner_voices"])
    primary_voice = prof_primary["inner_voices"][voice_idx]

    # Blend nuance if secondary emotion is close
    inner_thoughts_paragraph = prof_primary["summary"]
    if secondary["score"] >= primary["score"] - 14:
        inner_thoughts_paragraph += (
            f" 同時，作品也隱約流露出「{secondary['label']}」的細微底色，"
            "顯示當下的內心感受是立體而富有層次的。"
        )
    if title_or_note:
        inner_thoughts_paragraph += f" 結合你留下的文字「{title_or_note}」，更清楚看見這幅畫正在替此刻的心境發聲。"

    return {
        "status": "ok",
        "primaryEmotion": primary,
        "secondaryEmotion": secondary,
        "emotions": emotion_results,
        "innerVoice": primary_voice,
        "innerThoughts": inner_thoughts_paragraph,
        "incongruenceNote": incongruence_note,
        "carePrompts": prof_primary["care_prompts"],
        "detectedMotifs": detected_motifs,
        "pillars": {
            "color": {
                "title": "色彩與情感投射",
                "citation": "Burkitt et al. (2003); Alschuler & Hattwick (1947)",
                "dominantBucket": color_bucket_labels.get(dominant_color_bucket, dominant_color_bucket),
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
                "title": "繪畫歷程與象徵線索",
                "citation": "Koppitz Emotional Indicators (1968); Malchiodi (1998)",
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
