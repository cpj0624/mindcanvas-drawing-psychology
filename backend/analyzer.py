"""Multi-dimensional child drawing psychology analysis engine (v3.5 Empirical).

Targets children aged 3–13 (pre-adolescent) using 29 peer-reviewed international
empirical studies (image-first, 3C/tablet debiasing, pop-culture debiasing, and
neurodiversity/ASD guardrails) plus 6 classic frameworks under modern limits.
"""

from __future__ import annotations

import base64
import colorsys
import io
import math
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .knowledge_base import (
    AGE_GROUP_NORMS,
    EMOTION_KEYS,
    EMOTION_PROFILES,
    PSYCHOLOGY_REFERENCES,
    TOOL_HARDNESS,
    USER_MOTIF_WEIGHTS,
    VERBAL_LEXICON,
)

CRISIS_KEYWORDS: Tuple[str, ...] = (
    "自傷", "自殺", "想死", "輕生", "割腕", "跳樓", "不想活", "結束生命",
    "傷害自己", "受暴", "家暴", "施暴", "虐待", "性侵", "猥褻", "性騷擾",
    "被打到", "求救", "suicide", "self-harm", "self harm", "abuse",
)

COUNSELOR_OBJECT_CUES: Dict[str, Tuple[str, ...]] = {
    "sadness": (
        "哭", "淚", "雨", "烏雲", "閃電", "淋濕", "積水", "沒遮蔽", "死", "離開",
        "想念", "失去", "受傷", "跌倒", "生病", "墓", "枯萎", "眼淚", "空巢", "無底", "斷掉",
    ),
    "anger": (
        "生氣", "怪物", "怪獸", "炸彈", "火", "吵架", "打架", "打人", "刺", "武器",
        "壞人", "火山", "暴龍", "討厭", "撕", "牙齒", "大牙齒", "尖牙", "利爪", "尖爪", "交叉雙臂",
    ),
    "anxiety": (
        "躲", "藏", "保護罩", "盾", "牆", "圍牆", "籠", "鎖", "不敢", "害怕", "怕",
        "緊張", "考試", "考試卷", "考卷", "分數", "成績", "時鐘", "鬧鐘", "作業", "功課", "螢幕", "補習", "聯絡簿",
        "迷宮", "黑影", "抖", "惡夢", "門沒開", "縮起來",
        "框框", "圈起來", "包圍", "隔開", "隔線", "柵欄", "懸空", "浮在空中", "漂浮",
        "沒地面", "傾斜", "歪斜", "沒畫手", "沒畫腳", "沒有手", "沒有嘴巴",
    ),
    "loneliness": (
        "獨自", "一個人", "角落", "沒人", "沒有朋友", "孤單", "轉學", "新班級",
        "被排擠", "遠遠的", "空空", "自己待著", "背對", "隔很遠", "沒畫臉", "省略",
    ),
    "joy": (
        "笑", "玩", "好朋友", "朋友", "遊戲夥伴", "隊友", "家人", "寵物", "小狗", "小貓",
        "太陽", "花", "愛心", "禮物", "彩虹", "糖果", "派對", "遊樂園", "蛋糕", "開心",
        "梯子", "工具", "幫手", "小幫手", "伸手", "摘", "果實", "拿到了", "翅膀", "分享",
    ),
    "calm": (
        "安靜", "休息", "睡覺", "家", "樹", "山", "海", "散步", "平靜", "安心",
        "安全", "溫暖", "看書", "風景", "自然",
        "雨傘", "傘", "雨衣", "屋簷", "大樹", "遮蔽", "鳥巢", "巢", "秘密基地",
        "基地", "帳篷", "床", "耳機", "棉被", "被子", "毛毯", "玩偶", "娃娃",
        "地面", "腳踏實地", "草地", "穩固", "抱抱", "牽手", "照顧",
    ),
}

# Contemporary 3C / pop-culture / gaming cues (Ivashkevich, 2009)
POP_CULTURE_CUES: Tuple[str, ...] = (
    "怪物", "怪獸", "尖牙", "大牙齒", "利爪", "尖爪", "武器", "刀", "劍", "槍", "雷射",
    "恐龍", "暴龍", "動漫", "漫畫", "卡通", "遊戲", "電玩", "影片", "youtube", "超人",
    "英雄", "寶可夢", "麥塊", "minecraft", "薩爾達", "奧特曼", "機器人", "魔王", "boss",
)

# Neurodiversity / mechanical / repetitive-detail cues (Papangelo et al., 2020; Kotroni et al., 2018)
NEURODIVERSITY_CUES: Tuple[str, ...] = (
    "機械", "齒輪", "零件", "火車", "軌道", "捷運", "高鐵", "地圖", "路線圖", "交通工具",
    "幾何", "重複", "對稱", "規律", "數字", "字母", "自閉", "亞斯", "asd", "星兒", "神經多樣",
)

# Image-first (non-color) peer-reviewed empirical frameworks (v3.4 / v3.5)
IMAGE_FIRST_FRAMEWORK_RULES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "dapr_coping",
        "label": "壓力源 vs. 保護因應物件（雨中人 DAPR × 當代日常資源）",
        "citation": "Willis, Joy, & Kaiser (2010), The Arts in Psychotherapy, 37(3), 233–239",
        "keywords": (
            "雨傘", "傘", "雨衣", "屋簷", "遮蔽", "保護罩", "盾", "耳機", "棉被", "被子", "毛毯", "寵物",
            "下雨", "淋濕", "烏雲", "閃電", "積水", "考試卷", "考卷", "分數", "成績", "時鐘", "鬧鐘", "作業", "螢幕",
        ),
        "isResource": True,
        "summary": "依 Willis 等人（2010）雨中人實證，對照畫面中的壓力源（雨量／烏雲／考試卷／時鐘／作業／螢幕）與保護資源物件（雨傘／遮蔽物／耳機／棉被／寵物），可溫和理解孩子的知覺壓力與自我調節資源（r = .43, p < .01）。",
        "promptTip": "「如果想幫畫裡的主角再多加一樣能擋雨、隔開吵雜或讓他覺得安心的小東西（像雨傘、耳機或棉被），你會想畫在哪裡？」",
    },
    {
        "id": "kfd_boundaries",
        "label": "包覆圈、分隔線與人際阻隔物（KFD / 依附表徵）",
        "citation": "Wegmann & Lusebrink (2000), Arts in Psychotherapy; Fury et al. (1997), Child Development; Pace et al. (2022)",
        "keywords": ("框框", "圈起來", "包圍", "牆", "圍牆", "隔開", "隔線", "柵欄", "籠", "背對", "隔很遠", "角落", "遠遠的", "懸空", "浮在空中", "漂浮", "地面", "腳踏實地"),
        "isResource": False,
        "summary": "依 Wegmann & Lusebrink（2000）與 Fury 等人（1997）／Pace 等人（2022）研究，包覆圈（Encapsulation）、分隔線、阻隔物、人物背對或懸空無地面線皆為重要純圖像關係線索，解讀時須採整體構圖脈絡而非單一符號斷言。",
        "promptTip": "「這條線／這個框框把裡外分開了，對畫裡的主角來說，這裡是保護他的秘密空間，還是擋住了什麼呢？」",
    },
    {
        "id": "bnd_safe_base",
        "label": "安全基地與承接容器結構（鳥巢畫 BND）",
        "citation": "Kaiser (1996), Arts in Psychotherapy; Kaiser & Deaver (2009), Art Therapy, 26(1), 26–33",
        "keywords": ("鳥巢", "巢", "秘密基地", "基地", "帳篷", "床", "棉被", "保護殼", "空巢", "無底", "穩固", "照顧", "抱抱"),
        "isResource": True,
        "summary": "依 Kaiser（1996）與 Kaiser & Deaver（2009）鳥巢畫十年實證回顧，容器底部是否穩固、有無支撐環境與內部保護感（如鳥巢、帳篷、棉被、秘密基地），可溫和反映孩子的內在安全基地感。",
        "promptTip": "「這個秘密基地／小窩住起來穩不穩固？裡面誰（或哪隻寵物／夥伴）最常來陪伴主角呢？」",
    },
    {
        "id": "ppat_agency",
        "label": "主角主動性與問題解決工具（摘蘋果的人 PPAT）",
        "citation": "Bat Or, Ishai, Barkay, & Shalev (2022), Children, 9(9), 1387",
        "keywords": ("梯子", "工具", "幫手", "小幫手", "遊戲夥伴", "隊友", "伸手", "摘", "果實", "拿到了", "翅膀", "分享"),
        "isResource": True,
        "summary": "依 Bat Or 等人（2022）PPAT 與兒童敘事實證，主角主動伸手或運用梯子／工具／小幫手與遊戲夥伴克服困難，顯著關聯自我效能感（Self-potency）與內在韌性資源。",
        "promptTip": "「主角用這個方法／找來的幫手好有創意！他拿到想要的東西或完成挑戰後，接下來最想做什麼？」",
    },
    {
        "id": "skybo_hfd_norms",
        "label": "人物部位省略、誇大姿態與年齡發展檢核（HFD）",
        "citation": "Skybo, Ryan-Wenger, & Su (2007), Journal of Pediatric Nursing, 22(1), 15–28",
        "keywords": ("沒畫手", "沒畫腳", "沒有手", "沒有嘴巴", "沒畫臉", "省略", "牙齒", "大牙齒", "尖牙", "利爪", "尖爪", "交叉雙臂", "傾斜", "歪斜"),
        "isResource": False,
        "summary": "依 Skybo 等人（2007）人物畫實務批判回顧，省略手腳／五官或傾斜姿態常與退縮無力相伴，誇大牙齒／利爪／武裝則與防衛張力相關，但必先對照年齡發展階段（3–6 歲不扣分）與流行文化模仿，並搭配訪談佐證。",
        "promptTip": "「畫裡這個角色擺出這個姿勢／留白的地方，如果他現在要行動或開口，他最需要什麼幫忙？」",
    },
    {
        "id": "pop_culture_media",
        "label": "3C 流行文化與媒體角色模仿去偏",
        "citation": "Ivashkevich (2009), Studies in Art Education, 51(1), 50–63",
        "keywords": POP_CULTURE_CUES,
        "isResource": False,
        "summary": "依 Ivashkevich（2009）社會文化繪畫研究，兒童繪製怪物、尖牙、武器或誇張動漫／遊戲角色常為同儕文化交流與媒體角色模仿，輔導時應先檢核來源，避免直接誤判為攻擊性或創傷。",
        "promptTip": "「這個角色（或武器／怪物）好有特色！它是你自己發明的，還是你在影片、漫畫或遊戲裡看過的呢？」",
    },
    {
        "id": "neurodiversity_asd",
        "label": "神經多樣性（ASD）繪畫風格護欄",
        "citation": "Papangelo et al. (2020), Brain Sciences, 10(6), 398; Kotroni et al. (2018), IJDD, 65(4), 248–256",
        "keywords": NEURODIVERSITY_CUES,
        "isResource": False,
        "summary": "依 Papangelo 等人（2020）與 Kotroni 等人（2018）實證，自閉症譜系（ASD）兒童常偏好機械、交通工具、結構細節或重複幾何，且使用較少外顯視覺情緒線索，傳統 HFD 不可直接套用，也不應誤判為情緒冷漠或依附疏離。",
        "promptTip": "「你把這裡的結構／機械細節畫得好仔細！可以跟我介紹一下它是怎麼運作的、你最喜歡哪一個部分嗎？」",
    },
)

ZONE_3X3_LABELS: Tuple[str, ...] = ("左上", "正上", "右上", "正左", "中央", "正右", "左下", "正下", "右下")


def _zone_label_for_point(x: float, y: float, width: float, height: float) -> str:
    col = min(2, max(0, int((x / max(1.0, width)) * 3)))
    row = min(2, max(0, int((y / max(1.0, height)) * 3)))
    return ZONE_3X3_LABELS[row * 3 + col]


def _extract_process_highlights(
    raw_highlights: Any,
    stroke_prims: List[Dict[str, Any]],
    draw_strokes: List[Dict[str, Any]],
    eraser_strokes: List[Dict[str, Any]],
    width: float,
    height: float,
) -> Dict[str, Any]:
    """Merge client-recorded drawing process highlights with stroke-derived fallbacks."""
    rh = raw_highlights if isinstance(raw_highlights, dict) else {}

    def _clean_box(b: Any) -> Optional[Dict[str, float]]:
        if not isinstance(b, dict):
            return None
        w_val = _clamp(_safe_float(b.get("w"), 0.0), 0.0, width * 2.0)
        h_val = _clamp(_safe_float(b.get("h"), 0.0), 0.0, height * 2.0)
        if w_val <= 1.0 or h_val <= 1.0:
            return None
        return {
            "x": round(_clamp(_safe_float(b.get("x"), 0.0), 0.0, width), 1),
            "y": round(_clamp(_safe_float(b.get("y"), 0.0), 0.0, height), 1),
            "w": round(w_val, 1),
            "h": round(h_val, 1),
        }

    # 1. Repeat / dense redraw hotspot
    repeat_hotspot: Optional[Dict[str, Any]] = None
    raw_rep = rh.get("repeatHotspot")
    if isinstance(raw_rep, dict) and str(raw_rep.get("zone") or "").strip():
        repeat_hotspot = {
            "zone": str(raw_rep.get("zone")).strip()[:24],
            "strokeCount": max(1, int(_safe_float(raw_rep.get("strokeCount"), 1))),
            "bbox": _clean_box(raw_rep.get("bbox")),
        }
    elif len(stroke_prims) >= 3:
        zone_buckets: Dict[str, List[Dict[str, Any]]] = {}
        for p in stroke_prims:
            z = _zone_label_for_point(p["cx"], p["cy"], width, height)
            zone_buckets.setdefault(z, []).append(p)
        top_zone, top_items = max(zone_buckets.items(), key=lambda kv: len(kv[1]))
        if len(top_items) >= 3:
            bx0 = min(i["minX"] for i in top_items)
            by0 = min(i["minY"] for i in top_items)
            bx1 = max(i["maxX"] for i in top_items)
            by1 = max(i["maxY"] for i in top_items)
            repeat_hotspot = {
                "zone": top_zone,
                "strokeCount": len(top_items),
                "bbox": {
                    "x": round(bx0, 1),
                    "y": round(by0, 1),
                    "w": round(max(24.0, bx1 - bx0), 1),
                    "h": round(max(24.0, by1 - by0), 1),
                },
            }

    # 2. Revision / undo / eraser hotspot
    revision_hotspot: Optional[Dict[str, Any]] = None
    raw_rev = rh.get("revisionHotspot")
    if isinstance(raw_rev, dict) and str(raw_rev.get("zone") or "").strip():
        revision_hotspot = {
            "zone": str(raw_rev.get("zone")).strip()[:24],
            "count": max(1, int(_safe_float(raw_rev.get("count"), 1))),
            "bbox": _clean_box(raw_rev.get("bbox")),
        }
    elif len(eraser_strokes) >= 1:
        er_zones: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for es in eraser_strokes:
            pts = es.get("points") or []
            if not pts:
                continue
            xs = [_safe_float(pt.get("x"), 0.0) for pt in pts if isinstance(pt, dict)]
            ys = [_safe_float(pt.get("y"), 0.0) for pt in pts if isinstance(pt, dict)]
            if not xs or not ys:
                continue
            ex0, ex1, ey0, ey1 = min(xs), max(xs), min(ys), max(ys)
            ez = _zone_label_for_point((ex0 + ex1) * 0.5, (ey0 + ey1) * 0.5, width, height)
            er_zones.setdefault(ez, []).append((ex0, ey0, ex1, ey1))
        if er_zones:
            ez_top, ez_boxes = max(er_zones.items(), key=lambda kv: len(kv[1]))
            bx0 = min(b[0] for b in ez_boxes)
            by0 = min(b[1] for b in ez_boxes)
            bx1 = max(b[2] for b in ez_boxes)
            by1 = max(b[3] for b in ez_boxes)
            revision_hotspot = {
                "zone": ez_top,
                "count": len(ez_boxes),
                "bbox": {
                    "x": round(bx0, 1),
                    "y": round(by0, 1),
                    "w": round(max(28.0, bx1 - bx0), 1),
                    "h": round(max(28.0, by1 - by0), 1),
                },
            }

    # 3. Drawing sequence (first stroke vs last stroke)
    sequence: Optional[Dict[str, Any]] = None
    raw_seq = rh.get("sequence")
    if isinstance(raw_seq, dict) and str(raw_seq.get("firstZone") or "").strip():
        sequence = {
            "firstZone": str(raw_seq.get("firstZone")).strip()[:24],
            "lastZone": str(raw_seq.get("lastZone") or raw_seq.get("firstZone")).strip()[:24],
            "totalDrawStrokes": max(1, int(_safe_float(raw_seq.get("totalDrawStrokes"), len(stroke_prims)))),
        }
    elif len(stroke_prims) >= 2:
        first_p, last_p = stroke_prims[0], stroke_prims[-1]
        sequence = {
            "firstZone": _zone_label_for_point(first_p["cx"], first_p["cy"], width, height),
            "lastZone": _zone_label_for_point(last_p["cx"], last_p["cy"], width, height),
            "totalDrawStrokes": len(stroke_prims),
        }

    # 4. Tempo / tool-size transition (long pause > 2.5s OR notable tool/size shift)
    tempo_shift: Optional[Dict[str, Any]] = None
    raw_ts = rh.get("tempoShift")
    if isinstance(raw_ts, dict) and str(raw_ts.get("type") or "").strip():
        tempo_shift = {
            "type": str(raw_ts.get("type")).strip()[:32],
            "zone": str(raw_ts.get("zone") or "").strip()[:24],
            "pauseSec": round(_safe_float(raw_ts.get("pauseSec"), 0.0), 1) if raw_ts.get("pauseSec") is not None else None,
            "detail": str(raw_ts.get("detail") or "").strip()[:80],
            "strokeIdx": max(0, int(_safe_float(raw_ts.get("strokeIdx"), 0))),
        }
    elif len(draw_strokes) >= 2 and len(stroke_prims) == len(draw_strokes):
        tool_zh_map = {"marker": "麥克筆", "crayon": "蠟筆", "pencil": "鉛筆", "watercolor": "水彩"}
        best_pause = 0.0
        best_pause_info = None
        best_shift_info = None
        for idx in range(1, len(draw_strokes)):
            prev_s, cur_s = draw_strokes[idx - 1], draw_strokes[idx]
            prev_pts = prev_s.get("points") or []
            cur_pts = cur_s.get("points") or []
            if prev_pts and cur_pts:
                t_prev = _safe_float(prev_pts[-1].get("t"), 0.0)
                t_cur = _safe_float(cur_pts[0].get("t"), 0.0)
                gap_ms = t_cur - t_prev
                if gap_ms > 2500.0 and gap_ms > best_pause and gap_ms < 600_000.0:
                    best_pause = gap_ms
                    best_pause_info = {
                        "type": "long_pause",
                        "zone": _zone_label_for_point(stroke_prims[idx]["cx"], stroke_prims[idx]["cy"], width, height),
                        "pauseSec": round(gap_ms / 1000.0, 1),
                        "detail": "",
                        "strokeIdx": idx,
                    }
            prev_size = _safe_float(prev_s.get("size"), 7.0)
            cur_size = _safe_float(cur_s.get("size"), 7.0)
            prev_tool = str(prev_s.get("tool") or "crayon")
            cur_tool = str(cur_s.get("tool") or "crayon")
            if best_shift_info is None and (abs(cur_size - prev_size) >= 7.0 or cur_tool != prev_tool):
                shift_desc = (
                    "更粗的筆觸" if cur_size - prev_size >= 7.0
                    else ("更細的筆觸" if prev_size - cur_size >= 7.0 else tool_zh_map.get(cur_tool, cur_tool))
                )
                best_shift_info = {
                    "type": "tool_size_shift",
                    "zone": _zone_label_for_point(stroke_prims[idx]["cx"], stroke_prims[idx]["cy"], width, height),
                    "pauseSec": None,
                    "detail": shift_desc,
                    "strokeIdx": idx,
                }
        tempo_shift = best_pause_info or best_shift_info

    return {
        "repeatHotspot": repeat_hotspot,
        "revisionHotspot": revision_hotspot,
        "sequence": sequence,
        "tempoShift": tempo_shift,
    }


def _match_image_first_frameworks(texts: List[str]) -> List[Dict[str, Any]]:
    """Match pure image-centric (non-color) empirical literature rules against text cues."""
    joined = " ".join(t for t in texts if t).lower()
    if not joined:
        return []
    matched: List[Dict[str, Any]] = []
    for rule in IMAGE_FIRST_FRAMEWORK_RULES:
        hits = [kw for kw in rule["keywords"] if kw in joined]
        if hits:
            matched.append({
                "id": rule["id"],
                "label": rule["label"],
                "citation": rule["citation"],
                "hits": hits[:5],
                "isResource": rule["isResource"],
                "summary": rule["summary"],
                "promptTip": rule["promptTip"],
            })
    return matched


def _detect_safety_alert(texts: List[str]) -> Optional[Dict[str, Any]]:
    """Detect urgent self-harm or abuse cues and return a formal referral guardrail."""
    joined = " ".join(t for t in texts if t).lower()
    if not joined:
        return None
    hits = [kw for kw in CRISIS_KEYWORDS if kw in joined]
    if not hits:
        return None
    return {
        "level": "urgent",
        "matchedCount": len(hits),
        "title": "高風險安全與專業轉介提醒（倫理護欄）",
        "message": (
            "系統在文字或補充訪談脈絡中偵測到可能涉及自傷、受暴或立即安全疑慮的關鍵線索。"
            "繪畫分析工具絕不可取代正式危機評估或醫療處遇，請輔導人員立即依所屬學校／機構之危機處理與兒少保護通報流程進行安全評估，並連結專業資源。"
        ),
        "resources": [
            "衛福部安心專線：1925（24 小時免付費心理諮詢）",
            "生命線協談專線：1995",
            "婦幼／兒少保護專線：113（24 小時）",
            "緊急危難報案：110 ／ 救護：119，並依規定通報校安或社政單位",
        ],
    }


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


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _resample_points(
    points: List[Dict[str, Any]], step: float = 12.0, max_points: int = 800
) -> List[Tuple[float, float]]:
    if not points:
        return []
    step = max(1.0, _safe_float(step, 12.0))
    valid_pts: List[Tuple[float, float]] = []
    for pt in points:
        if not isinstance(pt, dict):
            continue
        px = _safe_float(pt.get("x"), float("nan"))
        py = _safe_float(pt.get("y"), float("nan"))
        if math.isfinite(px) and math.isfinite(py):
            valid_pts.append((px, py))
    if not valid_pts:
        return []
    out: List[Tuple[float, float]] = [valid_pts[0]]
    acc = 0.0
    prev_x, prev_y = valid_pts[0]
    for cx, cy in valid_pts[1:]:
        if len(out) >= max_points:
            break
        dx, dy = cx - prev_x, cy - prev_y
        dist = math.hypot(dx, dy)
        if not math.isfinite(dist) or dist < 1e-6:
            continue
        while acc + dist >= step and len(out) < max_points:
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
    last = valid_pts[-1]
    if len(out) < max_points and math.hypot(last[0] - out[-1][0], last[1] - out[-1][1]) > step * 0.35:
        out.append(last)
    return out


def _rdp_simplify(
    points: List[Tuple[float, float]], epsilon: float = 5.5
) -> List[Tuple[float, float]]:
    """Iterative Ramer-Douglas-Peucker simplification (safe against recursion limits)."""
    n = len(points)
    if n <= 2:
        return list(points)
    keep = [False] * n
    keep[0] = True
    keep[-1] = True
    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        start_i, end_i = stack.pop()
        if end_i - start_i <= 1:
            continue
        x1, y1 = points[start_i]
        x2, y2 = points[end_i]
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        max_dist = -1.0
        max_idx = start_i
        for i in range(start_i + 1, end_i):
            px, py = points[i]
            d = (
                math.hypot(px - x1, py - y1)
                if seg_len < 1e-6
                else abs(dy * px - dx * py + x2 * y1 - y2 * x1) / seg_len
            )
            if d > max_dist:
                max_dist = d
                max_idx = i
        if max_dist > epsilon and start_i < max_idx < end_i:
            keep[max_idx] = True
            stack.append((start_i, max_idx))
            stack.append((max_idx, end_i))
    return [pt for pt, flag in zip(points, keep) if flag]


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


_ALLOWED_IMAGE_PREFIXES = (
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/jpg;base64,",
    "data:image/webp;base64,",
)


def _sanitize_pixel_stats(raw: Any) -> Optional[Dict[str, Any]]:
    """Defensively validate untrusted clientImageStats payloads."""
    if not isinstance(raw, dict):
        return None
    cov = _safe_float(raw.get("inkCoverageRatio"), -1.0)
    sat = _safe_float(raw.get("meanSaturation"), -1.0)
    lum = _safe_float(raw.get("meanLuminance"), -1.0)
    grid = raw.get("grid3x3")
    hist = raw.get("colorHistogram")
    if not (0.0 <= cov <= 1.0 and 0.0 <= sat <= 1.0 and 0.0 <= lum <= 1.0):
        return None
    if not isinstance(grid, (list, tuple)) or len(grid) != 9:
        return None
    clean_grid: List[float] = []
    for cell in grid:
        v = _safe_float(cell, -1.0)
        if not (0.0 <= v <= 1.0):
            return None
        clean_grid.append(v)
    if not isinstance(hist, dict) or len(hist) > 16:
        return None
    clean_hist: Dict[str, float] = {}
    for k, v in hist.items():
        if not isinstance(k, str) or len(k) > 32:
            continue
        fv = _safe_float(v, -1.0)
        if 0.0 <= fv <= 1.0:
            clean_hist[k] = fv
    if not clean_hist:
        return None
    out: Dict[str, Any] = {
        "inkCoverageRatio": cov,
        "grid3x3": clean_grid,
        "colorHistogram": clean_hist,
        "meanSaturation": sat,
        "meanLuminance": lum,
    }
    center = raw.get("pixelCenter")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        cx = _safe_float(center[0], -1.0)
        cy = _safe_float(center[1], -1.0)
        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
            out["pixelCenter"] = (cx, cy)
    return out


def _analyze_image_buffer(
    image_data_url: str, bg_hex: str
) -> Optional[Dict[str, Any]]:
    if (
        not isinstance(image_data_url, str)
        or len(image_data_url) > 6_000_000
        or not image_data_url.lower().startswith(_ALLOWED_IMAGE_PREFIXES)
    ):
        return None
    try:
        _, b64_part = image_data_url.split(",", 1)
        raw = base64.b64decode(b64_part, validate=True)
        if len(raw) > 4_500_000:
            return None
        with Image.open(io.BytesIO(raw)) as opened:
            if (
                opened.width <= 0
                or opened.height <= 0
                or opened.width > 4096
                or opened.height > 4096
                or opened.width * opened.height > 4_000_000
            ):
                return None
            img = opened.convert("RGB")
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
    prims: List[Dict[str, Any]],
    width: float,
    height: float,
    omitted_detail_scale: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Assemble multi-stroke spatial topology into complete figures & expressions.

    Returns (recognized_figures, emotion_boosts_dict).
    Each figure includes a bounding box [x, y, w, h] for optional canvas overlay
    and its empirical expressive channel ('literal' vs. 'content').
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
        channel: str = "content",
    ) -> None:
        found_codes.add(code)
        x0, y0, x1, y1 = bbox
        figures.append({
            "code": code,
            "label": label,
            "category": category,
            "channel": channel,
            "theory": theory,
            "meaning": meaning,
            "impact": {emo: round(delta, 2) for emo, delta in emo_delta.items()},
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
                    "正向情境象徵",
                    "Jolley et al. (2004) 情境正向符號",
                    "畫面描繪了愛心圖案，直接表達善意、被愛的需求、珍惜或溫暖的情感連結。",
                    (p["minX"], p["minY"], p["maxX"], p["maxY"]),
                    {"joy": 0.68, "calm": -0.06, "anger": -0.14, "sadness": -0.08},
                    channel="content",
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
                "Jolley et al. (2004); Picard & Gauthier (2012) 正向情境隱喻",
                "辨識出圓體與周圍放射光線構成的太陽（不限色彩），象徵內心的溫暖、希望感、正向能量或渴望被照亮。",
                (min(xs), min(ys), max(xs), max(ys)),
                {"joy": 0.36, "calm": 0.14, "sadness": -0.14},
                channel="content",
            )
            used_face_ids.add(p["idx"])
            break

    # -------------------------------------------------------------------------
    # 3. FACE & FACIAL EXPRESSION RECOGNITION
    #    Aligned with Picard et al. (2007) & Nanayama Tanaka et al. (2020) FACED
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
                channel="literal",
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
            continue

        eyes_left = 0
        eyes_right = 0
        angry_left_brow = False
        angry_right_brow = False
        has_smile = False
        has_open_rounded_smile = False
        smile_width_ratio = 0.0
        has_frown = False
        has_jagged_mouth = False
        has_tense_flat_mouth = False
        has_any_mouth_stroke = False
        tear_count = 0

        for q in inner:
            rel_x = (q["cx"] - face["minX"]) / max(1.0, face["w"])
            rel_y = (q["cy"] - face["minY"]) / max(1.0, face["h"])

            # Upper-half features: eyes or angry furrowed brows
            if 0.10 <= rel_y <= 0.58:
                if q["isStraight"] and q["w"] >= face["w"] * 0.08 and q["w"] <= face["w"] * 0.45:
                    dx = q["x1"] - q["x0"]
                    dy = q["y1"] - q["y0"]
                    if abs(dx) > 4.0:
                        slope = dy / dx
                        if rel_x < 0.48 and slope > 0.22:
                            angry_left_brow = True
                        elif rel_x > 0.52 and slope < -0.22:
                            angry_right_brow = True

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

            # Lower-half mouth zone (centered horizontally; Nanayama Tanaka et al., 2020)
            if 0.48 <= rel_y <= 0.92 and 0.22 <= rel_x <= 0.78 and q["w"] >= face["w"] * 0.12:
                has_any_mouth_stroke = True
                mw_ratio = q["w"] / max(1.0, face["w"])
                if q["sharpTurns"] >= 2 and q["sharpRatio"] > 0.35:
                    has_jagged_mouth = True
                elif q["arcSagRatio"] >= 0.13:
                    # +y is downward -> center dips below endpoints -> SMILE U-arc
                    has_smile = True
                    smile_width_ratio = max(smile_width_ratio, mw_ratio)
                elif q["isClosed"] and not q["isScribble"] and mw_ratio >= 0.16 and q["h"] >= face["h"] * 0.08 and (q["w"] / max(1.0, q["h"])) <= 3.0:
                    # Enclosed wide/rounded open mouth (Nanayama Tanaka et al., 2020: lower negative mood)
                    has_open_rounded_smile = True
                    smile_width_ratio = max(smile_width_ratio, mw_ratio)
                elif q["arcSagRatio"] <= -0.13:
                    # Center arches above endpoints -> FROWN inverted-U arc
                    has_frown = True
                elif q["isStraight"] and q["w"] > q["h"] * 2.2:
                    has_tense_flat_mouth = True

        # Check tears slightly BELOW the face boundary too
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
                "字面表情辨識",
                "Picard et al. (2007); Koppitz (1968) 憤怒眉眼特徵",
                "辨識出圓形臉孔內帶有向內斜壓的八字眉或緊繃鋸齒嘴型，直接呈現憤怒、不滿、委屈抗議或內在高度防衛情緒。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"anger": 0.56, "anxiety": 0.16, "joy": -0.32, "calm": -0.24},
                channel="literal",
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
        elif (has_frown and tear_count >= 1) or (has_eyes and tear_count >= 2 and not has_smile and not has_open_rounded_smile):
            add_fig(
                "face_sad_crying",
                "哭泣落淚臉孔",
                "字面表情辨識",
                "Picard et al. (2007); Nanayama Tanaka et al. (2020) 悲傷表情指標",
                "辨識出下垂嘴角與眼下淚痕／淚滴，明確指向傷心、難過、委屈想哭或亟需被溫柔安慰的心理狀態。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"sadness": 0.72, "loneliness": 0.12, "joy": -0.36, "calm": -0.26},
                channel="literal",
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
        elif has_frown and (has_eyes or len(inner) >= 2):
            add_fig(
                "face_sad",
                "嘴角下垂的難過臉孔",
                "字面表情辨識",
                "Nanayama Tanaka et al. (2020) 上凸下垂嘴弧線實驗",
                "辨識出臉孔下半部呈現明顯下垂的倒弧線嘴型（上凸弧），實證研究顯示此幾何特徵顯著對應低落、沮喪與負向情緒狀態。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"sadness": 0.48, "loneliness": 0.14, "joy": -0.30, "calm": -0.20},
                channel="literal",
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
        elif has_smile or has_open_rounded_smile:
            wide_bonus = 0.08 if smile_width_ratio >= 0.34 else 0.0
            smile_lbl = "開口圓笑開朗臉孔" if (has_open_rounded_smile and not has_smile) else "微笑開朗臉孔"
            add_fig(
                "face_happy",
                smile_lbl,
                "字面表情辨識",
                "Nanayama Tanaka et al. (2020); Picard et al. (2007) 正向嘴部幾何實驗",
                "辨識出臉孔內帶有寬廣上揚微笑弧線或圓潤開口笑形狀，依 FACED 實證研究對應正向開朗、低負向壓力與友善開放心境（不受單色畫筆影響）。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"joy": 0.68 + wide_bonus, "calm": 0.06, "sadness": -0.28, "anger": -0.24},
                channel="literal",
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
        elif has_eyes and has_tense_flat_mouth:
            add_fig(
                "face_tense_neutral",
                "抿嘴緊繃／沉默臉孔",
                "字面表情辨識",
                "Nanayama Tanaka et al. (2020); Koppitz (1968) 短直線緊閉嘴型",
                "臉孔呈現平直緊閉的水平線嘴型，實證研究指出偏短直線嘴型常見於壓抑想法、無奈、緊張或一時不知如何表達的狀態。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"anxiety": 0.24, "sadness": 0.16, "loneliness": 0.14},
                channel="literal",
            )
            detected_faces.append(face)
            used_face_ids.add(face["idx"])
        elif eyes_left >= 1 and eyes_right >= 1 and not has_any_mouth_stroke and tear_count == 0 and len(inner) <= 4:
            # Omitted mouth (Koppitz 1968; Nanayama Tanaka et al. 2020 — attenuated for ages 3-6)
            sc = omitted_detail_scale
            add_fig(
                "face_no_mouth",
                "省略嘴巴的臉孔（有眼無口）",
                "字面表情辨識",
                "Nanayama Tanaka et al. (2020); Koppitz (1968) 省略嘴巴指標",
                "圓臉內畫有雙眼但省略了嘴巴，在實證研究中常與害羞、退縮、不願開口或承受壓力有關（若為幼齡或尚未畫完，屬常見發展現象，建議結合對話理解）。",
                (face["minX"], face["minY"], face["maxX"], face["maxY"]),
                {"anxiety": 0.22 * sc, "loneliness": 0.20 * sc, "sadness": 0.10 * sc},
                channel="literal",
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
                sc = omitted_detail_scale
                add_fig(
                    "house_closed",
                    "簡易外框房屋（未繪門窗・HTP 結構）",
                    "屋樹人情境結構",
                    "Picard et al. (2007); Buck (1948) 房屋結構詞彙（限作開放對話起點）",
                    "辨識出屋頂與牆體外框（未繪門窗）。依 Buck HTP 結構詞彙僅作為開放式對話切入點，未繪門窗常見於簡化畫法、尚未畫完或偏好外觀輪廓，不直接等同封閉退縮（3–6 歲不扣分）。",
                    house_bbox,
                    {"anxiety": 0.10 * sc, "loneliness": 0.10 * sc, "calm": -0.04 * sc},
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
                "Wegmann & Lusebrink (2000) KFD; Pénzes (2020) 結構界線指標",
                "辨識出整齊排列的柵欄或格柵框線，常呼應內在感到受限、需要自我保護邊界或想整理秩序的心情（建議結合對話釐清）。",
                xb,
                {"anxiety": 0.32, "loneliness": 0.22, "calm": -0.18, "joy": -0.16},
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
                sc = max(0.55, omitted_detail_scale)
                add_fig(
                    "person_tiny_isolated",
                    "角落中的微小人物",
                    "人物尺寸空間指標",
                    "Burkitt (2003b); Cao et al. (2023) 孤單縮聚尺寸實證",
                    "辨識出縮在畫紙角落的微小人物，周圍伴隨大片留白，在跨文化與華語實證中常對應孤單、害羞退縮或渴望被看見（3–6 歲已自動降權）。",
                    (px0, py0, px1, py1),
                    {"loneliness": 0.46 * sc, "sadness": 0.18 * sc, "joy": -0.18 * sc},
                )
            elif "face_sad" not in found_codes and "face_sad_crying" not in found_codes and "face_angry" not in found_codes:
                # Check if head has zero internal strokes (blank omitted face on a larger figure)
                head_inner = [q for q in prims if q["idx"] != head["idx"] and is_inside(q, head, margin=0.05)]
                if len(head_inner) == 0 and head["diag"] >= 38.0:
                    sc = omitted_detail_scale
                    add_fig(
                        "person_blank_face",
                        "五官留白的人物輪廓",
                        "Koppitz 人物畫指標",
                        "Koppitz (1968); Skybo et al. (2007); Papangelo et al. (2020) 發展常模檢核",
                        "人物頭部保留空白未畫五官，可能源自學齡前簡化畫法、寫實前期怕畫不像、神經發展認知風格或單純勾勒遠景輪廓（3–6 歲完全不扣分）。",
                        (px0, py0, px1, py1),
                        {"loneliness": 0.22 * sc, "anxiety": 0.16 * sc},
                        channel="literal",
                    )
                else:
                    add_fig(
                        "person_figure",
                        "人物輪廓",
                        "人物畫結構",
                        "Skybo et al. (2007); Koppitz HFD 人物結構",
                        "辨識出頭身完整的人物圖形，反映創作者以自我或重要他人作為畫面的情感聚焦中心。",
                        (px0, py0, px1, py1),
                        {"calm": 0.10, "joy": 0.08},
                    )
            break

    return figures, boosts


def analyze_drawing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run v3.5 Empirical Image-First + 3C/Neurodiversity Debiased psychology analysis."""
    canvas_info = payload.get("canvas") or {}
    width = _clamp(_safe_float(canvas_info.get("width"), 800.0), 50.0, 4096.0)
    height = _clamp(_safe_float(canvas_info.get("height"), 550.0), 50.0, 4096.0)
    bg_hex = str(canvas_info.get("backgroundColor") or "#FDFBF7")
    canvas_area = max(1.0, width * height)

    all_strokes: List[Dict[str, Any]] = payload.get("strokes") or []
    draw_strokes = [
        s for s in all_strokes if isinstance(s, dict) and s.get("tool") != "eraser" and len(s.get("points") or []) >= 1
    ]
    eraser_strokes = [s for s in all_strokes if isinstance(s, dict) and s.get("tool") == "eraser"]

    process = payload.get("processMetrics") or {}
    undo_count = max(0, int(_safe_float(process.get("undoCount"), 0)))
    eraser_count = max(max(0, int(_safe_float(process.get("eraserStrokeCount"), 0))), len(eraser_strokes))
    total_duration_ms = max(0.0, _safe_float(process.get("totalDurationMs"), 0.0))
    active_drawing_ms = max(0.0, _safe_float(process.get("activeDrawingMs"), 0.0))
    pause_count = max(0, int(_safe_float(process.get("pauseCount"), 0)))
    color_change_count = max(0, int(_safe_float(process.get("colorChangeCount"), 0)))
    tool_switch_count = max(0, int(_safe_float(process.get("toolSwitchCount"), 0)))
    size_change_count = max(0, int(_safe_float(process.get("sizeChangeCount"), 0)))
    paper_change_count = max(0, int(_safe_float(process.get("paperChangeCount"), 0)))
    reported_devices = [str(d)[:16] for d in (process.get("inputDevices") or []) if isinstance(d, str)]

    user_context = payload.get("userContext") or {}
    title_or_note = str(user_context.get("titleOrNote") or "").strip()[:300]
    raw_age_group = str(user_context.get("ageGroup") or "unspecified").strip()
    if raw_age_group == "11+":
        raw_age_group = "11-13"
    age_group = raw_age_group if raw_age_group in AGE_GROUP_NORMS else "unspecified"
    age_norm = AGE_GROUP_NORMS[age_group]
    omitted_detail_scale = float(age_norm["omittedDetailScale"])
    selected_motifs: List[str] = [str(m)[:48] for m in list(user_context.get("selectedMotifs") or [])[:20]]

    raw_counselor = payload.get("counselorContext") or {}
    if not isinstance(raw_counselor, dict):
        raw_counselor = {}
    counselor_verified_objects: List[str] = [
        str(obj).strip()[:80]
        for obj in list(raw_counselor.get("verifiedObjects") or [])[:15]
        if isinstance(obj, str) and str(obj).strip()
    ]
    counselor_follow_up_notes: str = str(raw_counselor.get("followUpNotes") or "").strip()[:600]
    has_counselor_context = bool(counselor_verified_objects or counselor_follow_up_notes)
    safety_alert = _detect_safety_alert(
        [title_or_note, counselor_follow_up_notes, *counselor_verified_objects]
    )

    if not draw_strokes:
        return {
            "status": "empty",
            "message": "畫布目前還是空白的，請自由畫上幾筆線條或圖案後再進行心理探索分析。",
        }

    # --- 1. Extract Per-Stroke Primitives, Kinematics, Software Interface Settings & Grid ---
    total_drawn_ink_area = 0.0
    velocities: List[float] = []
    pressures: List[float] = []
    tilts: List[float] = []
    contact_widths: List[float] = []
    widths: List[float] = []
    raw_sizes: List[float] = []
    tool_hardness_vals: List[float] = []
    tool_counts: Dict[str, int] = {}
    pointer_types: Dict[str, int] = {}
    jitter_ratios: List[float] = []
    inferred_active_ms = 0.0

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
        raw_pts = stroke.get("points") or []
        pts = [
            {
                "x": _clamp(_safe_float(p.get("x"), 0.0), -5000.0, 10000.0),
                "y": _clamp(_safe_float(p.get("y"), 0.0), -5000.0, 10000.0),
                "t": max(0.0, _safe_float(p.get("t"), 0.0)),
                "p": _clamp(_safe_float(p.get("p"), 0.5), 0.05, 1.0),
                "tx": _clamp(_safe_float(p.get("tx"), 0.0), -90.0, 90.0),
                "ty": _clamp(_safe_float(p.get("ty"), 0.0), -90.0, 90.0),
                "w": _clamp(_safe_float(p.get("w"), 0.0), 0.0, 200.0),
            }
            for p in raw_pts
            if isinstance(p, dict)
            and math.isfinite(_safe_float(p.get("x"), float("nan")))
            and math.isfinite(_safe_float(p.get("y"), float("nan")))
        ]
        if not pts:
            continue
        color_hex = str(stroke.get("color") or "#2A2B30").lower()
        tool = str(stroke.get("tool") or "crayon")
        p_type = str(stroke.get("pointerType") or "mouse").lower()
        if p_type not in ("mouse", "pen", "touch"):
            p_type = "mouse"
        pointer_types[p_type] = pointer_types.get(p_type, 0) + 1
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
        tool_hardness_vals.append(TOOL_HARDNESS.get(tool, 0.7))

        size = _clamp(_safe_float(stroke.get("size"), 7.0), 1.0, 80.0)
        raw_sizes.append(size)
        eff_width = size * {"pencil": 0.65, "crayon": 1.15, "marker": 1.3, "watercolor": 1.65}.get(tool, 1.0)
        widths.append(eff_width)

        r, g, b = _hex_to_rgb(color_hex)
        bucket, sat, lum = classify_rgb_bucket(r, g, b)

        s_len = 0.0
        s_min_x, s_max_x = width, 0.0
        s_min_y, s_max_y = height, 0.0
        touches_edge = False

        for idx, pt in enumerate(pts):
            x = pt["x"]
            y = pt["y"]
            pressures.append(pt["p"])
            tilt_mag = math.hypot(pt["tx"], pt["ty"])
            if tilt_mag > 0.5:
                tilts.append(min(90.0, tilt_mag))
            if pt["w"] > 1.0:
                contact_widths.append(pt["w"])
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            s_min_x, s_max_x = min(s_min_x, x), max(s_max_x, x)
            s_min_y, s_max_y = min(s_min_y, y), max(s_max_y, y)
            if x < width * 0.055 or x > width * 0.945 or y < height * 0.055 or y > height * 0.945:
                touches_edge = True
            if idx > 0:
                px = pts[idx - 1]["x"]
                py = pts[idx - 1]["y"]
                dt = max(4.0, pt["t"] - pts[idx - 1]["t"])
                seg_d = math.hypot(x - px, y - py)
                s_len += seg_d
                if seg_d > 0.5 and dt < 1500.0:
                    velocities.append(min(4.5, seg_d / dt))

        if len(pts) >= 2:
            stroke_dt = max(0.0, pts[-1]["t"] - pts[0]["t"])
            if 10.0 <= stroke_dt <= 120_000.0:
                inferred_active_ms += stroke_dt

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
        # Micro-tremor / jitter: ratio of raw high-frequency path length vs resampled smooth path
        if len(pts) >= 5 and len(resamp) >= 3 and s_len > 25.0:
            smooth_len = sum(
                math.hypot(resamp[i][0] - resamp[i - 1][0], resamp[i][1] - resamp[i - 1][1])
                for i in range(1, len(resamp))
            )
            if smooth_len > 10.0:
                jitter_ratios.append(_clamp((s_len / smooth_len - 1.0) / 0.25))

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
        x0, y0 = start_pt["x"], start_pt["y"]
        x1, y1 = end_pt["x"], end_pt["y"]
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

    if not stroke_prims:
        return {
            "status": "empty",
            "message": "畫布目前沒有有效的筆跡座標，請重新繪製後再進行分析。",
        }

    # --- 2. Combine Vector Telemetry with Pixel Stats & Detect Monochrome / Canonical Color ---
    pixel_stats = _analyze_image_buffer(str(payload.get("imageDataUrl") or ""), bg_hex)
    if not pixel_stats and payload.get("clientImageStats"):
        pixel_stats = _sanitize_pixel_stats(payload.get("clientImageStats"))

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

    # Monochrome / Single-Pencil Debiasing check (Picard & Gauthier, 2012)
    meaningful_colors = [
        hx for hx, area in color_ink_by_hex.items() if area / max(1.0, total_drawn_ink_area) >= 0.05
    ]
    is_monochrome_sketch = len(meaningful_colors) <= 1
    color_scale = 0.20 if is_monochrome_sketch else 0.50

    # --- 3. Composite Complete-Figure & Expression Recognition (Shape-First!) ---
    recognized_figures, fig_boosts = _detect_composite_figures(
        stroke_prims, width, height, omitted_detail_scale=omitted_detail_scale
    )
    fig_codes = {f["code"] for f in recognized_figures}

    # --- 3b. Software Drawing Interface Stroke Sensing & Kinematic Indices ---
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
    mean_raw_size = sum(raw_sizes) / max(1, len(raw_sizes))
    mean_tool_hardness = sum(tool_hardness_vals) / max(1, len(tool_hardness_vals))
    mean_pressure = sum(pressures) / max(1, len(pressures))
    pressure_var = sum((p - mean_pressure) ** 2 for p in pressures) / max(1, len(pressures))
    has_real_pressure_variance = math.sqrt(pressure_var) > 0.035

    all_device_set = set(pointer_types.keys()) | set(d.lower() for d in reported_devices if d.lower() in ("pen", "touch", "mouse"))
    if "pen" in all_device_set and len(all_device_set) == 1:
        sensing_mode = "pen_hardware"
        sensing_mode_label = "感壓筆硬體感測（真實筆壓＋傾斜角）"
    elif "touch" in all_device_set and "pen" not in all_device_set:
        sensing_mode = "touch_contact"
        sensing_mode_label = "觸控感測（接觸面積＋速度＋介面畫具）"
    elif len(all_device_set) > 1:
        sensing_mode = "mixed"
        sensing_mode_label = "混合裝置感測（感壓／觸控／介面設定）"
    else:
        sensing_mode = "mouse_interface"
        sensing_mode_label = "滑鼠／觸控板介面推估（畫具硬度×粗細×速度×重疊）"

    size_norm = _clamp(mean_raw_size / 22.0)
    if sensing_mode == "pen_hardware" and has_real_pressure_variance:
        tilt_bonus = _clamp((sum(tilts) / max(1, len(tilts))) / 75.0) * 0.12 if tilts else 0.0
        effective_force_index = _clamp(
            mean_pressure * 0.54 + mean_tool_hardness * 0.22 + size_norm * 0.16 + tilt_bonus
        )
    elif sensing_mode == "touch_contact" and contact_widths:
        contact_norm = _clamp((sum(contact_widths) / len(contact_widths)) / 42.0)
        effective_force_index = _clamp(
            contact_norm * 0.30 + mean_tool_hardness * 0.26 + size_norm * 0.22 + velocity_index * 0.22
        )
    else:
        # Mouse/trackpad: discard flat synthetic 0.5 pressure, derive from software interface settings
        effective_force_index = _clamp(
            mean_tool_hardness * 0.30 + size_norm * 0.28 + velocity_index * 0.24 + overlap_index * 0.18
        )

    weight_index = effective_force_index
    jitter_index = _clamp(sum(jitter_ratios) / max(1, len(jitter_ratios))) if jitter_ratios else 0.0

    eff_active_ms = active_drawing_ms if active_drawing_ms > 0 else inferred_active_ms
    if total_duration_ms > 1500.0 and eff_active_ms > 0.0 and len(draw_strokes) >= 2:
        in_air_pause_ratio = _clamp((total_duration_ms - min(total_duration_ms, eff_active_ms)) / total_duration_ms, 0.0, 0.95)
    else:
        in_air_pause_ratio = 0.0

    fragmentation_index = _clamp(
        (short_fragment_strokes / max(1, len(draw_strokes))) * 0.72
        + _clamp(pause_count / 6.0) * 0.20
        + (0.08 if in_air_pause_ratio > 0.65 else 0.0)
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

    # Crawford et al. (2012) Canonical Nature Palette Debiasing (blue sky + green grass / warm sun)
    has_sad_fig = bool(fig_codes & {"face_sad", "face_sad_crying", "rain_cloud_figure"})
    is_canonical_nature_palette = (
        not is_monochrome_sketch
        and not has_sad_fig
        and cool_blue >= 0.12
        and (fresh_green >= 0.14 or warm_bright >= 0.14)
        and (warm_bright + fresh_green) >= 0.28
    )

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
            "theory": "Koppitz (1968); Pénzes et al. (2020) 重複塗抹與張力指標",
            "meaning": "重複來回交疊的強烈筆觸，常代表內心積聚張力、煩躁、焦慮或試圖覆蓋某種不安感受。",
        })

    if "rain_cloud_figure" not in motif_codes and (
        downward_strokes >= 4 or (downward_strokes >= 3 and (cool_blue + dark_muted + neutral_grey) > 0.45)
    ):
        motif_codes.add("downward_streaks")
        detected_motifs.append({
            "code": "downward_streaks",
            "label": "向下垂落／雨絲般的線條",
            "theory": "Jolley et al. (2004); Malchiodi (1998) 動態線條方向象徵",
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
                    "theory": "Jolley et al. (2004) 情境正向光芒象徵",
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
            "theory": "Buck (1948); Pénzes et al. (2020) 邊界與結構控制指標",
            "meaning": "封閉方形或圍欄般的輪廓，反映對安全基地與保護的重視，或感受到某種框架與壓力。",
        })

    if "person_tiny_isolated" not in motif_codes and (
        bbox_area_ratio < 0.065 or (bbox_area_ratio < 0.20 and center_dist > 0.26)
    ):
        motif_codes.add("tiny_corner_cluster")
        detected_motifs.append({
            "code": "tiny_corner_cluster",
            "label": "微小／偏向一隅的構圖比例",
            "theory": "Burkitt et al. (2003b); Cao et al. (2023) 孤單縮聚尺寸實驗",
            "meaning": "主體僅佔畫紙一小角且周圍大片留白，在實證研究中與孤單、退縮、害羞或覺得自己聲音微弱相關。",
        })

    if edge_touch_ratio >= 0.30 or bbox_area_ratio > 0.80:
        motif_codes.add("edge_overflow")
        detected_motifs.append({
            "code": "edge_overflow",
            "label": "突破畫紙邊界的擴張構圖",
            "theory": "Burkitt et al. (2004); Cao et al. (2023) 高張力／高威脅擴大效應",
            "meaning": "筆觸大量延伸至畫布邊緣，顯示高漲的情緒張力、強烈表達欲或威脅防衛能量（實證指出負向高張力常呈現放大而非縮小）。",
        })

    if revision_events >= 3 or revision_index > 0.36:
        motif_codes.add("heavy_revision")
        detected_motifs.append({
            "code": "heavy_revision",
            "label": "多次復原與擦除修正歷程",
            "theory": "Mohd Shukri (2014); Price et al. (2015); Koppitz (1968) 數位歷程降權觀察",
            "meaning": "平板／觸控板上的復原與擦除常見於手指誤觸、多段調整或力求完美，系統已大幅降權其情緒扣分（約降權 56%），主要提供輔導員回放檢視與開放對話參考。",
        })

    # Dynamic Process Highlights for Replay & Open-Ended Dialogue (v3.5)
    process_highlights = _extract_process_highlights(
        payload.get("processHighlights"),
        stroke_prims,
        draw_strokes,
        eraser_strokes,
        width,
        height,
    )

    # Counselor-Assisted Image Clarification (v3.3/v3.4/v3.5 post-analysis human-in-the-loop verification)
    matched_image_frameworks = _match_image_first_frameworks([
        *counselor_verified_objects,
        counselor_follow_up_notes,
        title_or_note,
    ])
    for c_idx, obj_label in enumerate(counselor_verified_objects):
        obj_fw = _match_image_first_frameworks([obj_label])
        fw_theory = (
            obj_fw[0]["citation"]
            if obj_fw
            else "輔導人員協助辨識 × Gross & Hayne (1998) / Driessnack (2005) 脈絡驗證"
        )
        fw_meaning = (
            obj_fw[0]["summary"]
            if obj_fw
            else "由輔導人員於作畫與訪談後協助確認的實際畫中元素，優先於純幾何猜測作為脈絡判讀與延伸引導依據。"
        )
        detected_motifs.insert(c_idx, {
            "code": f"counselor_obj_{c_idx}",
            "label": f"輔導員確認：{obj_label}",
            "theory": fw_theory,
            "meaning": fw_meaning,
        })

    # Expressive channels taxonomy (Jolley 2004; Picard 2007, 2012; Misalidi & Bonoti 2014)
    expressive_channels: List[str] = []
    if any(f.get("channel") == "literal" for f in recognized_figures):
        expressive_channels.append("字面表情（Literal）")
    if any(f.get("channel") == "content" for f in recognized_figures) or counselor_verified_objects:
        expressive_channels.append("情境隱喻（Content）")
    if not expressive_channels or (sharpness_index > 0.34 or overlap_index > 0.42 or active_buckets >= 2):
        expressive_channels.append("抽象形式（Abstract）")

    # 3C Pop-Culture Detection (Ivashkevich, 2009) & Neurodiversity Guardrail Detection (Papangelo 2020; Kotroni 2018)
    all_context_blob = " ".join([title_or_note, *counselor_verified_objects, counselor_follow_up_notes]).lower()
    pop_culture_hits = [kw for kw in POP_CULTURE_CUES if kw in all_context_blob]
    if "monster_storm" in selected_motifs and "怪獸／風暴標籤" not in pop_culture_hits:
        pop_culture_hits.append("怪獸／風暴標籤")
    has_pop_culture_cues = bool(pop_culture_hits)
    pop_culture_attenuation = 0.55 if (age_group == "7-10" and has_pop_culture_cues) else 1.0

    neuro_text_hits = [kw for kw in NEURODIVERSITY_CUES if kw in all_context_blob]
    straight_count = sum(1 for p in stroke_prims if p["isStraight"])
    has_explicit_emotion_fig = any(
        f["code"] in ("face_happy", "face_sad", "face_sad_crying", "face_angry", "heart_figure", "sun_figure", "rain_cloud_figure")
        for f in recognized_figures
    )
    is_high_geometric_repeat = (
        (closed_loops >= 4 or straight_count >= 8)
        and sharpness_index < 0.32
        and not has_explicit_emotion_fig
    )
    is_sparse_emotion_structured = (
        len(draw_strokes) >= 6
        and not has_explicit_emotion_fig
        and (bool(neuro_text_hits) or straight_count >= 6 or closed_loops >= 3 or "person_blank_face" in fig_codes)
    )
    neurodiversity_triggered = bool(neuro_text_hits) or is_high_geometric_repeat or is_sparse_emotion_structured
    neurodiversity_note: Optional[str] = None
    if neurodiversity_triggered:
        neurodiversity_note = (
            "神經多樣性護欄（Papangelo et al., 2020; Kotroni et al., 2018）："
            "畫面呈現高度重複幾何、專注物件／機械細節或外顯情緒符號偏少時，請優先考量神經發展與認知風格差異"
            "（例如自閉症譜系 ASD 兒童在繪畫中常使用較少視覺情緒線索，且傳統 HFD 人物畫指標不可直接套用），切勿直接推論為情緒冷漠或依附疏離。"
        )

    # --- 5. Spontaneous Verbalization & Counselor Follow-Up Context (Gross & Hayne, 1998) ---
    verbal_scores = {k: 0.0 for k in EMOTION_KEYS}
    matched_words: Dict[str, List[str]] = {k: [] for k in EMOTION_KEYS}
    if title_or_note:
        lower_note = title_or_note.lower()
        for emo, words in VERBAL_LEXICON.items():
            hits = [w for w in words if w in lower_note]
            if hits:
                matched_words[emo] = hits
                base_v = _clamp(0.18 + 0.08 * (len(hits) - 1), 0.0, 0.36)
                if emo in ("anger", "anxiety") and pop_culture_attenuation < 1.0:
                    base_v *= pop_culture_attenuation
                verbal_scores[emo] = base_v

    counselor_scores = {k: 0.0 for k in EMOTION_KEYS}
    counselor_matched_cues: Dict[str, List[str]] = {k: [] for k in EMOTION_KEYS}
    if has_counselor_context:
        c_blob = (" ".join(counselor_verified_objects) + " " + counselor_follow_up_notes).lower()
        for emo in EMOTION_KEYS:
            combined_lex = list(dict.fromkeys([*COUNSELOR_OBJECT_CUES.get(emo, ()), *VERBAL_LEXICON.get(emo, ())]))
            c_hits = [w for w in combined_lex if w in c_blob]
            if c_hits:
                counselor_matched_cues[emo] = c_hits[:6]
                base_c = _clamp(0.18 + 0.06 * (len(c_hits) - 1), 0.0, 0.34)
                if emo in ("anger", "anxiety") and pop_culture_attenuation < 1.0:
                    base_c *= pop_culture_attenuation
                counselor_scores[emo] = base_c

    # --- 6. Shape-First + Empirical Calibrated Six-Dimension Psychological Inference (v3.5) ---
    scores: Dict[str, float] = {k: 0.10 + fig_boosts.get(k, 0.0) for k in EMOTION_KEYS}

    mono_dark_pen = 0.0 if is_monochrome_sketch else 1.0
    # Crawford (2012): attenuate cool_blue sadness weight in canonical nature scenes (blue sky + green grass)
    canonical_blue_scale = 0.38 if is_canonical_nature_palette else 1.0
    # Burkitt (2004) & Cao (2023): size asymmetry (large + high tension = threat/anger; large + smooth = positive/calm)
    large_tense_threat = (
        0.14
        if (bbox_area_ratio >= 0.48 and (sharpness_index >= 0.32 or overlap_index >= 0.42 or intense_red >= 0.18))
        else 0.0
    )
    large_open_positive = (
        0.10
        if (bbox_area_ratio >= 0.38 and sharpness_index < 0.28 and overlap_index < 0.35 and not has_sad_fig)
        else 0.0
    )

    # Alschuler & Hattwick (1947) v3.5 device split:
    # Disable pressure-to-emotion weight on mouse/trackpad & tablet touch; retain low weight only on hardware stylus with real pressure variance.
    pen_pressure_active = sensing_mode == "pen_hardware" and has_real_pressure_variance
    force_emotion_weight = 0.07 if pen_pressure_active else 0.0

    # (A) ANGER & FRUSTRATION (v3.5: overlap attenuated -55%, force weight disabled on mouse/touch)
    scores["anger"] += (
        sharpness_index * 0.34
        + velocity_index * 0.20
        + effective_force_index * force_emotion_weight
        + overlap_index * (0.09 if (sharpness_index > 0.3 or intense_red > 0.15) else 0.03)
        + edge_touch_ratio * 0.15
        + large_tense_threat
        + (0.22 if "dense_scribble_knot" in motif_codes and ((velocity_index > 0.56 and bbox_area_ratio >= 0.25) or intense_red > 0.15) else 0.0)
        + (0.14 if radiating_hub_tense else 0.0)
        + color_scale * (
            intense_red * 0.50
            + dark_muted * (0.24 if (sharpness_index > 0.28 or velocity_index > 0.55) else 0.04) * mono_dark_pen
            - (warm_bright + fresh_green) * 0.12
        )
        - smoothness_index * 0.12
    )

    # (B) SADNESS & MELANCHOLY (v3.5: Pulver vertical bottom_weight psychological inference zeroed = 0.0)
    bottom_weight = grid_3x3[6] + grid_3x3[7] + grid_3x3[8]
    downward_ratio = downward_strokes / max(1, len(draw_strokes))
    low_speed = 1.0 - velocity_index
    melancholic_cool = (cool_blue * canonical_blue_scale + neutral_grey * 0.9) * (1.25 if mean_lum < 0.52 else 0.85)
    scores["sadness"] += (
        downward_ratio * 0.34
        + low_speed * 0.15
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

    # (C) ANXIETY & TENSION (v3.5: dynamic digital process weights attenuated by ~55%-62% per Price 2015 & Mohd Shukri 2014)
    scores["anxiety"] += (
        revision_index * 0.15
        + fragmentation_index * 0.10
        + overlap_index * 0.11
        + sharpness_index * 0.15
        + jitter_index * 0.05
        + (0.03 if in_air_pause_ratio > 0.65 and len(draw_strokes) >= 4 else 0.0)
        + (0.08 if "heavy_revision" in motif_codes else 0.0)
        + (0.08 if "enclosed_shelter_or_barrier" in motif_codes and (overlap_index > 0.30 or revision_events >= 2) else 0.0)
        + (0.14 if "dense_scribble_knot" in motif_codes and intense_red < 0.25 else 0.0)
        + _clamp(pause_count / 5.0) * 0.04
        + color_scale * ((dark_muted + neutral_grey) * 0.24 * mono_dark_pen - warm_bright * 0.10)
        - smoothness_index * 0.10
    )

    # (D) LONELINESS & SEEKING CONNECTION (Burkitt 2003b; Cao 2023: loneliness specifically shrinks)
    sparsity = _clamp(1.0 - ink_coverage / 0.10)
    tiny_factor = _clamp(1.0 - bbox_area_ratio / 0.22) * (0.35 + 0.65 * center_dist)
    scores["loneliness"] += (
        tiny_factor * 0.36 * (0.65 if age_group == "3-6" else 1.0)
        + sparsity * 0.16
        + center_dist * 0.30
        + (0.22 if "tiny_corner_cluster" in motif_codes else 0.0)
        + (0.10 if (len(draw_strokes) <= 7 and bbox_area_ratio < 0.16 and center_dist > 0.24) else 0.0)
        + color_scale * ((cool_blue * canonical_blue_scale + neutral_grey) * 0.24 - intense_red * 0.14)
        - (0.14 if bbox_area_ratio > 0.55 else 0.0)
    )

    # (E) JOY & VITALITY (v3.5: overlap penalty attenuated -57% so watercolor layering/filling does not depress joy)
    generous_balanced = 1.0 - abs(bbox_area_ratio - 0.50) / 0.50
    color_richness = _clamp((active_buckets - 1) / 3.0)
    upward_ratio = upward_strokes / max(1, len(draw_strokes))
    scores["joy"] += (
        smoothness_index * 0.20
        + _clamp(generous_balanced) * 0.16
        + large_open_positive * 0.65
        + upward_ratio * 0.12
        + (0.20 if radiating_hub_warm and "sun_figure" not in motif_codes else 0.0)
        + color_scale * (
            warm_bright * 0.62
            + pink_purple * 0.42
            + fresh_green * 0.24
            + color_richness * 0.24
            + mean_lum * 0.16
            + mean_sat * 0.16
            - dark_heavy_palette * 0.28 * mono_dark_pen
        )
        - sharpness_index * 0.18
        - overlap_index * 0.06
    )

    # (F) CALM & GROUNDEDNESS (Cao 2023: Chinese '平和' is positive-valence & expansive)
    centeredness = 1.0 - center_dist
    steady_tempo = 1.0 - abs(velocity_index - 0.34) / 0.66
    low_tension = (1.0 - sharpness_index) * (1.0 - overlap_index * 0.5) * (1.0 - revision_index * 0.5)
    serene_blue = cool_blue * (
        0.46 if (is_canonical_nature_palette or (mean_lum >= 0.44 and "downward_streaks" not in motif_codes and "rain_cloud_figure" not in motif_codes)) else 0.08
    )
    scores["calm"] += (
        smoothness_index * 0.24
        + centeredness * 0.20
        + large_open_positive * 0.65
        + _clamp(steady_tempo) * 0.14
        + low_tension * 0.20
        + (1.0 - fragmentation_index) * 0.03
        + color_scale * (
            fresh_green * 0.56
            + serene_blue
            + warm_bright * 0.06
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
            adj_w = w * pop_culture_attenuation if (tag == "monster_storm" and emo in ("anger", "anxiety")) else w
            scores[emo] += adj_w
    for emo, v_score in verbal_scores.items():
        scores[emo] += v_score
    for emo, c_score in counselor_scores.items():
        scores[emo] += c_score

    # --- 6b. Multi-Evidence Convergence Guardrail (Joiner 1996; Catte & Cox 1999; Thomas & Jolley 1998) ---
    # Do not count zero-weighted omission indicators (e.g. ages 3-6) as converging distress evidence.
    active_recognized_figures = [
        f for f in recognized_figures if any(abs(v) >= 0.01 for v in (f.get("impact") or {}).values())
    ]
    converging_channels = 0
    if active_recognized_figures or counselor_verified_objects:
        converging_channels += 2 if (len(active_recognized_figures) + len(counselor_verified_objects)) >= 2 else 1
    if sharpness_index >= 0.34 or overlap_index >= 0.48 or smoothness_index >= 0.44:
        converging_channels += 1
    if bbox_area_ratio < 0.18 or edge_touch_ratio >= 0.30 or centeredness >= 0.72:
        converging_channels += 1
    if title_or_note or selected_motifs or counselor_follow_up_notes:
        converging_channels += 1
    if not is_monochrome_sketch and not is_canonical_nature_palette and active_buckets >= 2:
        converging_channels += 1

    if converging_channels >= 4 or ((active_recognized_figures or counselor_verified_objects) and converging_channels >= 3):
        conv_level = "high"
        conv_label = "高收斂（多維證據一致）"
        alt_note = "本次圖形結構、筆觸動力與構圖呈現多維度一致線索；仍建議傾聽創作者自己的故事作為最終理解（Thomas & Jolley, 1998）。"
        max_score_cap = 95
    elif converging_channels >= 2 or len(draw_strokes) >= 5:
        conv_level = "moderate"
        conv_label = "中度收斂（探索性參考）"
        alt_note = "部分特徵具情緒線索，但也可能與當下畫技習慣、平板多段短筆畫或工具探索有關，適合當作開啟對話的起點（Thomas & Jolley, 1998; Mohd Shukri, 2014）。"
        max_score_cap = 88
    else:
        conv_level = "low"
        conv_label = "低線索（高度多元可能）"
        alt_note = "目前筆跡較簡短或缺少多維收斂證據。依 Joiner (1996) 與 Catte & Cox (1999) 實證，單一線條或尺寸不可直接推斷情緒困擾，系統已自動調降極端分數。"
        max_score_cap = 66

    # v3.5 Three-Stage Age (3-13) & 3C / Neurodiversity Alternative Explanations
    if age_group == "3-6":
        alt_note += " 針對 3–6 歲學齡前：已完全停用 Koppitz 部位省略扣分（×0.00），並請優先排除平板手指誤觸與多段短筆畫操作特性（Price et al., 2015; Mohd Shukri, 2014）。"
    elif age_group == "7-10" and has_pop_culture_cues:
        alt_note += f" 偵測到 7–10 歲常見動漫／遊戲符號（{'、'.join(pop_culture_hits[:3])}），已依 Ivashkevich (2009) 自動降權負向張力，建議先以開放問題確認是否源自影片或遊戲角色。"
    elif age_group == "11-13":
        has_preadolescent_style = (
            bool(fig_codes & {"person_blank_face", "face_no_mouth", "person_tiny_isolated", "house_closed"})
            or len(draw_strokes) <= 8
            or revision_events >= 3
            or is_monochrome_sketch
            or dark_muted >= 0.35
            or has_pop_culture_cues
        )
        if has_preadolescent_style:
            alt_note += " 針對 11–13 歲寫實前期：極簡火柴人、細節省略、頻繁復原或暗黑酷炫畫風，常源於「怕畫不像的畫技顧慮」或「青少年前期同儕審美風格」（Lowenfeld, 1987; Ivashkevich, 2009），不應直接解讀為負面困擾。"
    if neurodiversity_note:
        alt_note += " 另已觸發神經多樣性護欄（Papangelo et al., 2020; Kotroni et al., 2018），請勿將偏重幾何／物件細節或較少表情線索誤判為冷漠疏離。"

    clamped = {k: max(0.03, v) for k, v in scores.items()}
    max_val = max(clamped.values())
    total_val = sum(clamped.values())

    emotion_results = []
    for k in EMOTION_KEYS:
        rel_pct = round((clamped[k] / total_val) * 100)
        dominance = clamped[k] / max(0.45, max_val)
        intensity = _clamp(clamped[k] / 1.35, 0.0, 1.0)
        raw_strength = dominance * 56 + intensity * 38
        if conv_level == "low":
            raw_strength = min(raw_strength, max_score_cap)
        strength = int(round(_clamp(raw_strength, 6, max_score_cap)))
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

    # --- 6c. Formal Structure Balance & Positive Strengths (Pénzes et al., 2020) ---
    if overlap_index >= 0.44 or revision_index >= 0.42 or jitter_index >= 0.42:
        structure_balance = {
            "code": "tight_controlled",
            "label": "高重複疊畫／細修結構",
            "summary": "依 Pénzes 等人（2020）與 Price 等人（2015），畫面疊畫或反覆修改比例偏高；在數位畫布上可能源於仔細填色、調整位置或內在思量，系統已大幅降權其焦慮分數。",
        }
        strengths_note = "創作者透過反覆描繪、調整與界線整理用心完成畫面，展現高度專注、自我覺察與尋求秩序的內在資源（Pénzes et al., 2020）。"
    elif sharpness_index >= 0.34 and (velocity_index >= 0.50 or (pen_pressure_active and effective_force_index >= 0.62)):
        structure_balance = {
            "code": "expansive_dynamic",
            "label": "高動態外放結構",
            "summary": "線條動態感與落筆能量強烈，展現充沛的情緒外化與突破框限張力（Pénzes et al., 2020）。",
        }
        strengths_note = "畫面呈現鮮明的表達勇氣與主動外化能量，能把內在想法直接轉化為視覺張力，是健康的表達與溝通起點（Pénzes et al., 2020）。"
    elif bbox_area_ratio < 0.18 and len(draw_strokes) <= 8:
        structure_balance = {
            "code": "sparse_tentative",
            "label": "低密度內收／試探結構",
            "summary": "畫面留白比例高且筆觸精簡謹慎，反映小心試探與保留自我空間的節奏（Pénzes et al., 2020）。",
        }
        strengths_note = "雖然筆跡精簡內斂，但創作者仍願意落筆留下專屬印記，維持清晰的自我邊界；在溫和陪伴下常能一步步展開更多表達（Gross & Hayne, 1998）。"
    else:
        structure_balance = {
            "code": "balanced_flexible",
            "label": "結構平衡・具心理彈性",
            "summary": "依 Pénzes 等人（2020）研究，畫面在輪廓清晰度、流動感與空間變化間保持良好平衡，呈現正向心理適應與創造彈性。",
        }
        strengths_note = "畫面展現良好的結構組織力與情緒調節彈性，能自在地整合圖形、空間與節奏來表達內在世界（Pénzes et al., 2020）。"

    resource_fws = [fw for fw in matched_image_frameworks if fw.get("isResource")]
    if resource_fws:
        rfw = resource_fws[0]
        strengths_note = (
            f"{strengths_note} 另偵測到正向因應與保護圖像線索（{'、'.join(rfw['hits'][:3])}），"
            f"呼應 {rfw['label']} 實證所示的內在資源與自我效能。"
        )

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
        if color_change_count == 0:
            color_insight = (
                "偵測到全程沿用預設單色畫筆未主動換色（類似鉛筆素描）。依 Picard & Gauthier (2012) 與 Burkitt (2003a) 實證，"
                "系統已自動啟動「單色去偏」，絕不因預設黑筆給予負向扣分，改以完整圖形、表情與筆觸為主導。"
            )
        else:
            color_insight = (
                "偵測到本次為單一色系素描。依 Picard & Gauthier (2012) 實證，單色作畫時創作者會改以形狀與表情表達情緒，"
                "系統已啟動單色去偏並以完整圖形結構為第一判準。"
            )
    elif is_canonical_nature_palette:
        color_insight = (
            "偵測到藍、綠、暖黃等「自然寫實原色」組合。依 Crawford 等人（2012）實驗（寫實配色與情緒強度 r = -0.14 不顯著），"
            "系統已自動降低藍色的悲傷誤判權重，避免把藍天綠草判成低落。"
        )
    elif (intense_red + dark_muted) >= 0.55 and intense_red >= 0.18:
        color_insight = "色彩由高張力紅與深暗色輔助主導，參考 Burkitt (2003a) 與 Cao 等人（2023）實驗，高對比紅色常呼應威脅警覺、憤怒或迫切的情感宣洩。"
    elif (cool_blue + neutral_grey + dark_muted) >= 0.58:
        color_insight = "色調偏向冷色或暗沉色系，參考 Burkitt (2003a) 與 Cao 等人（2023）研究常與低落或孤單相伴；提醒仍須對照創作者日常個人喜愛色。"
    elif positive_warm_palette >= 0.55:
        color_insight = "暖明色與自然色系帶來溫暖正向的氛圍基調，輔助強化作品的開放與活力感受。"
    else:
        color_insight = "色彩冷暖交錯具多層次變化，與畫面圖形共同映照出豐富細微的情緒層次（仍須參照個人偏好色）。"

    tool_zh = {"marker": "麥克筆", "crayon": "蠟筆", "pencil": "鉛筆", "watercolor": "水彩"}
    dom_tool = max(tool_counts, key=lambda k: tool_counts[k]) if tool_counts else "crayon"
    tool_summary = f"主畫具：{tool_zh.get(dom_tool, dom_tool)}（平均粗細 {round(mean_raw_size, 1)}px）"
    interface_signals: List[str] = [sensing_mode_label, tool_summary]
    if pen_pressure_active:
        interface_signals.append("硬體壓感低權重啟用")
    else:
        interface_signals.append("數位筆壓情緒權重歸零")
    if color_change_count > 0:
        interface_signals.append(f"主動換色 {color_change_count} 次")
    elif is_monochrome_sketch:
        interface_signals.append("沿用預設筆色（未主動換色）")
    if tool_switch_count > 0:
        interface_signals.append(f"切換畫具 {tool_switch_count} 次")
    if size_change_count > 0:
        interface_signals.append(f"調整粗細 {size_change_count} 次")
    if in_air_pause_ratio > 0.05:
        interface_signals.append(f"提筆停頓佔比 {round(in_air_pause_ratio * 100)}%")

    if sharpness_index >= 0.34 and velocity_index >= 0.50:
        stroke_insight = (
            f"透過「{sensing_mode_label}」觀察，筆觸呈現明顯尖銳折角與較快劃速"
            + (f"（真壓感輔助筆力 {round(effective_force_index * 100)}%）" if pen_pressure_active else "（滑鼠／觸控已停用筆壓情緒推論）")
            + "，展現較強動態能量。"
        )
    elif overlap_index >= 0.44:
        stroke_insight = f"多處出現重複疊畫或填色痕跡（結構分類：{structure_balance['label']}）。依 Price 等人（2015），平板／數位畫布上的疊色與重塗權重已調降約 58%，建議搭配輔導員縮時回放觀察底層圖案。"
    elif jitter_index >= 0.40 or in_air_pause_ratio >= 0.65:
        stroke_insight = f"軟體介面記錄到微震顫（{round(jitter_index * 100)}%）或停筆思考時間（{round(in_air_pause_ratio * 100)}%）；因數位操作常含自然停頓，已調降約 60% 情緒加權並優先轉化為分頁 2 對話切入點。"
    elif fragmentation_index >= 0.45:
        stroke_insight = "畫面採用較多短段筆畫。依 Mohd Shukri（2014），兒童在平板／觸控板上常主動採「多段短筆畫」降低畫錯風險，系統已大幅降權（-62%）避免誤判焦慮或退縮。"
    elif smoothness_index >= 0.42:
        stroke_insight = f"整體線條圓潤連貫、轉折柔和（{tool_summary}），依 Pénzes 等人（2020）顯示運筆流動順暢、結構協調。"
    else:
        stroke_insight = f"筆觸兼具直線結構與柔和曲線（{tool_summary}），節奏平穩可控，反映創作者正以有條理的方式組織畫面。"

    # Pulver (1931/1953) graphology psychological symbolism REMOVED (weight = 0); pure objective geometry ONLY
    horiz_pos = "水平偏左" if cx < 0.38 else ("水平偏右" if cx > 0.62 else "水平居中")
    vert_pos = "垂直偏上" if cy < 0.38 else ("垂直偏下" if cy > 0.62 else "垂直居中")
    if bbox_area_ratio < 0.18:
        spatial_insight = f"作品主體僅佔畫布約 {round(bbox_area_ratio * 100)}% 且幾何重心位於{horiz_pos}、{vert_pos}。依 Burkitt (2003b) 與 Cao 等人（2023）實證，偏小縮聚構圖常與孤單、害羞或退縮相關（不含 Pulver 方位心理推論）。"
    elif edge_touch_ratio >= 0.30 or bbox_area_ratio > 0.78:
        spatial_insight = f"畫面大幅延展至畫布邊緣（覆蓋約 {round(bbox_area_ratio * 100)}%，重心位於{horiz_pos}、{vert_pos}）。依 Burkitt (2004) 與 Cao 等人（2023）實驗，大畫幅可能是正向開展，也可能是高張力外放，需結合表情與對話判讀。"
    else:
        spatial_insight = f"畫面佔比約 {round(bbox_area_ratio * 100)}%，客觀幾何重心位於{horiz_pos}、{vert_pos}，留白與主體比例協調舒展（Burkitt et al., 2004; Cao et al., 2023）。"

    ch_str = " ＋ ".join(expressive_channels)
    if recognized_figures:
        fig_names = "、".join(f["label"] for f in recognized_figures[:3])
        process_insight = f"系統透過多筆空間拓樸辨識出「{fig_names}」（表達通道：{ch_str}），並依 29 篇國際期刊實證原則以完整圖形與情境內容為優先判準。"
    elif revision_events >= 3:
        process_insight = f"繪畫過程中進行了 {revision_events} 次復原或擦除動作（表達通道：{ch_str}），已依數位媒材實證降權情緒分數，建議於下方輔導員區使用「作畫歷程縮時回放」觀察修改熱區。"
    else:
        process_insight = f"全作共 {len(draw_strokes)} 筆完成（表達通道：{ch_str}），未偵測到固定具象圖式，系統以低權重動力特徵與空間比例作探索參考，建議透過分頁 2 開放式對話了解孩子想法。"

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
    if counselor_verified_objects:
        inner_thoughts_paragraph += f" 納入輔導人員協助辨識的畫中元素（{'、'.join(counselor_verified_objects[:3])}）後，更能貼近創作者原本想表達的真實畫面。"
    if counselor_follow_up_notes:
        inner_thoughts_paragraph += " 並已結合後續訪談溝通脈絡重新校準整體判讀。"

    # --- 8. Explainable Reasoning Tree (3-Level Interactive Logic & Evidence Summary) ---
    emo_short = {k: EMOTION_PROFILES[k]["short_label"] for k in EMOTION_KEYS}

    def _fmt_impact(impact_map: Dict[str, float]) -> str:
        parts = [
            f"{emo_short.get(k, k)} {'+' if v >= 0 else ''}{v:.2f}"
            for k, v in sorted(impact_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
            if abs(v) >= 0.02
        ]
        return "、".join(parts) if parts else "僅作對話參考（0 扣分）"

    # Branch 1: Figures & Content (Primary priority)
    fig_children: List[Dict[str, str]] = []
    if counselor_verified_objects:
        c_impact_str = _fmt_impact(counselor_scores) if any(v > 0 for v in counselor_scores.values()) else "人工脈絡確認"
        fig_children.append({
            "id": "fig_counselor_verified",
            "title": f"輔導人員協助辨識：{('、'.join(counselor_verified_objects[:4]))}",
            "tag": c_impact_str,
            "summary": "已由輔導人員於繪畫後協助確認畫中具體物件／角色，優先納入內容脈絡。",
            "detail": (
                f"輔導人員共補充確認 {len(counselor_verified_objects)} 項畫面元素（{'、'.join(counselor_verified_objects)}）。"
                "依 Gross & Hayne (1998)、Driessnack (2005) 與 Picard 等人（2007）研究，由創作者口頭澄清並經輔導人員確認之畫面主題，"
                "其脈絡可信度高於純幾何猜測，已同步更新至情緒收斂權重與互動引導提問。"
            ),
            "citation": "輔導人員人工協助辨識 × Gross & Hayne (1998); Driessnack (2005); Picard et al. (2007)",
        })
    for fw in matched_image_frameworks[:3]:
        fig_children.append({
            "id": f"fig_fw_{fw['id']}",
            "title": f"純圖像架構匹配：{fw['label']}",
            "tag": f"命中：{'、'.join(fw['hits'][:3])}",
            "summary": fw["summary"],
            "detail": (
                f"系統於畫面確認元素／口述脈絡中偵測到非色彩圖像線索「{'、'.join(fw['hits'])}」，"
                f"自動對照 {fw['citation']} 之實證判讀架構。延伸引導建議：{fw['promptTip']}"
            ),
            "citation": fw["citation"],
        })
    for f in recognized_figures:
        ch_lbl = "字面表情（Literal）" if f.get("channel") == "literal" else "情境隱喻（Content）"
        fig_children.append({
            "id": f"fig_{f['code']}",
            "title": f"{f['label']}（{ch_lbl}）",
            "tag": _fmt_impact(f.get("impact") or {}),
            "summary": f["meaning"],
            "detail": (
                f"空間拓樸依據：系統由多筆座標關係定位此結構（範圍 {round(f['bbox']['w'])}×{round(f['bbox']['h'])}px），"
                f"依「{f['theory']}」賦予情緒方向權重（{_fmt_impact(f.get('impact') or {})}）。"
            ),
            "citation": f["theory"],
        })
    if not fig_children:
        fig_children.append({
            "id": "fig_none",
            "title": "未偵測到 17 類固定具象圖式",
            "tag": "改由動力與空間推估",
            "summary": "畫面為自由抽象線條或非固定幾何物體，未套用固定圖形加權。",
            "detail": (
                "方案 A 採幾何拓樸辨識（臉部表情、屋樹門窗、太陽、雨雲、愛心、打叉、柵欄、人物）。"
                "當未命中固定圖式時，系統不會硬猜物件，而是自動將權重轉移至低權重筆觸動力、空間比例與口語脈絡；"
                "輔導人員亦可直接於下方「輔導人員 AI 諮詢與圖像補充區」觀看作畫縮時回放、輸入實際辨識內容並重新解析，或切換方案 B（Gemini 視覺）。"
            ),
            "citation": "Jolley et al. (2004); Picard et al. (2007)",
        })

    # Branch 2: Software Interface Strokes & Kinematics (v3.5 device split & digital debiasing)
    stroke_children: List[Dict[str, str]] = [
        {
            "id": "stroke_sensing",
            "title": f"裝置感測與筆壓分流：{sensing_mode_label}",
            "tag": f"真壓低權重 ×0.07" if pen_pressure_active else "筆壓情緒權重歸零",
            "summary": f"{tool_summary}；Alschuler & Hattwick (1947) 筆壓推論僅限真壓電繪板啟用。",
            "detail": (
                f"目前感測模式為「{sensing_mode_label}」。"
                + (
                    f"偵測到硬體感壓筆且具真實壓感變化，已保留低權重動作張力參考（等效筆力 {round(effective_force_index * 100)}%，情緒權重 ×0.07）。"
                    if pen_pressure_active
                    else f"在滑鼠／筆電觸控板或平板手指觸控模式下，已完全停用筆壓輕重的情緒推論（權重 0.00；遙測參考值 {round(effective_force_index * 100)}%），避免把無壓感硬體特性誤判為情緒力道。"
                )
            ),
            "citation": "Price et al. (2015); Mohd Shukri (2014); Alschuler & Hattwick (1947 裝置限縮)",
        },
        {
            "id": "stroke_geometry",
            "title": f"線條幾何動力：尖銳 {round(sharpness_index * 100)}% ／ 流暢 {round(smoothness_index * 100)}% ／ 重疊 {round(overlap_index * 100)}%",
            "tag": structure_balance["label"],
            "summary": stroke_insight,
            "detail": (
                f"透過 Ramer-Douglas-Peucker 折角偵測與網格覆蓋分析：尖銳轉折比例 {round(sharpness_index * 100)}%、"
                f"圓潤弧度比例 {round(smoothness_index * 100)}%、劃動速度 {round(velocity_index * 100)}%、"
                f"重疊塗抹 {round(overlap_index * 100)}%（v3.5 已調降重疊與斷續短線權重 55%–62%，防止將平板多段短筆畫與疊色填塗誤判為焦慮）。"
            ),
            "citation": "Pénzes et al. (2020); Price et al. (2015); Mohd Shukri (2014)",
        },
        {
            "id": "stroke_hesitation",
            "title": f"作畫歷程軌跡（轉作回放與對話錨點）：微震顫 {round(jitter_index * 100)}% ／ 停頓 {round(in_air_pause_ratio * 100)}% ／ 修改 {revision_events} 次",
            "tag": "已降權 58%・供回放與對話用",
            "summary": f"全作共 {len(draw_strokes)} 筆、歷時 {round(total_duration_ms / 1000.0, 1)} 秒（復原 {undo_count} 次、橡皮擦 {eraser_count} 次）。",
            "detail": (
                f"微震顫（{round(jitter_index * 100)}%）、提筆停頓（{round(in_air_pause_ratio * 100)}%）與復原／擦除（{revision_events} 次）在平板與觸控板常源於手指誤觸或微調位置，"
                "v3.5 已將其情緒加權全面調降 56%–62%，改為驅動下方「輔導員縮時回放熱區」與分頁 2 的自然開放式提問。"
            ),
            "citation": "Mohd Shukri (2014); Price et al. (2015); Pénzes et al. (2020)",
        },
    ]

    # Branch 3: Spatial Composition & Size Asymmetry (Pulver graphology zeroed)
    size_asymmetry_tag = (
        "大畫幅高張力（威脅外放）" if large_tense_threat > 0
        else ("大畫幅舒展（正向／平和）" if large_open_positive > 0
              else ("微小縮聚（孤單／內收）" if bbox_area_ratio < 0.18 else "均衡幅面"))
    )
    spatial_children: List[Dict[str, str]] = [
        {
            "id": "spatial_size",
            "title": f"主體佔比 {round(bbox_area_ratio * 100)}%（實際著墨 {round(ink_coverage * 100, 1)}%）",
            "tag": size_asymmetry_tag,
            "summary": spatial_insight,
            "detail": (
                f"依 Burkitt 等人（2003b; 2004）與 Cao 等人（2023）實證，尺寸與情緒呈「非對稱關係」："
                f"正向／平和畫作會穩定舒展放大（本次大畫幅舒展加權 +{large_open_positive:.2f}），"
                f"高威脅／憤怒則可能放大且伴隨尖銳重塗（本次高張力加權 +{large_tense_threat:.2f}），"
                f"唯獨孤單／退縮顯著呈現微小縮聚（主體佔比 < 18%）。"
            ),
            "citation": "Burkitt et al. (2003b, 2004); Cao et al. (2023)",
        },
        {
            "id": "spatial_center",
            "title": f"客觀幾何重心 X:{round(cx * 100)}% / Y:{round(cy * 100)}%・邊界觸及 {round(edge_touch_ratio * 100)}%",
            "tag": f"{horiz_pos}・{vert_pos}",
            "summary": f"客觀位置：{horiz_pos}、{vert_pos}；底部著墨比例 {round(bottom_weight * 100)}%（Pulver 左右上下心理推論權重已歸零）。",
            "detail": (
                f"v3.5 已移除傳統筆跡學左右／上下空間象徵（Pulver 心理推論權重 = 0.00），此欄僅記錄客觀幾何位置與中心距離（{round(center_dist * 100)}%）；"
                f"僅結合 Burkitt／Cao 縮聚尺寸實證與邊界觸及比例（{round(edge_touch_ratio * 100)}%）進行空間輔助判讀。"
            ),
            "citation": "Burkitt et al. (2003b, 2004); Cao et al. (2023); Pulver (1953 僅限幾何座標)",
        },
    ]

    # Branch 4: Color Debiasing & Cultural Context
    color_mode_tag = (
        "單色鉛筆去偏（權重×0.20）" if is_monochrome_sketch
        else ("寫實原色去偏（藍色降權）" if is_canonical_nature_palette else f"多色輔助（權重×{color_scale:.2f}）")
    )
    color_children: List[Dict[str, str]] = [
        {
            "id": "color_debias",
            "title": f"色彩去偏機制：{color_mode_tag}",
            "tag": f"主動換色 {color_change_count} 次",
            "summary": color_insight,
            "detail": (
                f"色彩在本引擎中僅為「次要調整項」（縮放係數 {color_scale:.2f}），絕不凌駕於完整圖形與表情之上。"
                + (
                    "本次偵測為單一色系素描，黑色／暗色負向扣分已強制歸零（monoDarkPen = 0.0）。"
                    if is_monochrome_sketch
                    else (
                        "本次偵測到藍天、綠草、暖黃等自然寫實配色，依 Crawford 等人（2012）將藍色悲傷權重調降為 0.38 倍。"
                        if is_canonical_nature_palette
                        else "系統已同時提醒：單一顏色受個人日常喜愛色影響，需搭配圖形與對話解讀（Burkitt et al., 2003a）。"
                    )
                )
            ),
            "citation": "Picard & Gauthier (2012); Crawford et al. (2012); Burkitt et al. (2003a)",
        },
        {
            "id": "color_dist",
            "title": f"主導色系：{color_bucket_labels.get(dominant_color_bucket, dominant_color_bucket)}",
            "tag": f"彩度 {round(mean_sat * 100)}% ／ 明度 {round(mean_lum * 100)}%",
            "summary": (
                f"暖明 {round(warm_bright * 100)}%・紅 {round(intense_red * 100)}%・綠 {round(fresh_green * 100)}%・"
                f"藍 {round(cool_blue * 100)}%・暗色 {round(dark_muted * 100)}%。"
            ),
            "detail": (
                "依 Burkitt (2003a) 與 Cao 等人（2023）華語文化實證：暖明與自然綠多支持愉悅／平和，"
                "高飽和紅在搭配尖銳筆觸或大畫幅時指向威脅警覺或憤怒，冷暗色搭配縮聚或下垂線條時輔助指向低落或孤單。"
            ),
            "citation": "Burkitt et al. (2003a); Cao et al. (2023)",
        },
    ]

    # Branch 5: Context, 3-13 Age Norm, 3C Media & Neurodiversity Guardrail
    matched_verbal_flat = [f"{emo_short[k]}({','.join(v)})" for k, v in matched_words.items() if v]
    context_children: List[Dict[str, str]] = [
        {
            "id": "ctx_verbal",
            "title": "自發口語敘說與元素標籤（Draw & Tell）",
            "tag": "言畫反差提醒" if incongruence_note else ("已納入文字／標籤" if (title_or_note or selected_motifs) else "未填寫"),
            "summary": (
                f"文字「{title_or_note}」" if title_or_note else "創作者未填寫文字說明"
            ) + (f"；命中詞彙：{'、'.join(matched_verbal_flat)}" if matched_verbal_flat else "") + (
                f"；自選標籤：{'、'.join(selected_motifs)}" if selected_motifs else ""
            ),
            "detail": (
                incongruence_note
                if incongruence_note
                else "依 Gross & Hayne (1998) 與 Driessnack (2005, d = 0.95) 實證，讓兒童透過畫作敘說能大幅提升表達細節；文字詞彙僅作為輔助加權（最高 +0.36），若與畫面張力相反系統會主動提醒「可能的情緒掩飾」。"
            ),
            "citation": "Gross & Hayne (1998); Driessnack (2005)",
        },
        {
            "id": "ctx_age",
            "title": f"3–13 歲年齡發展常模：{age_norm['label']}",
            "tag": f"省略細節係數 ×{omitted_detail_scale:.2f}",
            "summary": age_norm["guidance"],
            "detail": (
                f"預期表達風格：{age_norm['expectedStyle']}。"
                "3–6 歲學齡前完全停用 Koppitz 部位省略扣分（×0.00）並排除平板誤觸；"
                "7–10 歲自動啟動流行文化／遊戲角色去偏；11–13 歲提供寫實前期畫技顧慮與暗黑酷炫風格替代解釋。"
            ),
            "citation": "Picard & Gauthier (2012); Ivashkevich (2009); Price et al. (2015); Lowenfeld (1987)",
        },
        {
            "id": "ctx_guardrail",
            "title": f"多維證據收斂護欄：{conv_label}",
            "tag": f"命中 {converging_channels} 維・分數上限 {max_score_cap}",
            "summary": alt_note,
            "detail": (
                f"本次跨「圖形結構、筆觸動力、空間構圖、口語標籤、多色分佈」共命中 {converging_channels} 項收斂證據，"
                f"因此啟用分數上限 {max_score_cap} 分，嚴格遵守 Joiner (1996) 與 Catte & Cox (1999)「單一線索不足以推斷困擾」的實證倫理護欄；"
                f"並依 Pénzes (2020) 同步標註正向資源：{strengths_note}"
            ),
            "citation": "Joiner et al. (1996); Catte & Cox (1999); Thomas & Jolley (1998); Pénzes et al. (2020)",
        },
    ]

    if has_pop_culture_cues:
        context_children.insert(2, {
            "id": "ctx_pop_culture",
            "title": "3C 媒體與流行文化去偏檢核",
            "tag": "負向張力 ×0.55" if pop_culture_attenuation < 1.0 else "建議檢核媒體來源",
            "summary": f"偵測到流行文化／角色線索：{'、'.join(pop_culture_hits[:4])}，避免直接將動漫或遊戲角色誤判為攻擊性或創傷。",
            "detail": (
                "依 Ivashkevich (2009) 實證，7–10 歲為動漫與遊戲模仿高峰期，繪製怪獸、尖牙、武器或誇張戰鬥畫面常屬同儕文化創作；"
                "分頁 2 已優先安排「媒體來源檢核提問」，協助大人以好奇而非責備口吻確認角色靈感來源。"
            ),
            "citation": "Ivashkevich (2009), Studies in Art Education, 51(1), 50–63",
        })

    if neurodiversity_note:
        context_children.append({
            "id": "ctx_neurodiversity",
            "title": "神經多樣性（ASD）評估護欄",
            "tag": "不可直接套用傳統 HFD",
            "summary": neurodiversity_note,
            "detail": (
                "Papangelo 等人（2020）指出傳統人物畫（HFD）不可直接套用於推論自閉症譜系兒童的情緒或社會知覺；"
                "Kotroni 等人（2018）亦證實 ASD 兒童雖具表達社會情緒的能力，但繪畫中的視覺情緒線索通常較少，偏好機械、結構細節或重複幾何屬於獨特認知優勢與風格。"
            ),
            "citation": "Papangelo et al. (2020), Brain Sciences; Kotroni et al. (2018), IJDD",
        })

    if has_counselor_context:
        c_matched_flat = [
            f"{emo_short[k]}({','.join(v)})"
            for k, v in counselor_matched_cues.items()
            if v
        ]
        context_children.insert(1, {
            "id": "ctx_counselor",
            "title": "輔導人員協助辨識與後續訪談脈絡",
            "tag": _fmt_impact(counselor_scores) if any(v > 0 for v in counselor_scores.values()) else "已納入人工補充",
            "summary": (
                (f"協助辨識：{'、'.join(counselor_verified_objects)}。" if counselor_verified_objects else "")
                + (f"後續溝通紀錄：「{counselor_follow_up_notes[:120]}{'…' if len(counselor_follow_up_notes) > 120 else ''}」" if counselor_follow_up_notes else "")
            ),
            "detail": (
                f"輔導人員於初次分析後補入之脈絡已整合至雙分頁重算"
                + (f"（命中語意線索：{'、'.join(c_matched_flat)}，上限 +0.34）" if c_matched_flat else "（維持客觀中性權重，未硬性加權）")
                + "；系統仍遵循 Thomas & Jolley (1998) 多元詮釋原則，不以單一線索下診斷定論。"
            ),
            "citation": "Human-in-the-Loop Counselor Context × Gross & Hayne (1998); Driessnack (2005)",
        })

    # Build top-level judgment summary explaining WHY primary emotion won
    lead_reasons: List[str] = []
    if counselor_verified_objects:
        lead_reasons.append(f"納入輔導人員協助辨識「{'、'.join(counselor_verified_objects[:2])}」")
    if recognized_figures:
        lead_reasons.append(f"優先辨識出「{'、'.join(f['label'] for f in recognized_figures[:2])}」")
    if is_monochrome_sketch:
        lead_reasons.append("已啟動「單色鉛筆去偏」排除黑筆誤判")
    elif is_canonical_nature_palette:
        lead_reasons.append("已啟動「寫實原色去偏」避免把藍天綠草判成低落")
    if sharpness_index >= 0.34 or overlap_index >= 0.42:
        lead_reasons.append("結合較高的尖銳／重疊塗抹張力")
    elif smoothness_index >= 0.42:
        lead_reasons.append("搭配圓潤流暢的筆觸節奏")
    if bbox_area_ratio < 0.18:
        lead_reasons.append("加上偏小縮聚的空間比例")
    elif large_tense_threat > 0:
        lead_reasons.append("加上大畫幅高張力外放構圖")
    if conv_level == "low":
        lead_reasons.append("因筆跡線索較簡短已套用 66 分上限護欄")

    score_gap = primary["score"] - secondary["score"]
    gap_desc = (
        f"領先次高「{secondary['label']}（{secondary['score']}）」達 {score_gap} 分，主導方向明確。"
        if score_gap >= 12
        else f"與次高「{secondary['label']}（{secondary['score']}）」僅差 {score_gap} 分，呈現雙重情緒交織狀態。"
    )
    root_summary = (
        f"綜合 5 大維度判斷，本次結果總結為「{primary['label']}（{primary['score']} 分）」：{gap_desc}"
        + (f"核心判斷依據為：{'，'.join(lead_reasons)}。" if lead_reasons else "")
        + "點擊下方任一分支節點，可逐層展開查看模型採用了哪些數據、規則與國際期刊文獻。"
    )

    total_content_count = len(recognized_figures) + len(counselor_verified_objects)
    fig_branch_summary = (
        f"已整合輔導人員確認元素「{'、'.join(counselor_verified_objects[:2])}」"
        + (f"與系統偵測「{'、'.join(f['label'] for f in recognized_figures[:2])}」" if recognized_figures else "")
        + f"，共同作為「{primary['shortLabel']}」之核心內容判準。"
        if counselor_verified_objects
        else (
            f"已偵測「{'、'.join(f['label'] for f in recognized_figures[:3])}」，對「{primary['shortLabel']}」構成核心判準。"
            if recognized_figures
            else "未命中 17 類固定圖式，系統自動以筆觸動力、空間比例與文字作為主要推論來源。"
        )
    )

    reasoning_tree = {
        "rootTitle": f"結論總結：{primary['label']}（{primary['score']} 分）",
        "rootBadge": conv_label,
        "rootSummary": root_summary,
        "engineSource": "local_empirical_v3",
        "branches": [
            {
                "id": "figures",
                "title": "① 圖像內容與表情辨識（第一優先）",
                "status": "primary" if total_content_count > 0 else "neutral",
                "badge": (
                    f"共 {total_content_count} 項（含輔導員確認）"
                    if counselor_verified_objects
                    else (f"辨識 {len(recognized_figures)} 項圖形" if recognized_figures else "無固定圖式（轉由動力推估）")
                ),
                "summary": fig_branch_summary,
                "children": fig_children,
            },
            {
                "id": "strokes",
                "title": "② 軟體介面筆觸與形式結構",
                "status": "supporting",
                "badge": f"等效筆力 {round(effective_force_index * 100)}% · {structure_balance['label'].split('・')[0]}",
                "summary": stroke_insight,
                "children": stroke_children,
            },
            {
                "id": "spatial",
                "title": "③ 空間構圖與尺寸非對稱",
                "status": "supporting",
                "badge": f"幅面 {round(bbox_area_ratio * 100)}% · {size_asymmetry_tag.split('（')[0]}",
                "summary": spatial_insight,
                "children": spatial_children,
            },
            {
                "id": "color",
                "title": "④ 色彩輔助與去偏校準",
                "status": "attenuated" if (is_monochrome_sketch or is_canonical_nature_palette) else "supporting",
                "badge": color_mode_tag,
                "summary": color_insight,
                "children": color_children,
            },
            {
                "id": "context_guardrail",
                "title": "⑤ 口語脈絡、3–13 歲常模、3C 媒體與神經多樣性護欄",
                "status": "guardrail" if (conv_level == "low" or incongruence_note or age_group == "3-6" or safety_alert or neurodiversity_note or has_pop_culture_cues) else "neutral",
                "badge": f"{conv_label.split('（')[0]} · 上限 {max_score_cap}",
                "summary": alt_note,
                "children": context_children,
            },
        ],
    }

    # --- 9. Non-Diagnostic Draw & Tell Interactive Dialogue Guide (Gross & Hayne, 1998; Driessnack, 2005; Malchiodi, 1998) ---
    age_tone_meta = {
        "3-6": {
            "toneLabel": "3–6 歲｜遊戲與角色扮演語氣",
            "companionTip": "蹲下來與孩子平視，用好奇玩伴的口吻聊畫裡的小世界；在平板上多段短筆畫或誤觸很正常，每次只問 1 題並給足 5–10 秒等待回應。",
        },
        "7-10": {
            "toneLabel": "7–10 歲｜故事主角與冒險探索語氣",
            "companionTip": "把孩子當成這幅畫的「小導演」，邀請他介紹角色與靈感來源（是否來自影片、漫畫或遊戲），避免評價畫得像不像或急著貼情緒標籤。",
        },
        "11-13": {
            "toneLabel": "11–13 歲｜平等尊重與青春期前視角語氣",
            "companionTip": "保持輕鬆平等的聊天節奏，接納「怕畫不像而畫得簡略或暗黑風格」的自我要求，尊重孩子決定要分享到哪裡。",
        },
        "unspecified": {
            "toneLabel": "3–13 歲通用模式｜開放式非誘導對話",
            "companionTip": "依 Gross & Hayne (1998) 與 Driessnack (2005, d = 0.95) 實證原則：只描述你看到的畫面細節與作畫歷程並問開放問題，不預設立場或替孩子貼標籤。",
        },
    }[age_group]

    fig_first = recognized_figures[0] if recognized_figures else None
    fig_code = fig_first["code"] if fig_first else ""
    verified_first = counselor_verified_objects[0] if counselor_verified_objects else ""
    fig_name = verified_first or (fig_first["label"] if fig_first else "")

    if fig_name:
        opening_cue = f"「我看見你在畫紙上畫了『{fig_name}』，感覺這裡面藏著一個很特別的故事。你可以當小導遊，帶我看看這幅畫嗎？」"
    elif title_or_note:
        opening_cue = f"「你幫這幅畫留下了『{title_or_note}』這句話，我很想知道畫裡正在發生什麼事，可以跟我說說看嗎？」"
    elif sharpness_index >= 0.34 or overlap_index >= 0.44:
        opening_cue = f"「哇，你用{tool_zh.get(dom_tool, '畫筆')}畫出很有存在感的筆觸，感覺剛剛畫得很投入！這裡面最想先介紹給我的是哪一塊？」"
    else:
        opening_cue = f"「謝謝你願意用{tool_zh.get(dom_tool, '畫筆')}把腦海裡的畫面畫下來。你最喜歡這幅畫裡的哪一個角落？可以跟我介紹嗎？」"

    # Build Process-Driven Open-Ended Prompts (v3.5)
    ph_repeat = process_highlights.get("repeatHotspot")
    ph_revision = process_highlights.get("revisionHotspot")
    ph_seq = process_highlights.get("sequence")
    ph_tempo = process_highlights.get("tempoShift")

    process_s1_prompts: List[str] = []
    process_s3_prompts: List[str] = []

    if ph_repeat and ph_repeat.get("zone"):
        process_s1_prompts.append(
            f"我看見你在畫「畫面{ph_repeat['zone']}」這個部分的時候好像重複畫了很久、疊了好幾層，畫這裡的時候心裡在想什麼呢？"
        )
    if ph_revision and ph_revision.get("zone"):
        process_s1_prompts.append(
            f"我注意到你剛剛在「畫面{ph_revision['zone']}」這一塊修改了好幾次，你原本想把它畫成什麼樣子呢？"
        )
    if ph_seq and ph_seq.get("firstZone") and ph_seq.get("lastZone") and ph_seq["firstZone"] != ph_seq["lastZone"]:
        process_s3_prompts.append(
            f"你一開始先畫了畫面{ph_seq['firstZone']}的地方，最後才在{ph_seq['lastZone']}加上那一筆，這中間發生了什麼事呀？"
        )
    if ph_tempo and ph_tempo.get("type") == "long_pause" and ph_tempo.get("zone"):
        p_sec = ph_tempo.get("pauseSec")
        sec_txt = f"（停了約 {p_sec} 秒）" if p_sec else ""
        process_s3_prompts.append(
            f"剛剛停下來想了一下{sec_txt}之後才畫出畫面{ph_tempo['zone']}這一塊，那時候腦海裡浮現了什麼畫面？"
        )
    elif ph_tempo and ph_tempo.get("type") == "tool_size_shift" and ph_tempo.get("zone"):
        detail_txt = ph_tempo.get("detail") or "不同的筆刷"
        process_s3_prompts.append(
            f"剛剛你換了「{detail_txt}」才畫出畫面{ph_tempo['zone']}這一塊，那時候腦海裡浮現了什麼畫面？"
        )

    # Stage 1: Notice & Warm-up (Purely descriptive, zero judgment, prioritizing process highlights)
    s1_prompts: List[str] = []
    if process_s1_prompts:
        s1_prompts.extend(process_s1_prompts[:2])
    if len(s1_prompts) < 2:
        s1_prompts.append(
            f"我注意到畫裡有「{fig_name}」，你可以告訴我他是誰、或者他正在這裡做什麼嗎？"
            if fig_name
            else "如果我們一起走進這幅畫裡，你最想先帶我看哪一個地方？（你剛剛是從哪一筆開始畫的？）"
        )
    if len(s1_prompts) < 2:
        s1_prompts.append("這裡的線條和留白安排得很特別，你畫這一塊的時候心裡想到的是什麼畫面或聲音呢？")
    s1_prompts = s1_prompts[:3]

    # Stage 2: Step Into the Story (Third-person projective bridge + 7-10 3C Media Source Check)
    if fig_code in ("face_sad_crying", "face_sad", "rain_cloud_figure"):
        s2_prompts = [
            "如果畫裡的角色（或這場雨）會說話，他現在最想跟誰說一句悄悄話？",
            "在這個畫面出現之前，你覺得剛剛發生了什麼事情？接下來他希望發生什麼事？",
        ]
    elif fig_code in ("face_angry", "cross_out_figure"):
        s2_prompts = [
            "如果畫裡最有張力的那個部分發出聲音，聽起來會像什麼聲音？他在保護或抗議什麼呢？",
            "假設這個角色有一張「想說清楚的話」大字報，他會在上面寫什麼？",
        ]
    elif fig_code in ("house_open", "house_closed", "cage_bars_figure"):
        s2_prompts = [
            "這座空間裡面住了誰？平常誰可以敲門進來，誰要留在外面呢？",
            "如果畫裡有一個秘密機關或窗戶，打開之後會看到什麼風景？",
        ]
    elif fig_code in ("person_tiny_isolated", "face_no_mouth", "person_blank_face"):
        s2_prompts = [
            "畫裡的這個主角看起來好安靜，他現在正在看什麼、或是心裡在想什麼小秘密呢？",
            "如果旁邊有一隻會讀心術的小動物走過來陪他，小動物會對他說什麼？",
        ]
    elif fig_code in ("face_happy", "sun_figure", "heart_figure"):
        s2_prompts = [
            "畫裡這個亮亮／開心的畫面，讓你想到哪一個好玩的人、地方或回憶呢？",
            "如果這幅畫可以變成一部小動畫，下一秒畫裡的角色會做什麼好玩的事？",
        ]
    else:
        s2_prompts = [
            "如果這幅畫是一部短片或繪本的其中一頁，你會幫這一頁取什麼名字？",
            "如果畫裡的線條或形狀有自己的聲音與動作，它們現在正在說什麼、或是要去哪裡呢？",
        ]

    # Prioritize 3C Media Source Check Prompt for 7-10 peak media age or whenever pop-culture cues appear (Ivashkevich, 2009)
    if has_pop_culture_cues or (age_group == "7-10" and fig_code in ("face_angry", "cross_out_figure")):
        media_check_q = "這個角色（或武器／怪物）好有氣勢！它是你自己想出來的，還是你在影片、漫畫或遊戲裡看過的呢？"
        s2_prompts = [media_check_q, s2_prompts[0]]

    # Stage 3: Bridge to Inner Thoughts (Gentle, optional connection to real life + process sequence/tempo anchor)
    s3_prompts: List[str] = []
    if process_s3_prompts:
        s3_prompts.append(process_s3_prompts[0])
    s3_prompts.append(
        "畫完這幅畫之後，你現在身體和心裡的感覺，比較像什麼天氣（晴天、微風、陰陰的、還是雷陣雨）？"
    )
    if len(s3_prompts) < 3:
        s3_prompts.append(
            f"剛剛你寫了「{title_or_note}」，最近生活裡有沒有一件事，讓你也有這種感覺，想多跟我聊一點嗎？"
            if title_or_note
            else "最近在學校、家裡或跟朋友相處時，有沒有哪個時刻也有一點點像這幅畫裡的感覺？不想說也沒關係喔。"
        )
    s3_prompts = s3_prompts[:3]

    # Stage 4: Co-Create & Empower (Actionable invitation to add one more stroke together)
    if fig_code in ("face_sad_crying", "face_sad", "rain_cloud_figure"):
        co_draw_idea = {
            "title": "幫畫面加上一樣「暖暖的小幫手」",
            "prompt": "如果你想幫畫裡的主角加一把超能力雨傘、耳機、一條棉被、一隻小寵物或一個好朋友，你想畫在哪裡？現在就一起畫上去吧！",
            "inviteAction": "邀請孩子選一個他最喜歡的顏色，在畫布上添上一樣「讓人覺得被照顧或安心」的小東西。",
        }
    elif fig_code in ("face_angry", "cross_out_figure") or (sharpness_index >= 0.34 and velocity_index >= 0.50):
        co_draw_idea = {
            "title": "把這股力量變成「守護盾牌或安全出口」",
            "prompt": "這幅畫好有能量！如果我們幫這股力量畫一個安全的大框框，或是幫主角畫一個超能力道具來保護他重視的東西，你想畫什麼？",
            "inviteAction": "邀請孩子一起在畫布邊緣或空白處加上「守護道具」或「宣洩完之後的休息角落」。",
        }
    elif fig_code in ("person_tiny_isolated", "house_closed", "cage_bars_figure") or bbox_area_ratio < 0.18:
        co_draw_idea = {
            "title": "在空白處邀請一位「想見的夥伴」或秘密基地",
            "prompt": "畫紙旁邊還有好寬的空間，如果可以邀請一個你喜歡的角色、寵物、遊戲夥伴或一道小門進來陪他，你想畫在哪裡？",
            "inviteAction": "先詢問孩子同不同意你一起畫（尊重邊界），再由孩子指揮你想在哪個角落加上夥伴或秘密基地。",
        }
    elif fig_code in ("face_happy", "sun_figure", "heart_figure"):
        co_draw_idea = {
            "title": "把這份好心情擴建成「秘密遊樂園」",
            "prompt": "這個畫面好舒服！如果可以再多加一個你最喜歡的食物、遊戲或想一起分享的人，你想把它畫在哪裡？",
            "inviteAction": "和孩子一人輪流畫一筆，延續這份正向愉快的親子共創時光。",
        }
    else:
        co_draw_idea = {
            "title": "「你畫一筆、我畫一筆」故事接龍",
            "prompt": "如果你想幫這幅畫加上一個「讓你覺得安心或好玩的新元素」，你想加什麼？也可以由你出題、我來幫你畫一筆喔！",
            "inviteAction": "透過共同添上一筆讓孩子感受到「大人願意走進我的世界一起玩」，自然打開更多心裡話。",
        }

    s4_prompts = [
        co_draw_idea["prompt"],
        "今天這幅畫裡，有沒有哪一句話是你想留給明天自己的小提醒或小願望？",
    ]

    # Tailored "Avoid vs. Better" Non-Leading Phrasing Guide
    avoid_phrases: List[Dict[str, str]] = []
    if has_pop_culture_cues or (age_group == "7-10" and fig_code in ("face_angry", "cross_out_figure")):
        avoid_phrases.append({
            "avoid": "「怎麼畫怪獸／武器／尖牙？你是不是很生氣或想打人？」",
            "better": "「這個角色好有氣勢！他是你在影片、漫畫或遊戲裡看過的，還是你自己發明的呢？」",
        })
    if age_group == "3-6" or sensing_mode == "touch_contact":
        avoid_phrases.append({
            "avoid": "「怎麼畫得一段一段的、還一直按復原？是不是很緊張？」",
            "better": "「在平板上慢慢一段一段畫、調整到喜歡的樣子很棒喔！你剛剛最專心畫哪裡？」",
        })
    if is_monochrome_sketch:
        avoid_phrases.append({
            "avoid": "「你怎麼只用黑色畫？是不是心情不好或發生壞事？」",
            "better": "「你用了單一顏色把整個畫面畫出來，像鉛筆素描一樣！你最想讓我先注意哪個形狀？」",
        })
    if bbox_area_ratio < 0.18 or fig_code == "person_tiny_isolated":
        avoid_phrases.append({
            "avoid": "「你怎麼畫得這麼小、縮在角落？是不是很孤單？」",
            "better": "「這個小小的角落看起來很安靜，畫裡的主角在這裡做什麼呢？」",
        })
    if revision_events >= 3 or overlap_index >= 0.44 or sharpness_index >= 0.34:
        avoid_phrases.append({
            "avoid": "「不要一直擦掉或亂塗！怎麼畫得這麼亂？」",
            "better": "「我看見你在這塊反覆修改／疊畫了好幾次，你那時候正在想把它變成什麼樣子呢？」",
        })
    if fig_code in ("face_no_mouth", "person_blank_face", "house_closed"):
        avoid_phrases.append({
            "avoid": "「怎麼忘了畫嘴巴／門窗？這樣不完整喔！」",
            "better": "「我發現這裡留了白，如果這個人／這間房子現在有心裡話，他會想怎麼表達呢？」",
        })
    if len(avoid_phrases) < 2:
        avoid_phrases.append({
            "avoid": "「這是貓咪對不對？／你畫的是爸爸生氣對不對？」（急著猜與誘導）",
            "better": "「這個形狀好特別，可以告訴我這是什麼、它在做什麼嗎？」",
        })
    if len(avoid_phrases) < 3:
        avoid_phrases.append({
            "avoid": "「畫得好漂亮／畫得不像喔」（以美醜或像不像評價）",
            "better": "「你花了很多心思把細節畫出來，畫到哪個部分的時候你最有感覺？」",
        })

    dialogue_guide = {
        "title": "Draw & Tell 圖像互動引導指南",
        "subtitle": "針對 3–13 歲兒童：純粹透過畫作與作畫歷程陪伴孩子說出內心想法（零診斷標籤・非誘導開放提問）",
        "ageToneLabel": age_tone_meta["toneLabel"],
        "openingCue": opening_cue,
        "companionTip": age_tone_meta["companionTip"],
        "theoryBasis": "Gross & Hayne (1998) Draw & Tell 實驗；Driessnack (2005) 統合分析（d = 0.95）；Ivashkevich (2009) 流行文化去偏；Price et al. (2015) / Mohd Shukri (2014) 數位繪畫策略；Papangelo et al. (2020) / Kotroni et al. (2018) 神經多樣性護欄；Willis et al. (2010) DAPR；Bat Or et al. (2022) PPAT",
        "stages": [
            {
                "id": "stage_notice",
                "stepLabel": "第 1 步",
                "stageTitle": "看見畫面（暖身與作畫歷程好奇）",
                "goal": "從具體看到的形狀、重複描繪或修改熱區開啟對話，讓孩子知道過程被溫柔看見，沒有對錯壓力。",
                "prompts": s1_prompts,
                "followUpTip": "孩子回答後，可以輕輕複述他的關鍵字（例如：「原來這是你在保護的基地呀」），再自然進入下一步。",
            },
            {
                "id": "stage_story",
                "stepLabel": "第 2 步",
                "stageTitle": "走進故事（替畫中主角發聲與媒體來源檢核）",
                "goal": "透過「聊畫裡的角色」建立安全距離並釐清動漫／遊戲靈感來源，孩子往往比直接回答「你怎麼了」更願意分享。",
                "prompts": s2_prompts,
                "followUpTip": "順著孩子的想像走，若角色來自遊戲或動畫，也可以順口問「你最喜歡這個角色的哪個能力？」。",
            },
            {
                "id": "stage_feeling",
                "stepLabel": "第 3 步",
                "stageTitle": "貼近心裡（從作畫順序與停頓溫和連結感受）",
                "goal": "結合先畫／後畫順序與停頓轉折輕柔橋接到生活經驗，邀請孩子表達想法，同時尊重他選擇不說的權利。",
                "prompts": s3_prompts,
                "followUpTip": "如果孩子沉默或說「不知道」，溫柔回應「沒關係，畫出來本身就很棒了」，切勿追問逼答。",
            },
            {
                "id": "stage_cocreate",
                "stepLabel": "第 4 步",
                "stageTitle": "一起共創（賦能與添畫保護資源）",
                "goal": "把對話轉化為正向行動，邀請孩子在畫布上添上一筆新的力量、保護物、夥伴或安全基地。",
                "prompts": s4_prompts,
                "followUpTip": co_draw_idea["inviteAction"],
            },
        ],
        "coDrawIdea": co_draw_idea,
        "avoidPhrases": avoid_phrases[:3],
    }

    return {
        "status": "ok",
        "engineMode": "local_empirical_v3",
        "engineVersion": "3.5.0",
        "isMonochromeSketch": is_monochrome_sketch,
        "isCanonicalNaturePalette": is_canonical_nature_palette,
        "ageGroup": age_group,
        "ageNorm": age_norm,
        "expressiveChannels": expressive_channels,
        "processHighlights": process_highlights,
        "popCultureHits": pop_culture_hits,
        "neurodiversityNote": neurodiversity_note,
        "evidenceConvergence": {
            "level": conv_level,
            "label": conv_label,
            "channelCount": converging_channels,
            "alternativeNote": alt_note,
        },
        "structureBalance": structure_balance,
        "strengthsNote": strengths_note,
        "reasoningTree": reasoning_tree,
        "dialogueGuide": dialogue_guide,
        "counselorContext": {
            "verifiedObjects": counselor_verified_objects,
            "followUpNotes": counselor_follow_up_notes,
            "hasContext": has_counselor_context,
        },
        "safetyAlert": safety_alert,
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
                "citation": "Burkitt et al. (2003a); Crawford et al. (2012); Cao et al. (2023)",
                "dominantBucket": color_bucket_labels.get(dominant_color_bucket, dominant_color_bucket),
                "isMonochromeSketch": is_monochrome_sketch,
                "isCanonicalNaturePalette": is_canonical_nature_palette,
                "histogram": color_hist,
                "meanSaturation": round(mean_sat, 3),
                "meanLuminance": round(mean_lum, 3),
                "insight": color_insight,
            },
            "strokes": {
                "title": "線條品質與介面筆觸動力（v3.5 數位降權）",
                "citation": "Pénzes et al. (2020); Price et al. (2015); Mohd Shukri (2014)",
                "strokeCount": len(draw_strokes),
                "sensingMode": sensing_mode,
                "sensingModeLabel": sensing_mode_label,
                "penPressureEmotionActive": pen_pressure_active,
                "effectiveForceIndex": round(effective_force_index, 3),
                "sharpnessIndex": round(sharpness_index, 3),
                "smoothnessIndex": round(smoothness_index, 3),
                "velocityIndex": round(velocity_index, 3),
                "overlapIndex": round(overlap_index, 3),
                "fragmentationIndex": round(fragmentation_index, 3),
                "jitterIndex": round(jitter_index, 3),
                "inAirPauseRatio": round(in_air_pause_ratio, 3),
                "toolSummary": tool_summary,
                "interfaceSignals": interface_signals,
                "insight": stroke_insight,
            },
            "spatial": {
                "title": "空間構圖與客觀重心（不含方位心理推論）",
                "citation": "Burkitt et al. (2003b, 2004); Cao et al. (2023); Pulver (1953 僅限幾何)",
                "grid3x3": grid_3x3,
                "centerOfMass": {"x": round(cx, 3), "y": round(cy, 3)},
                "bboxCoverage": round(bbox_area_ratio, 3),
                "inkCoverage": round(ink_coverage, 4),
                "edgeTouchRatio": round(edge_touch_ratio, 3),
                "insight": spatial_insight,
            },
            "process": {
                "title": "完整圖形辨識與作畫歷程",
                "citation": "Jolley (2004); Picard (2007, 2012); Mohd Shukri (2014); Price et al. (2015)",
                "undoCount": undo_count,
                "eraserCount": eraser_count,
                "colorChangeCount": color_change_count,
                "toolSwitchCount": tool_switch_count,
                "sizeChangeCount": size_change_count,
                "revisionIndex": round(revision_index, 3),
                "durationSeconds": round(total_duration_ms / 1000.0, 1),
                "matchedVerbalCues": {k: v for k, v in matched_words.items() if v},
                "insight": process_insight,
            },
        },
        "references": PSYCHOLOGY_REFERENCES,
        "ethicalNotice": (
            "本分析針對 3–13 歲兒童整合 29 篇國際心理學同儕審查期刊實證研究（涵蓋 Jolley、Picard、Willis、Fury/Pace、Wegmann、"
            "Kaiser、Bat Or、Skybo、Nanayama Tanaka、Burkitt、Crawford、Pénzes、Gross & Hayne、Driessnack、Cao，以及 3C 時代與神經多樣性文獻 "
            "Ivashkevich、Price、Mohd Shukri、Papangelo、Kotroni）與 6 部受限經典架構（共 35 項來源），嚴格遵循「不以單一線索下定論」原則，"
            "僅作為溫柔的情緒探索與開放式溝通引導起點，絕非臨床心理或醫療診斷。"
        ),
    }


def consult_counselor(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Local zero-key adult/counselor consultation responder grounded in empirical report data."""
    question = str(payload.get("question") or "").strip()[:500]
    if not question:
        return {
            "status": "error",
            "reply": "請先輸入想討論的畫面細節、辨識補充或後續提問。",
            "safetyAlert": None,
            "suggestedObject": None,
            "suggestedNote": None,
        }

    report = payload.get("analysisSummary") or {}
    if not isinstance(report, dict):
        report = {}
    counselor_ctx = payload.get("counselorContext") or {}
    if not isinstance(counselor_ctx, dict):
        counselor_ctx = {}

    verified_objects = [
        str(x).strip()[:80]
        for x in list(counselor_ctx.get("verifiedObjects") or [])[:15]
        if isinstance(x, str) and str(x).strip()
    ]
    follow_up_notes = str(counselor_ctx.get("followUpNotes") or "").strip()[:600]

    safety_alert = _detect_safety_alert([question, follow_up_notes, *verified_objects])
    if safety_alert:
        res_lines = "\n".join(f"• {r}" for r in safety_alert["resources"])
        reply = (
            "【安全與專業轉介優先提醒】\n"
            "您在提問或訪談紀錄中提到可能涉及自傷、受暴或立即安全疑慮的線索。"
            "繪畫心理工具僅為輔助理解之起點，絕不可替代正式危機評估或法定保護處遇。\n\n"
            "建議輔導人員立即採取：\n"
            "1. 優先確認個案當下人身安全與情緒穩定，避免單獨留置風險。\n"
            "2. 依貴校／機構危機處理與兒少保護責任通報流程啟動評估與轉介。\n"
            f"3. 立即聯絡專業資源：\n{res_lines}"
        )
        return {
            "status": "ok",
            "engineMode": "local_consult_v3",
            "reply": reply,
            "safetyAlert": safety_alert,
            "suggestedObject": None,
            "suggestedNote": None,
        }

    pri_label = str(report.get("primaryLabel") or "主要傾向")[:40]
    pri_score = int(_safe_float(report.get("primaryScore"), 0))
    sec_label = str(report.get("secondaryLabel") or "次要傾向")[:40]
    sec_score = int(_safe_float(report.get("secondaryScore"), 0))
    conv_label = str(report.get("convergenceLabel") or "探索性參考")[:60]
    age_label = str(report.get("ageNormLabel") or "未指定（3–13 歲通用）")[:40]
    root_summary = str(report.get("reasoningSummary") or "")[:420]
    alt_note = str(report.get("alternativeNote") or "同一特徵也可能源自畫技習慣或工具探索（Thomas & Jolley, 1998）。")[:260]
    strengths_note = str(report.get("strengthsNote") or "")[:260]
    stroke_insight = str(report.get("strokeInsight") or "")[:260]
    figures_list = [str(f)[:40] for f in list(report.get("recognizedFigures") or [])[:6] if isinstance(f, str)]

    q_lower = question.lower()
    suggested_obj: Optional[str] = None
    suggested_note: Optional[str] = None
    q_frameworks = _match_image_first_frameworks([question])

    # 1. Counselor clarifying ambiguous drawn content
    clarify_markers = ("其實是", "畫的是", "代表", "＝", "=", "左下", "右下", "左上", "右上", "中間那", "角落那", "那一塊", "模糊", "補充辨識")
    if any(m in question for m in clarify_markers):
        cleaned = question.replace("我補充一下", "").replace("補充辨識", "").strip(" ：:，,。")
        if cleaned and len(cleaned) <= 72:
            suggested_obj = cleaned
        else:
            suggested_note = question[:180]
        matched_dirs = []
        for emo in EMOTION_KEYS:
            cues = [w for w in COUNSELOR_OBJECT_CUES.get(emo, ()) if w in q_lower]
            if cues:
                matched_dirs.append(f"{EMOTION_PROFILES[emo]['short_label']}（線索：{'、'.join(cues[:3])}）")
        cue_line = (
            f"此描述在兒童繪畫情境脈絡中可能呼應：{'；'.join(matched_dirs)}。"
            if matched_dirs
            else "此元素屬高度個人化敘事象徵，建議以孩子自己的故事定義為準，避免直接套用制式符號表。"
        )
        fw_block = ""
        follow_prompt = "「你剛剛說這裡畫的是這個角色／東西，如果它現在能講一句話，它最想說什麼？」"
        if q_frameworks:
            fw0 = q_frameworks[0]
            fw_block = f"• 現代實證文獻對照（{fw0['citation']}）：{fw0['summary']}\n"
            follow_prompt = fw0["promptTip"]
        reply = (
            "【協助圖像辨識與脈絡校準】\n"
            "感謝您補充畫面實際內容！依 Picard 等人（2007）、Gross & Hayne（1998）與 Driessnack（2005, d = 0.95）實證，創作者親口澄清並經輔導人員確認的物件意義，準確度遠高於純幾何推估。\n"
            f"• 脈絡檢視：{cue_line}\n"
            f"{fw_block}"
            f"• 建議延伸問法：{follow_prompt}\n"
            "• 下一步：您可以點擊上方「納入補充資訊並重新解析」，系統會將您確認的元素寫入判斷樹狀圖與互動引導分頁。"
        )
    # 2. Tablet / Touchpad / Undo / Multi-stroke inquiry (Price et al., 2015; Mohd Shukri, 2014)
    elif any(k in question for k in ("平板", "觸控", "誤觸", "多段", "短筆", "一直復原", "一直擦", "重塗")):
        reply = (
            "【3C 數位媒材與平板作畫去偏說明（Price et al., 2015; Mohd Shukri, 2014）】\n"
            "• 實證依據：Price 等人（2015）與 Mohd Shukri（2014）證實兒童在平板／觸控板上作畫時，為避免畫錯常採「多段短筆畫」與頻繁復原，也常因手指接觸滑順而產生疊畫或誤觸。\n"
            "• 系統處理：心畫室 v3.5 已將復原、斷續短線、疊畫、微震顫與停頓的情緒加權全面調降 55%–62%，並停用滑鼠／手指觸控的筆壓情緒推論。\n"
            "• 建議操作：請直接開啟上方「作畫歷程縮時回放檢視」並勾選「標示重複描繪／修改熱區」，先看清楚底層原本畫了什麼，再以開放問句邀請孩子分享。"
        )
    # 3. Explaining scoring, trade-off, or alternative non-emotional hypotheses
    elif any(k in question for k in ("為什麼", "為何", "依據", "分數", "取捨", "次高", "替代", "誤判", "判斷", "差異", "邏輯")):
        gap = abs(pri_score - sec_score)
        gap_txt = (
            f"主要傾向「{pri_label}（{pri_score}）」領先次高「{sec_label}（{sec_score}）」達 {gap} 分，視覺／行為證據相對集中。"
            if gap >= 12
            else f"主要傾向「{pri_label}（{pri_score}）」與次高「{sec_label}（{sec_score}）」僅差 {gap} 分，顯示感受具雙重交織性，解讀時不宜只看單一標籤。"
        )
        reply = (
            "【判斷依據與多元替代詮釋（29 篇國際實證核對）】\n"
            f"• 目前結論與差距：{gap_txt}（證據收斂：{conv_label}；常模：{age_label}）\n"
            f"• 模型推理摘要：{root_summary or stroke_insight or '系統採完整圖形與表情優先，並已調降數位操作權重與移除空間方位心理推論。'}\n"
            f"• 非情緒替代解釋（Thomas & Jolley, 1998; Skybo et al., 2007; Ivashkevich, 2009）：{alt_note}\n"
            f"• 正向資源檢視（Pénzes, 2020; Bat Or et al., 2022）：{strengths_note or '畫面仍展現創作者運用視覺媒介主動表達與整理經驗的彈性。'}\n"
            "• 實務建議：依 Joiner (1996)、Catte & Cox (1999) 與 Pace 等人（2022）整體量表原則，請把上述分數視為『假設清單』，並可先播放上方作畫縮時回放再進行訪談。"
        )
    # 4. Child is silent, says "不知道", or avoids answering
    elif any(k in question for k in ("不知道", "不想說", "沉默", "不講", "抗拒", "逃避", "怎麼接", "怎麼問", "下一步", "接話", "沒嘴巴", "沒門窗")):
        focus_obj = verified_objects[0] if verified_objects else (figures_list[0] if figures_list else "畫裡最顯眼的那個形狀")
        reply = (
            "【面對沉默／「不知道」之非誘導接話策略（Gross & Hayne, 1998; Driessnack, 2005; Malchiodi, 1998）】\n"
            "孩子說「不知道」常不是刻意隱瞞，而是還沒找到合適詞彙，或對直接被問「你是不是很……？」感到壓力。Driessnack（2005）統合分析證實把焦點從「審視孩子」轉移到「一起看畫」可達 d = 0.95 的溝通促進效果。\n"
            "建議改用『降壓＋第三人稱投射』三步驟：\n"
            "1. 先接納沉默：「沒關係喔，不用急著想答案，你願意把它畫下來就已經很棒了。」\n"
            f"2. 回到具體細節而非情緒：「我看見這裡有『{focus_obj}』，如果幫這幅畫配一種聲音或背景音樂，你覺得會像什麼聲音？」\n"
            "3. 給選擇題或共創邀約（Willis et al., 2010; Bat Or et al., 2022）：「如果你想幫畫裡加一把保護傘、耳機、秘密基地或小幫手，下次我們可以畫在旁邊嗎？」\n"
            "• 提醒：每次只問 1 題並給足 5–10 秒靜默等待，切勿連環追問或替孩子猜情緒。"
        )
    # 5. Summarize current notes / prepare for re-analysis
    elif any(k in question for k in ("整理", "摘要", "重點", "回填", "總結")):
        obj_txt = "、".join(verified_objects) if verified_objects else "（尚未填寫協助辨識物件）"
        note_txt = follow_up_notes if follow_up_notes else "（尚未填寫後續訪談紀錄）"
        reply = (
            "【目前輔導補充脈絡整理】\n"
            f"• 系統原本辨識圖形：{'、'.join(figures_list) if figures_list else '無固定圖式（以低權重動力與空間參考）'}\n"
            f"• 輔導員補充辨識物件：{obj_txt}\n"
            f"• 後續溝通／訪談紀錄：{note_txt}\n"
            "• 建議操作：確認上方兩欄已補上孩子實際說明的物件與反應後，點擊『納入補充資訊並重新解析』，即可同步刷新「內在情緒分析」與「圖像互動引導」兩個分頁。"
        )
    # 6. Pure image-first / pop-culture / neurodiversity framework inquiry
    elif q_frameworks:
        fw0 = q_frameworks[0]
        suggested_obj = question.strip(" ：:，,。？?")[:70] if len(question) <= 72 else None
        suggested_note = None if suggested_obj else question[:180]
        reply = (
            f"【現代實證脈絡解析：{fw0['label']}】\n"
            f"• 對應文獻：{fw0['citation']}\n"
            f"• 圖像涵義依據：{fw0['summary']}\n"
            f"• 建議非誘導開放問法：{fw0['promptTip']}\n"
            "• 實務提醒：建議將孩子對此物件／位置的親口說法填入上方補充欄並按重新解析，系統會以『整體構圖＋口述脈絡』同步更新雙分頁。"
        )
    else:
        suggested_note = question[:180] if len(question) >= 8 else None
        reply = (
            "【本地實證諮詢摘要（免金鑰模式 A・29 篇國際期刊依據）】\n"
            f"• 當前作品焦點：主要傾向「{pri_label}（{pri_score}）」、次高「{sec_label}（{sec_score}）」，收斂度為「{conv_label}」（常模：{age_label}）。\n"
            f"• 筆觸與正向資源：{stroke_insight or '已採數位媒材降權處理。'} {strengths_note}\n"
            "• 建議後續探索方向：\n"
            "  1. 可先播放上方「作畫歷程縮時回放」並開啟熱區標示，確認底層圖案與重複修改位置。\n"
            "  2. 若畫中有動漫角色、機械細節、保護物（耳機／棉被／雨傘）或模糊元素，可填入上方『協助辨識畫面元素』並重算。\n"
            "（提示：若您已在右上角啟用方案 B 個人 Gemini API Key，本諮詢區會自動升級為可檢視完整畫作影像的開放式多模態 AI 顧問。）"
        )

    return {
        "status": "ok",
        "engineMode": "local_consult_v3",
        "reply": reply,
        "safetyAlert": None,
        "suggestedObject": suggested_obj,
        "suggestedNote": suggested_note,
    }

