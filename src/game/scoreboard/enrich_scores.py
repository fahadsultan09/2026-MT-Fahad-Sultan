"""
Tennis Score Enrichment Pipeline
==================================
Reads your existing match XML (with points + hits), runs OCR on
saved scoreboard crops, then injects the correct score into each
<point> and <game> element based on timestamp matching.

Workflow:
  1. Run YOLO with save_crop=True  →  crops saved to disk
  2. Run this script               →  enriched XML output

Run from project root:
    python3 src/game/scoreboard/enrich_scores.py

Install:
    pip install paddleocr paddlepaddle opencv-python lxml
"""

import glob
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from copy import deepcopy
from xml.dom import minidom

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# Your existing match XML
INPUT_XML   = "data/processed/fahad_sultan_output"

# Where YOLO saved the scoreboard crops (save_crop=True)
CROPS_DIR   = "runs/detect/ocr_ready/crops/boardT/output3"

# Output enriched XML
OUTPUT_XML  = "Jabeur_with_scores.xml"

# Video FPS (used to convert frame number in filename → seconds)
VIDEO_FPS   = 25

# Temporal filter settings
TF_WINDOW    = 10
TF_THRESHOLD = 0.6


# ─────────────────────────────────────────────────────────────
# 1.  OCR ENGINE
# ─────────────────────────────────────────────────────────────

print("Loading PaddleOCR...")
_ocr = PaddleOCR(use_textline_orientation=True, lang="en")


# ─────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess(crop: np.ndarray) -> np.ndarray:
    h, w   = crop.shape[:2]
    img    = cv2.resize(crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    gray   = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=8,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────
# 3.  OCR
# ─────────────────────────────────────────────────────────────

def run_ocr(image: np.ndarray) -> list[dict]:
    results = _ocr.ocr(image)
    blocks  = []
    if not results or not results[0]:
        return blocks
    for line in results[0]:
        # New PaddleOCR returns: [bbox, (text, conf)]
        # Older versions: same structure but accessed differently
        try:
            bbox, (text, conf) = line
        except (TypeError, ValueError):
            continue
        cx = (bbox[0][0] + bbox[2][0]) / 2
        cy = (bbox[0][1] + bbox[2][1]) / 2
        blocks.append({"text": text.strip(), "confidence": conf,
                        "bbox": bbox, "cx": cx, "cy": cy})
    blocks.sort(key=lambda b: (round(b["cy"]/10), b["cx"]))
    return blocks


# ─────────────────────────────────────────────────────────────
# 4.  CHARACTER CORRECTION
# ─────────────────────────────────────────────────────────────

_CHAR_FIXES = {"O":"0","o":"0","I":"1","l":"1","|":"1",
               "S":"5","G":"6","B":"8","Z":"2"}
_POINT_MAP  = {"0":"0","O":"0","15":"15","30":"30","40":"40",
               "AD":"AD","A":"AD","ADV":"AD","DEUCE":"40"}

# Numeric index used in the XML output:
#   0  →  0   (love)
#   15 →  1
#   30 →  2
#   40 →  3
#   AD →  4
_POINT_INDEX = {"0": "0", "15": "1", "30": "2", "40": "3", "AD": "4"}
_SKIP_WORDS = {
    "SET","SETS","GAME","GAMES","AD","ADV","DEUCE",
    "TUN","AUS","USA","GBR","FRA","ESP","GER","ITA",
    "SRB","POL","ROU","CZE","ARG","JPN","CHN","SUI",
    "BEL","NED","CRO","SVK","UKR","KAZ","CAN","BRA",
    "POR","HUN","GRE","SWE","NOR","DEN","FIN","AUT",
}
_SERVER_TOKENS = {"→","►",">","●","•","*","->","⇒"}

def _fix(t):
    return _CHAR_FIXES.get(t, t) if len(t) == 1 else t

def _norm_point(t):
    return _POINT_MAP.get(t.upper().strip())


# ─────────────────────────────────────────────────────────────
# 5.  ROW SPLITTER + PARSER
# ─────────────────────────────────────────────────────────────

_SET_RE   = re.compile(r"\b([0-7])\s*[-–]\s*([0-7])\b")
_GAME_RE  = re.compile(r"\b([0-6])\b")
_POINT_RE = re.compile(r"\b(0|15|30|40|AD|ADV|A|DEUCE)\b", re.IGNORECASE)


def _split_rows(blocks):
    if not blocks: return [], []
    ys = sorted(set(round(b["cy"]/5)*5 for b in blocks))
    if len(ys) < 2: return blocks, []
    gaps   = [(ys[i+1]-ys[i], (ys[i]+ys[i+1])/2) for i in range(len(ys)-1)]
    _, mid = max(gaps)
    return [b for b in blocks if b["cy"] <= mid], [b for b in blocks if b["cy"] > mid]


def _parse_row(blocks):
    texts = [_fix(b["text"]) for b in blocks]
    raw   = " ".join(texts)
    result = {"name": None, "nationality": None,
              "games": None, "points": None, "is_serving": False}

    if any(b["text"] in _SERVER_TOKENS for b in blocks):
        result["is_serving"] = True

    name_parts     = [b["text"] for b in blocks
                      if re.match(r"^[A-Za-z]{2,}$", b["text"])
                      and b["text"].upper() not in _SKIP_WORDS]
    nat_candidates = [p for p in name_parts if p.isupper() and 2 <= len(p) <= 3]
    name_cands     = [p for p in name_parts if p not in nat_candidates]

    if name_cands:     result["name"]        = " ".join(name_cands)
    if nat_candidates: result["nationality"] = nat_candidates[-1]

    for pm in reversed(_POINT_RE.findall(raw)):
        val = _norm_point(pm)
        if val:
            result["points"] = val
            break

    stripped    = _POINT_RE.sub("", raw)
    game_digits = [int(g) for g in _GAME_RE.findall(stripped) if int(g) <= 6]
    if game_digits:
        result["games"] = game_digits[-1]

    return result


def parse_score(blocks: list[dict]) -> dict:
    r1b, r2b = _split_rows(blocks)
    r1, r2   = _parse_row(r1b), _parse_row(r2b)

    full_raw = " ".join(_fix(b["text"]) for b in blocks)
    sets = []
    for m in _SET_RE.finditer(full_raw):
        p1g, p2g = int(m.group(1)), int(m.group(2))
        if _valid_set(p1g, p2g):
            sets.append((p1g, p2g))

    server = 1 if r1["is_serving"] else (2 if r2["is_serving"] else None)

    return {
        "player1": r1["name"] or "A",
        "player2": r2["name"] or "B",
        "sets": sets,
        "current_games":  (r1["games"], r2["games"]),
        "current_points": (r1["points"], r2["points"]),
        "server": server,
    }


def _valid_set(p1, p2):
    if p1 == 7 and p2 in (5, 6): return True
    if p2 == 7 and p1 in (5, 6): return True
    if p1 == 6 and p2 <= 5:      return True
    if p2 == 6 and p1 <= 5:      return True
    return p1 <= 5 and p2 <= 5


# ─────────────────────────────────────────────────────────────
# 6.  TEMPORAL FILTER
# ─────────────────────────────────────────────────────────────

def _is_valid_transition(prev, curr):
    if prev is None: return True
    if len(curr["sets"]) < len(prev["sets"]): return False
    pg, cg = prev["current_games"], curr["current_games"]
    if None not in pg and None not in cg:
        if abs((cg[0] or 0) - (pg[0] or 0)) > 1: return False
        if abs((cg[1] or 0) - (pg[1] or 0)) > 1: return False
    return True


class TemporalFilter:
    def __init__(self, window=10, threshold=0.6):
        self.window    = window
        self.threshold = threshold
        self._buf:    list = []
        self._stable       = None

    def update(self, score):
        if score is None: return self._stable
        self._buf.append(score)
        if len(self._buf) > self.window: self._buf.pop(0)
        candidate = self._majority()
        if candidate and _is_valid_transition(self._stable, candidate):
            self._stable = candidate
        return self._stable

    def _majority(self):
        if not self._buf: return None
        def key(s):
            return (str(s["current_points"]), str(s["current_games"]), str(s["sets"]))
        counts = Counter(key(s) for s in self._buf)
        top, n = counts.most_common(1)[0]
        if n / len(self._buf) >= self.threshold:
            for s in reversed(self._buf):
                if key(s) == top: return s
        return None


# ─────────────────────────────────────────────────────────────
# 7.  TIMESTAMP HELPERS
# ─────────────────────────────────────────────────────────────

def timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS or MM:SS string to total seconds."""
    parts = ts.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0]*3600 + parts[1]*60 + parts[2]
    elif len(parts) == 2:
        return parts[0]*60 + parts[1]
    return float(parts[0])


def filename_to_absolute_seconds(filename: str, fps: float = 25.0,
                                  chunk_duration: float = 60.0,
                                  overlap: float = 10.0) -> float | None:
    """
    Convert a crop filename like 'output3_47.jpg' to absolute video seconds.

    Video part structure:
        output1  starts at 0s
        output2  starts at (60 - 10) = 50s
        output3  starts at (60 - 10) * 2 = 100s
        outputN  starts at (N-1) * (chunk_duration - overlap)

    Filename format: output{part}_{frame}.jpg
        part  = which 1-minute chunk (1-based)
        frame = frame number within that chunk (from vid_stride sampling)

    Returns absolute seconds in the full video.
    """
    # Match e.g. "output3_47" or "output3_frame47"
    m = re.match(r"output(\d+)[_](?:frame)?(\d+)", filename, re.IGNORECASE)
    if not m:
        return None

    part_num  = int(m.group(1))   # which video chunk (1-based)
    frame_num = int(m.group(2))   # frame index within chunk

    # Absolute start of this chunk in the full video
    chunk_start = (part_num - 1) * (chunk_duration - overlap)

    # Time within this chunk
    time_in_chunk = frame_num / fps

    return chunk_start + time_in_chunk


# ─────────────────────────────────────────────────────────────
# 8.  BUILD SCORE TIMELINE FROM CROPS
# ─────────────────────────────────────────────────────────────

def build_score_timeline(crops_dir: str, fps: float = 25.0,
                         chunk_duration: float = 60.0,
                         overlap: float = 10.0) -> list[tuple[float, dict]]:
    """
    Process all saved crop images and return a timeline:
        [(absolute_seconds, stable_score_dict), ...]
    sorted by absolute time, only including frames where score changed.

    Handles overlapping video chunks correctly — crops from the overlap
    region of chunk N+1 are deduplicated against chunk N using the
    absolute timestamp.
    """
    pattern = os.path.join(crops_dir, "*.jpg")
    crops   = sorted(glob.glob(pattern))
    if not crops:
        crops = sorted(glob.glob(os.path.join(crops_dir, "*.png")))

    if not crops:
        raise FileNotFoundError(
            f"No crop images found in {crops_dir}\n"
            "Make sure model.predict() ran with save_crop=True"
        )

    # Convert all filenames to absolute seconds first, then sort chronologically
    # This is critical — glob sort is lexicographic, not temporal
    timed_crops = []
    skipped     = 0
    for path in crops:
        fname   = os.path.basename(path)
        seconds = filename_to_absolute_seconds(
            fname, fps=fps,
            chunk_duration=chunk_duration,
            overlap=overlap,
        )
        if seconds is None:
            skipped += 1
            continue
        timed_crops.append((seconds, path))

    timed_crops.sort(key=lambda x: x[0])   # sort by absolute time

    if skipped:
        print(f"  [INFO] Skipped {skipped} files with unrecognised filename format")

    print(f"Found {len(timed_crops)} crop images — running OCR...")

    tf         = TemporalFilter(window=TF_WINDOW, threshold=TF_THRESHOLD)
    timeline   = []
    last        = None
    seen_times  = set()   # deduplicate overlapping frames

    for seconds, path in timed_crops:
        # Round to nearest 0.2s to deduplicate overlap region crops
        rounded = round(seconds * 5) / 5
        if rounded in seen_times:
            continue
        seen_times.add(rounded)

        crop = cv2.imread(path)
        if crop is None:
            continue

        blocks = run_ocr(preprocess(crop))
        score  = parse_score(blocks) if blocks else None
        stable = tf.update(score)

        if stable and stable != last:
            timeline.append((seconds, deepcopy(stable)))
            last = stable

    print(f"Built score timeline: {len(timeline)} score changes detected")
    return timeline


# ─────────────────────────────────────────────────────────────
# 9.  SCORE LOOKUP — find closest score for a given timestamp
# ─────────────────────────────────────────────────────────────

def lookup_score(timeline: list[tuple[float, dict]], seconds: float) -> dict | None:
    """
    Return the most recent score that was active at `seconds`.
    i.e. the last score change that happened AT or BEFORE this timestamp.
    """
    best = None
    for t, score in timeline:
        if t <= seconds:
            best = score
        else:
            break   # timeline is sorted
    return best


# ─────────────────────────────────────────────────────────────
# 10.  XML ENRICHMENT
# ─────────────────────────────────────────────────────────────

def enrich_xml(input_path: str, timeline: list, output_path: str):
    """
    Read the existing match XML, inject score_A / score_B into
    each <point> and <game> based on timestamp, then save.
    """
    tree = ET.parse(input_path)
    root = tree.getroot()

    points_enriched = 0
    points_missing  = 0

    for set_el in root.findall("set"):
        for game_el in set_el.findall("game"):
            game_score_a = None
            game_score_b = None

            for point_el in game_el.findall("point"):
                # Use first hit timestamp as the point's reference time
                first_hit = point_el.find("hit")
                if first_hit is None:
                    points_missing += 1
                    continue

                time_str = first_hit.get("time", "")
                if not time_str:
                    points_missing += 1
                    continue

                seconds = timestamp_to_seconds(time_str)
                score   = lookup_score(timeline, seconds)

                if score:
                    p1, p2 = score["current_points"]
                    g1, g2 = score["current_games"]

                    # Inject point-level score as numeric index
                    # 0→0, 15→1, 30→2, 40→3, AD→4
                    point_el.set("score_A", _POINT_INDEX.get(p1, "") if p1 else "")
                    point_el.set("score_B", _POINT_INDEX.get(p2, "") if p2 else "")

                    # Track game score for this game element
                    if g1 is not None: game_score_a = str(g1)
                    if g2 is not None: game_score_b = str(g2)

                    # Inject server
                    if score.get("server"):
                        point_el.set("service",
                                     "A" if score["server"] == 1 else "B")

                    points_enriched += 1
                else:
                    points_missing += 1

            # Inject game-level score (from last point in this game)
            if game_score_a is not None:
                game_el.set("score_A", game_score_a)
            if game_score_b is not None:
                game_el.set("score_B", game_score_b)

        # Inject set-level score from the sets list in the last detected score
        last_score = timeline[-1][1] if timeline else None
        if last_score and last_score["sets"]:
            set_num = int(set_el.get("id", 1)) - 1
            if set_num < len(last_score["sets"]):
                g1, g2 = last_score["sets"][set_num]
                set_el.set("score_A", str(g1))
                set_el.set("score_B", str(g2))

    # Pretty-print and save
    raw     = ET.tostring(root, encoding="unicode")
    pretty  = minidom.parseString(raw).toprettyxml(indent="  ")

    # Remove duplicate XML declaration added by toprettyxml
    lines   = pretty.split("\n")
    if lines[0].startswith("<?xml"):
        lines = lines[1:]
    final   = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version=\'1.0\' encoding=\'utf-8\'?>\n')
        f.write(final)

    print(f"\nEnriched XML saved to: {output_path}")
    print(f"  Points enriched : {points_enriched}")
    print(f"  Points missing  : {points_missing}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Build score timeline from saved crop images
    # chunk_duration = 60s per part, overlap = 10s between parts
    timeline = build_score_timeline(
        CROPS_DIR,
        fps=VIDEO_FPS,
        chunk_duration=60.0,
        overlap=10.0,
    )

    if not timeline:
        print("No stable scores found. Check your crops directory and OCR output.")
        raise SystemExit(1)

    # Step 2: Enrich the existing match XML with scores
    enrich_xml(INPUT_XML, timeline, OUTPUT_XML)

    # Preview first few score changes
    print("\nScore timeline preview (first 5 changes):")
    for t, s in timeline[:5]:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        sec = int(t % 60)
        pts = s["current_points"]
        gms = s["current_games"]
        print(f"  {h:02d}:{m:02d}:{sec:02d}  "
              f"games=({gms[0]}-{gms[1]})  "
              f"points=({pts[0]}-{pts[1]})")
    # Step 1: Build score timeline from saved crop images
    timeline = build_score_timeline(CROPS_DIR, fps=VIDEO_FPS)

    if not timeline:
        print("No stable scores found. Check your crops directory and OCR output.")
        raise SystemExit(1)

    # Step 2: Enrich the existing match XML with scores
    enrich_xml(INPUT_XML, timeline, OUTPUT_XML)

    # Preview first few score changes
    print("\nScore timeline preview (first 5 changes):")
    for t, s in timeline[:5]:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        sec = int(t % 60)
        pts = s["current_points"]
        gms = s["current_games"]
        print(f"  {h:02d}:{m:02d}:{sec:02d}  "
              f"games=({gms[0]}-{gms[1]})  "
              f"points=({pts[0]}-{pts[1]})")