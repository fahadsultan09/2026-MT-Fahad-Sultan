"""
Efficient Tennis Score Extractor
==================================
Instead of processing every crop frame, this script:
  1. Reads the match XML to find the LAST HIT timestamp of each point
  2. Seeks directly to that frame in the video
  3. Runs YOLO to detect the scoreboard
  4. Runs OCR only on that one crop
  5. Injects the real score back into the XML
  6. Regroups points into real <game>/<set> boundaries using the OCR'd
     point score (love/15/30/40/AD), which is far more reliable than the
     on-screen games-won tally for detecting where a game actually ends.

This is ~50x faster than processing every frame.

Run:
    python3 efficient_score_extractor.py

Install:
    pip install paddleocr paddlepaddle opencv-python ultralytics lxml
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import os


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

INPUT_XML    = "../../../data/processed/jabeur_xml_upgraded.xml"
OUTPUT_XML   = "Jabeur_scored3.xml"
VIDEO_FILE   = "../../../video/jabeur.mp4"
YOLO_MODEL   = "../../../runs/detect/train/weights/best.pt"
YOLO_CONF    = 0.5
VIDEO_FPS    = 25.0
DEBUG_DIR    = "debug"

# Seconds AFTER the last hit to sample the frame
# (scoreboard updates slightly after the point ends)
SCORE_OFFSET = 1.5


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

CHAR_FIXES = {
    "O": "0", "o": "0", "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5", "G": "6", "g": "6", "B": "8", "Z": "2", "z": "2",
}
POINT_CANONICAL = {
    "0": "0", "O": "0", "15": "15", "30": "30", "40": "40",
    "AD": "AD", "A": "AD", "ADV": "AD", "DEUCE": "AD",
}
# Point value → numeric index used in XML
POINT_INDEX = {"0": "0", "15": "1", "30": "2", "40": "3", "AD": "4"}

NATIONALITY_CODES = {
    "TUN", "AUS", "USA", "GBR", "FRA", "ESP", "GER", "ITA", "SRB", "POL",
    "ROU", "CZE", "ARG", "JPN", "CHN", "SUI", "BEL", "NED", "CRO", "SVK",
    "UKR", "KAZ", "CAN", "BRA", "POR", "HUN", "GRE", "SWE", "NOR", "DEN",
    "FIN", "AUT", "RUS", "RSA", "MEX", "COL", "CHI", "IND",
}
SKIP_WORDS = NATIONALITY_CODES | {
    "SET", "SETS", "GAME", "GAMES", "AD", "ADV", "DEUCE", "MATCH",
}
SERVER_TOKENS = {"→", "►", ">", "●", "•", "*", "->", "⇒", "▶"}

_POINT_RE = re.compile(r"\b(0|15|30|40|AD|ADV|A|DEUCE)\b", re.IGNORECASE)
_SET_RE   = re.compile(r"\b([0-7])\s*[-–]\s*([0-7])\b")
_GAME_RE  = re.compile(r"\b([0-6])\b")


# ═══════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScoreReading:
    player_a: Optional[str] = None
    player_b: Optional[str] = None
    nat_a:    Optional[str] = None
    nat_b:    Optional[str] = None
    server:   Optional[str] = None     # "A" or "B"
    sets:     list = field(default_factory=list)   # [{"a":6,"b":4},...]
    games:    dict = field(default_factory=dict)   # {"a":3,"b":2}
    points:   dict = field(default_factory=dict)   # {"a":"30","b":"0"}
    raw_text: list = field(default_factory=list)

    def point_index(self, player: str) -> str:
        """Return numeric point index (0/1/2/3/4) for player 'a' or 'b'."""
        val = self.points.get(player)
        return POINT_INDEX.get(val, "") if val else ""


# ═══════════════════════════════════════════════════════════════
# TIMESTAMP HELPERS
# ═══════════════════════════════════════════════════════════════

def ts_to_seconds(ts: str) -> float:
    """'HH:MM:SS' or 'MM:SS' → float seconds."""
    parts = [float(p) for p in ts.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def seconds_to_frame(seconds: float, fps: float) -> int:
    return int(seconds * fps)


def format_ts(seconds: float) -> str:
    """Convert seconds → safe filename timestamp like 01h02m03s"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


# ═══════════════════════════════════════════════════════════════
# XML READER — extract last-hit timestamps per point
# ═══════════════════════════════════════════════════════════════

def extract_rally_endpoints(xml_path: str) -> list[dict]:
    """
    Parse the match XML and return a list of rally endpoints.

    Each entry:
        {
          set_id:    int,
          game_id:   int,
          point_id:  int,
          last_hit_time: float  (seconds),
          point_el:  ET.Element  (reference for later injection)
          game_el:   ET.Element
          set_el:    ET.Element
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    endpoints = []

    for set_el in root.findall("set"):
        sid = int(set_el.get("id", 0))
        for game_el in set_el.findall("game"):
            gid = int(game_el.get("id", 0))
            for point_el in game_el.findall("point"):
                pid = int(point_el.get("id", 0))
                hits = point_el.findall("hit")
                if not hits:
                    continue
                # Last hit = end of rally
                if len(hits) >= 2:
                    target_hit = hits[-2]   # second last hit
                else:
                    target_hit = hits[-1]   # fallback

                last_time_str = target_hit.get("time", "")
                if not last_time_str:
                    continue
                last_seconds = ts_to_seconds(last_time_str)
                endpoints.append({
                    "set_id":        sid,
                    "game_id":       gid,
                    "point_id":      pid,
                    "last_hit_time": last_seconds,
                    "last_hit_str":  last_time_str,
                    "point_el":      point_el,
                    "game_el":       game_el,
                    "set_el":        set_el,
                })

    print(f"Found {len(endpoints)} rally endpoints in {xml_path}")
    return tree, root, endpoints


# ═══════════════════════════════════════════════════════════════
# FRAME EXTRACTOR — seek to exact frame in video
# ═══════════════════════════════════════════════════════════════

class VideoSeeker:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # ✅ use actual FPS from video (fix drift)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Opened video: {video_path}")
        print(f"Detected FPS: {self.fps}")

    def get_frame(self, seconds: float):
        # ✅ accurate time-based seeking
        self.cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        self.cap.release()


# ═══════════════════════════════════════════════════════════════
# SCOREBOARD DETECTOR — YOLO crop from frame
# ═══════════════════════════════════════════════════════════════

class ScoreboardDetector:
    def __init__(self, model_path: str, conf: float = 0.5):
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf

    def crop(self, frame: np.ndarray, padding: int = 6) -> Optional[np.ndarray]:
        """
        Run YOLO on frame, return the highest-confidence scoreboard crop.
        Returns None if no scoreboard detected above threshold.
        """
        results = self.model(frame, verbose=False)
        best_crop = None
        best_conf = 0.0

        for result in results:
            for box in result.boxes:
                c = float(box.conf)
                if c < self.conf or c <= best_conf:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                fh, fw = frame.shape[:2]
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(fw, x2 + padding)
                y2 = min(fh, y2 + padding)
                if x2 > x1 and y2 > y1:
                    best_crop = frame[y1:y2, x1:x2].copy()
                    best_conf = c

        return best_crop


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def preprocess(img: np.ndarray) -> np.ndarray:
    """US Open dark-blue scoreboard: Otsu threshold works best."""
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(bin_, cv2.COLOR_GRAY2BGR)


# ═══════════════════════════════════════════════════════════════
# OCR ENGINE
# ═══════════════════════════════════════════════════════════════

class OCREngine:
    def __init__(self):
        print("Loading PaddleOCR...")
        self._ocr = PaddleOCR(use_textline_orientation=True, lang="en")

    def read(self, image: np.ndarray) -> list[dict]:
        results = self._ocr.ocr(image)
        blocks = []
        if not results or not results[0]:
            return blocks
        for line in results[0]:
            try:
                bbox, (text, conf) = line
            except (TypeError, ValueError):
                continue
            text = text.strip()
            if not text:
                continue
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            blocks.append({"text": text, "conf": conf, "cx": cx, "cy": cy, "bbox": bbox})
        blocks.sort(key=lambda b: (round(b["cy"] / 8), b["cx"]))
        return blocks


# ═══════════════════════════════════════════════════════════════
# SCORE PARSER
# ═══════════════════════════════════════════════════════════════

def _fix(t: str) -> str:
    return CHAR_FIXES.get(t, t) if len(t) == 1 else t


def _canon_point(t: str) -> Optional[str]:
    return POINT_CANONICAL.get(t.upper().strip())


def _is_nat(t: str) -> bool:
    return t.upper() in NATIONALITY_CODES


def _is_name(t: str) -> bool:
    return (re.match(r"^[A-Za-z]{2,}$", t) is not None
            and t.upper() not in SKIP_WORDS)


def _split_rows(blocks):
    if not blocks:
        return [], []
    ys = sorted(set(round(b["cy"] / 5) * 5 for b in blocks))
    if len(ys) < 2:
        return blocks, []
    gaps = [(ys[i + 1] - ys[i], (ys[i] + ys[i + 1]) / 2) for i in range(len(ys) - 1)]
    _, mid = max(gaps)
    return [b for b in blocks if b["cy"] <= mid], [b for b in blocks if b["cy"] > mid]


def _parse_row(blocks) -> dict:
    fixed = [_fix(b["text"]) for b in blocks]
    raw = " ".join(fixed)
    r = {"name": None, "nat": None, "games": None, "points": None, "is_serving": False}

    if any(b["text"] in SERVER_TOKENS for b in blocks):
        r["is_serving"] = True

    nat_toks = [b["text"] for b in blocks if _is_nat(b["text"])]
    name_toks = [b["text"] for b in blocks if _is_name(b["text"]) and not _is_nat(b["text"])]
    if name_toks:
        r["name"] = " ".join(name_toks).upper()
    if nat_toks:
        r["nat"] = nat_toks[-1].upper()

    for pm in reversed(_POINT_RE.findall(raw)):
        val = _canon_point(pm)
        if val:
            r["points"] = val
            break

    stripped = _POINT_RE.sub("", raw)
    game_digits = [int(g) for g in _GAME_RE.findall(stripped) if int(g) <= 6]
    if game_digits:
        r["games"] = game_digits[-1]

    return r


def parse_score(blocks: list[dict]) -> ScoreReading:
    ra_b, rb_b = _split_rows(blocks)
    ra, rb = _parse_row(ra_b), _parse_row(rb_b)

    full = " ".join(_fix(b["text"]) for b in blocks)
    sets = []
    for m in _SET_RE.finditer(full):
        a, b_ = int(m.group(1)), int(m.group(2))
        if _valid_set(a, b_):
            sets.append({"a": a, "b": b_})

    return ScoreReading(
        player_a=ra["name"],
        player_b=rb["name"],
        nat_a=ra["nat"],
        nat_b=rb["nat"],
        server="A" if ra["is_serving"] else ("B" if rb["is_serving"] else None),
        sets=sets,
        games={"a": ra["games"], "b": rb["games"]},
        points={"a": ra["points"], "b": rb["points"]},
        raw_text=[b["text"] for b in blocks],
    )


def _valid_set(a, b):
    if a == 7 and b in (5, 6):
        return True
    if b == 7 and a in (5, 6):
        return True
    if a == 6 and b <= 5:
        return True
    if b == 6 and a <= 5:
        return True
    return a <= 5 and b <= 5


# ═══════════════════════════════════════════════════════════════
# GAME/SET REGROUPING — boundary-detected from the POINT score,
# not the (noisy, laggy) games-won tally
# ═══════════════════════════════════════════════════════════════

def _point_score(sr: Optional[ScoreReading]):
    """(pa, pb) as point-index ints 0..4, or (None, None) if unreadable."""
    if sr is None:
        return (None, None)
    pa = sr.point_index("a")
    pb = sr.point_index("b")
    pa = int(pa) if pa not in ("", None) else None
    pb = int(pb) if pb not in ("", None) else None
    return pa, pb


def _games_tuple(sr: Optional[ScoreReading], prev):
    """Games-won tally — used only to LABEL a game, never to decide boundaries."""
    if sr is None:
        return prev
    ga, gb = sr.games.get("a"), sr.games.get("b")
    return (ga if ga is not None else prev[0], gb if gb is not None else prev[1])


def _is_valid_continuation(prev_pa, prev_pb, pa, pb):
    """True if (pa,pb) could plausibly follow (prev_pa,prev_pb) within the SAME game."""
    if prev_pa is None or prev_pb is None:
        return True
    # deuce (40) <-> AD (index 3 <-> 4) legally bounces both ways
    if {prev_pa, prev_pb} & {3, 4} and {pa, pb} & {3, 4}:
        return True
    # otherwise a point score never decreases within a game
    return pa >= prev_pa and pb >= prev_pb


def regroup_into_games(root: ET.Element, endpoints: list[dict], readings: list) -> None:
    """
    Rebuild <set>/<game>/<point> from OCR readings.

    Primary boundary signal: the per-point love/15/30/40/AD score. A game
    always starts at 0-0 and (outside the deuce/AD swing) never goes
    backwards, so a reset to 0-0 or any other decrease means a new game
    has begun. The on-screen "games won" tally is noisy and lags by a
    point or two, so it's used only to label each game's score_A/score_B,
    never to decide where the boundary falls.
    """
    for old in list(root):
        root.remove(old)

    cur_set_el = cur_game_el = None
    set_idx = game_idx = point_idx = 0
    prev_pa = prev_pb = None
    prev_games = (0, 0)

    for ep, sr in zip(endpoints, readings):
        point_el = ep["point_el"]
        pa, pb = _point_score(sr)
        games_now = _games_tuple(sr, prev_games)

        start_new_game = cur_game_el is None
        if not start_new_game and pa is not None and pb is not None:
            reset_to_love = (pa, pb) == (0, 0) and (prev_pa, prev_pb) not in (None, (0, 0))
            went_backwards = not _is_valid_continuation(prev_pa, prev_pb, pa, pb)
            start_new_game = reset_to_love or went_backwards

        # A new set starts alongside a new game, only if the games tally
        # itself also dropped back down (i.e. this isn't just game-to-game
        # within the same set)
        start_new_set = (
            start_new_game and cur_set_el is not None
            and (games_now[0] < prev_games[0] or games_now[1] < prev_games[1])
        )

        if cur_set_el is None or start_new_set:
            set_idx += 1
            game_idx = 0
            cur_set_el = ET.SubElement(root, "set", id=str(set_idx))

        if cur_game_el is None or start_new_game:
            game_idx += 1
            point_idx = 0
            cur_game_el = ET.SubElement(
                cur_set_el, "game", id=str(game_idx),
                score_A=str(games_now[0]), score_B=str(games_now[1]),
            )

        point_idx += 1
        point_el.set("id", str(point_idx))
        cur_game_el.append(point_el)

        if pa is not None:
            prev_pa = pa
        if pb is not None:
            prev_pb = pb
        prev_games = games_now


# ═══════════════════════════════════════════════════════════════
# XML WRITER
# ═══════════════════════════════════════════════════════════════

def pretty_xml(root: ET.Element) -> str:
    """Clean, consistent indentation using ET's built-in indenter (Python 3.9+)."""
    ET.indent(root, space="  ", level=0)
    raw = ET.tostring(root, encoding="unicode")
    return "<?xml version='1.0' encoding='utf-8'?>\n" + raw + "\n"


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run():
    # ── Load models ───────────────────────────────────────────
    detector = ScoreboardDetector(YOLO_MODEL, conf=YOLO_CONF)
    ocr = OCREngine()
    video = VideoSeeker(VIDEO_FILE)

    os.makedirs(DEBUG_DIR, exist_ok=True)

    # ── Parse XML, get rally endpoints ────────────────────────
    tree, root, endpoints = extract_rally_endpoints(INPUT_XML)

    total = len(endpoints)
    success = 0
    failed = 0
    all_readings: list = []   # one entry per endpoint, aligned by index; None = failed read

    print(f"\nProcessing {total} rally endpoints...\n")

    for i, ep in enumerate(endpoints):
        set_id = ep["set_id"]
        game_id = ep["game_id"]
        point_id = ep["point_id"]
        t_end = ep["last_hit_time"]
        t_sample = t_end   # sample after last hit
        # t_sample = t_end + SCORE_OFFSET   # sample 1.5s after last hit

        label = f"Set{set_id} Game{game_id} Point{point_id}"
        print(f"[{i+1:03d}/{total}] {label:25s} "
              f"last_hit={ep['last_hit_str']} sample={t_sample:.2f}s", end="  ")

        # ── 1. Seek to frame ──────────────────────────────────
        frame = video.get_frame(t_sample)
        if frame is None:
            # Try slightly earlier if we're past video end
            frame = video.get_frame(t_end - 0.5)
        if frame is None:
            print("→ NO FRAME")
            failed += 1
            all_readings.append(None)
            continue

        ts = format_ts(t_sample)
        frame_path = os.path.join(DEBUG_DIR, f"{i:03d}_{ts}_frame.jpg")
        ok = cv2.imwrite(frame_path, frame)
        if not ok:
            print("⚠️ Failed to save frame:", frame_path)

        # ── 2. YOLO crop ──────────────────────────────────────
        crop = detector.crop(frame)
        if crop is None:
            print("→ NO SCOREBOARD")
            frame_path = os.path.join(DEBUG_DIR, f"{i:03d}_{ts}_no_scoreboard.jpg")
            cv2.imwrite(frame_path, frame)
            failed += 1
            all_readings.append(None)
            continue

        # ── 3. Preprocess + OCR ───────────────────────────────
        blocks = ocr.read(preprocess(crop))
        if not blocks:
            print("→ NO TEXT")
            failed += 1
            all_readings.append(None)
            continue

        # ── 4. Parse score ────────────────────────────────────
        score = parse_score(blocks)
        all_readings.append(score)

        pa = score.point_index("a")
        pb = score.point_index("b")
        ga = str(score.games.get("a", "")) if score.games.get("a") is not None else ""
        gb = str(score.games.get("b", "")) if score.games.get("b") is not None else ""

        print(f"→ pts=({score.points.get('a', '?')}/{score.points.get('b', '?')})  "
              f"games=({ga}-{gb})  "
              f"players=({score.player_a or '?'} vs {score.player_b or '?'})")

        # ── 5. Inject into XML (ONLY if valid detection) ───────
        pt_el = ep["point_el"]

        if score.points.get("a") is not None or score.points.get("b") is not None:
            if pa:
                pt_el.set("score_A", pa)
            if pb:
                pt_el.set("score_B", pb)
            if score.server:
                pt_el.set("service", score.server)

            # Game-level score
            gm_el = ep["game_el"]
            if ga:
                gm_el.set("score_A", ga)
            if gb:
                gm_el.set("score_B", gb)

            # Set-level score
            st_el = ep["set_el"]
            set_idx = set_id - 1
            if set_idx < len(score.sets):
                st_el.set("score_A", str(score.sets[set_idx]["a"]))
                st_el.set("score_B", str(score.sets[set_idx]["b"]))
        else:
            # IMPORTANT: do nothing → keep previous XML values
            print("→ Skipping update (no valid scoreboard detected)")

        # Player names on first successful read
        if score.player_a and not root.get("player_a"):
            root.set("player_a", score.player_a)
            root.set("player_b", score.player_b or "")
            root.set("nat_a", score.nat_a or "")
            root.set("nat_b", score.nat_b or "")

        success += 1

    video.close()

    # ── Regroup points into real games/sets using OCR'd point score ──
    regroup_into_games(root, endpoints, all_readings)

    # ── Sanity check: flag any reconstructed game with a suspicious
    #    point count (too short/long usually means a missed boundary) ──
    for set_el in root.findall("set"):
        for game_el in set_el.findall("game"):
            n = len(game_el.findall("point"))
            if n < 4 or n > 12:
                print(f"⚠️  Set {set_el.get('id')} Game {game_el.get('id')} "
                      f"has {n} points — worth a manual check")

    # ── Save enriched XML ─────────────────────────────────────
    with open(OUTPUT_XML, "w", encoding="utf-8") as f:
        f.write(pretty_xml(root))

    print(f"\n{'─'*60}")
    print(f"Done.  Success: {success}  Failed: {failed}  Total: {total}")
    print(f"Enriched XML saved to: {OUTPUT_XML}")
    print(f"\nPoint score mapping used:")
    print(f"  0=love  1=15  2=30  3=40  4=AD")


if __name__ == "__main__":
    run()