import cv2
import pytesseract
from PIL import Image
from lxml import etree
import os

DEBUG = True
DEBUG_DIR = "debug_frames"
os.makedirs(DEBUG_DIR, exist_ok=True)

# -----------------------------
# LOAD CONFIG
# -----------------------------
def load_config(config_file):
    tree = etree.parse(config_file)

    paths = {
        "input": tree.findtext(".//input_file"),
        "output": "../../../data/processed/output.xml",
        "video": tree.findtext(".//video_file"),
        "save_dir": tree.findtext(".//save_dir")
    }

    ocr = {
        "psm": tree.findtext(".//psm"),
        "whitelist": tree.findtext(".//whitelist")
    }

    score_map = {
        s.get("ocr"): s.get("value")
        for s in tree.xpath("//score")
    }

    return paths, ocr, score_map

# -----------------------------
# TIME
# -----------------------------
def time_to_seconds(t):
    h, m, s = t.split(":")
    return int(h)*3600 + int(m)*60 + float(s)

# -----------------------------
# SCORE CLEANING
# -----------------------------
VALID = {"0", "15", "30", "40", "AD"}

def clean_score(text, score_map):
    text = text.strip().upper()

    if text in score_map:
        return score_map[text]

    fixes = {
        "4O": "40",
        "1S": "15",
        "O": "0",
        "": "0"
    }

    text = fixes.get(text, text)

    return score_map.get(text, "0")

# -----------------------------
# DETECT SCOREBOARD
# -----------------------------
def detect_scoreboard(frame):
    h, w, _ = frame.shape
    search = frame[0:int(h * 0.35), :]

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / float(ch)

        if 2 < ar < 12 and cw > 120 and ch > 30:
            candidates.append((x, y, cw, ch))

    if not candidates:
        return None

    x, y, cw, ch = max(candidates, key=lambda b: b[2]*b[3])
    return search[y:y+ch, x:x+cw]

# -----------------------------
# SPLIT SCOREBOARD
# -----------------------------
def split_scoreboard(img):
    h, w = img.shape[:2]
    return img[:, :w//2], img[:, w//2:]

# -----------------------------
# OCR
# -----------------------------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    return th

def ocr_score(img, psm, whitelist, score_map):
    if img is None or img.size == 0:
        return "0"

    img = preprocess(img)
    img = cv2.resize(img, None, fx=2, fy=2)

    config = f'--psm {psm} -c tessedit_char_whitelist={whitelist}'
    text = pytesseract.image_to_string(Image.fromarray(img), config=config)

    return clean_score(text, score_map)

# -----------------------------
# MAIN
# -----------------------------
def main(config_file):
    paths, ocr_cfg, score_map = load_config(config_file)

    os.makedirs(paths["save_dir"], exist_ok=True)

    tree = etree.parse(paths["input"])
    root = tree.getroot()

    cap = cv2.VideoCapture(paths["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    points = root.findall(".//point")

    print(f"Processing {len(points)} points...")

    for i, point in enumerate(points, start=1):

        hits = point.findall("./hit")
        if not hits:
            continue

        t = time_to_seconds(hits[0].attrib["time"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))

        ret, frame = cap.read()
        if not ret:
            continue

        scoreboard = detect_scoreboard(frame)


        if scoreboard is not None:
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, int(h*0.25)), (0, 255, 0), 2)
        if DEBUG:
            cv2.imwrite(f"{DEBUG_DIR}/frame_{i}.png", frame)

        if scoreboard is None:
            print(f"[WARN] No scoreboard at point {i}")
            point.set("score_A", "0")
            point.set("score_B", "0")
            continue

        # Save debug image
        cv2.imwrite(os.path.join(paths["save_dir"], f"score_{i}.png"), scoreboard)

        left, right = split_scoreboard(scoreboard)

        score_a = ocr_score(left, ocr_cfg["psm"], ocr_cfg["whitelist"], score_map)
        score_b = ocr_score(right, ocr_cfg["psm"], ocr_cfg["whitelist"], score_map)

        point.set("score_A", score_a)
        point.set("score_B", score_b)

        print(f"Point {i}: {score_a} - {score_b}")

    cap.release()

    tree.write(paths["output"], pretty_print=True, xml_declaration=True, encoding="utf-8")

    print(f"\n✅ Output saved to {paths['output']}")

# -----------------------------
if __name__ == "__main__":
    main("../../config.xml")