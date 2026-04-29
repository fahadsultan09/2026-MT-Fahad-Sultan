import cv2
import pytesseract
from PIL import Image
from lxml import etree
import os

# --- Configuration Loader ---
def load_full_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    
    tree = etree.parse(config_file)
    
    paths = {
        "input": tree.findtext(".//input_file"),
        "output": tree.findtext(".//output_file"),
        "video": tree.findtext(".//video_file"),
        "save_dir": tree.findtext(".//save_dir")
    }
    
    players = []
    for p_elem in tree.xpath("//players/player"):
        players.append({
            "id": p_elem.get("id"),
            "name": p_elem.get("name")
        })
    
    ocr_configurations = {
        "offset": float(tree.findtext(".//offset")),
        "psm": tree.findtext(".//psm"),
        "whitelist": tree.findtext(".//whitelist")
    }
    
    roi_elem = tree.find(".//roi")
    rois = {
        "scoreboard": [int(roi_elem.find("scoreboard").get(k)) for k in ["y_start", "y_end", "x_start", "x_end"]],
        "score_a": [int(roi_elem.find("score_a").get(k)) for k in ["y_start", "y_end", "x_start", "x_end"]],
        "score_b": [int(roi_elem.find("score_b").get(k)) for k in ["y_start", "y_end", "x_start", "x_end"]]
    }
    
    score_map = {item.get("ocr"): item.get("value") for item in tree.xpath("//score")}
    
    return paths, players, ocr_configurations, rois, score_map

def time_to_seconds(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def extract_ocr_score(img, score_map, psm, whitelist):
    if img is None or img.size == 0:
        return "0"
    config = f'--psm {psm} -c tessedit_char_whitelist={whitelist}'
    cv2.imwrite(f"debug_thresh_pt_{whitelist}.png", img)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(Image.fromarray(img), config=config).strip()
    return score_map.get(text, text if text else "0")

def main():
    paths, player_data, ocr_cfg, rois, score_map = load_full_config("../../config.xml")
    os.makedirs(paths["save_dir"], exist_ok=True)
    
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(paths["input"], parser)
    root = tree.getroot()
    set_elem = root.find(".//set")

    # Insert Player Metadata
    for p in reversed(player_data):
        p_elem = etree.Element("player", id=p["id"], name=p["name"])
        root.insert(0, p_elem)

    cap = cv2.VideoCapture(paths["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # For this version, we keep all points in one flat list under the set or first game
    all_points = root.findall(".//point")
    
    # Optional: Clear existing games if you want a clean flat point list
    for g in set_elem.findall("game"):
        set_elem.remove(g)
    
    # Create a single container for all points
    container = etree.SubElement(set_elem, "game", id="1")

    print("Processing points: Updating score based on first-hit timestamp...")

    for i, point in enumerate(all_points, start=1):
        hits = point.findall("./hit")
        if not hits:
            # If no hits exist, we can't determine time; skip OCR
            container.append(point)
            continue

        # Strategy: Get scoreboard at the exact time of the FIRST hit
        t_start = time_to_seconds(hits[0].attrib["time"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * fps))
        ret, frame = cap.read()

        ocr_a, ocr_b = "0", "0"
        if ret:
            sy1, sy2, sx1, sx2 = rois["scoreboard"]
            scoreboard = frame[sy1:sy2, sx1:sx2]
            
            # Save image using a flat index for "fast reach"
            img_name = f"point_{i}_start.png"
            cv2.imwrite(os.path.join(paths["save_dir"], img_name), scoreboard)

            gray = cv2.cvtColor(scoreboard, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            ocr_a = extract_ocr_score(thresh[rois["score_a"][0]:rois["score_a"][1], rois["score_a"][2]:rois["score_a"][3]], 
                                     score_map, ocr_cfg["psm"], ocr_cfg["whitelist"])
            ocr_b = extract_ocr_score(thresh[rois["score_b"][0]:rois["score_b"][1], rois["score_b"][2]:rois["score_b"][3]], 
                                     score_map, ocr_cfg["psm"], ocr_cfg["whitelist"])

        # Update point attributes and append to container
        point.set("id", str(i))
        point.set("score_A", ocr_a)
        point.set("score_B", ocr_b)
        container.append(point)
        
        print(f"Point {i}: Time {hits[0].attrib['time']} -> Score A: {ocr_a}, Score B: {ocr_b}")

    # Final Save
    tree.write(paths["output"], pretty_print=True, xml_declaration=True, encoding="utf-8")
    cap.release()
    print(f"Success! Updated points saved to {paths['output']}")

if __name__ == "__main__":
    main()