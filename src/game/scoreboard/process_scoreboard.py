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
    
    # Extract Player Info
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
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(Image.fromarray(img), config=config).strip()
    return score_map.get(text, text if text else "0")

def main():
    # Load all configurations from the external XML
    paths, player_data, ocr_cfg, rois, score_map = load_full_config("../../config.xml")
    
    # Ensure the output directory for images exists
    os.makedirs(paths["save_dir"], exist_ok=True)
    
    # Parse the input XML
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(paths["input"], parser)
    root = tree.getroot()
    set_elem = root.find(".//set")

    # 1. Insert Player Metadata
    for p in reversed(player_data):
        p_elem = etree.Element("player", id=p["id"], name=p["name"])
        root.insert(0, p_elem)

    # Initialize video capture
    cap = cv2.VideoCapture(paths["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Initialize tracking variables
    current_score_a, current_score_b = "0", "0"
    game_counter = 1
    point_counter = 1 
    
    # 2. Extract points and clear existing game structure
    all_points = root.findall(".//point")
    for p in all_points:
        p.getparent().remove(p)
    for g in set_elem.findall("game"):
        set_elem.remove(g)

    # Create the first game container
    current_game_elem = etree.SubElement(set_elem, "game", 
                                         id=str(game_counter),
                                         service="A", 
                                         score_A="0", 
                                         score_B="0")

    print("Processing points and generating unique filenames...")

    for point in all_points:
        # 3. Assign IDs and Scores
        point.set("id", str(point_counter))
        point.set("score_A", current_score_a)
        point.set("score_B", current_score_b)
        current_game_elem.append(point)

        hits = point.findall("./hit")
        if not hits: 
            point_counter += 1
            continue

        # 4. OCR Extraction
        t_end = time_to_seconds(hits[-1].attrib["time"])
        capture_time = t_end + ocr_cfg["offset"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(capture_time * fps))
        ret, frame = cap.read()

        if ret:
            sy1, sy2, sx1, sx2 = rois["scoreboard"]
            scoreboard = frame[sy1:sy2, sx1:sx2]
            
            # UNIQUE FILENAME: point_{game_id}_{point_id}.png
            # This allows you to instantly verify Point 5 of Game 2 by looking for point_2_5.png
            img_name = f"point_{game_counter}_{point_counter}.png"
            img_name = f"game_{game_counter}_point_{point_counter}.png"
            print('image name' ,img_name)
            cv2.imwrite(os.path.join(paths["save_dir"], img_name), scoreboard)

            gray = cv2.cvtColor(scoreboard, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            new_a = extract_ocr_score(thresh[rois["score_a"][0]:rois["score_a"][1], rois["score_a"][2]:rois["score_a"][3]], 
                                     score_map, ocr_cfg["psm"], ocr_cfg["whitelist"])
            new_b = extract_ocr_score(thresh[rois["score_b"][0]:rois["score_b"][1], rois["score_b"][2]:rois["score_b"][3]], 
                                     score_map, ocr_cfg["psm"], ocr_cfg["whitelist"])

            # 5. Game Transition Logic
            was_active = current_score_a != "0" or current_score_b != "0"
            is_reset = new_a == "0" and new_b == "0"

            if was_active and is_reset:
                game_counter += 1
                point_counter = 1  # Reset point ID for the new game
                current_game_elem = etree.SubElement(set_elem, "game", 
                                                     id=str(game_counter),
                                                     service="A", 
                                                     score_A="0", 
                                                     score_B="0")
            else:
                point_counter += 1
            
            current_score_a, current_score_b = new_a, new_b
        else:
            point_counter += 1

    # Final Save
    tree.write(paths["output"], pretty_print=True, xml_declaration=True, encoding="utf-8")
    cap.release()
    print(f"Success! Images saved in {paths['save_dir']} and XML updated.")

if __name__ == "__main__":
    main()