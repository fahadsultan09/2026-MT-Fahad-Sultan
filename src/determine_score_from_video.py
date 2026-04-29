import cv2
import csv
from lxml import etree
import os

XML_FILE = "../data/processed/fahad_sultan_output.xml"
VIDEO_FILE = "jabeur.mp4"
OUTPUT_DIR = "scoreboard_log"
LOG_FILE = "capture_times.csv"

# Using a 10-second offset as you suggested
OFFSET = 10.0 

def time_to_seconds(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def get_rally_ends(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    points = []
    for point in root.findall(".//point"):
        hits = point.findall("./hit")
        if not hits: continue
        t_end = time_to_seconds(hits[-1].attrib["time"])
        points.append({
            "id": point.attrib.get("id", "unknown"),
            "rally_end_time": t_end
        })
    return points

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    points = get_rally_ends(XML_FILE)
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Prepare to log the times
    log_data = []

    for p in points:
        capture_time = p["rally_end_time"] + OFFSET
        frame_idx = int(capture_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # ROI: 71, 594, 385, 650
            scoreboard_crop = frame[594:650, 71:385]
            
            filename = f"point_{p['id']}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), scoreboard_crop)
            
            # Store info for the CSV
            log_data.append({
                "Point_ID": p['id'],
                "Rally_End": p['rally_end_time'],
                "Capture_Time_Seconds": capture_time,
                "Capture_Time_HHMMSS": f"{int(capture_time//3600):02}:{int((capture_time%3600)//60):02}:{capture_time%60:05.2f}",
                "Image_Path": filename
            })
        else:
            print(f"Skipped Point {p['id']}: Frame not found at {capture_time}s")

    cap.release()

    # Write the log file
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Point_ID", "Rally_End", "Capture_Time_Seconds", "Capture_Time_HHMMSS", "Image_Path"])
        writer.writeheader()
        writer.writerows(log_data)

    print(f"Done! Images are in '{OUTPUT_DIR}' and time log is in '{LOG_FILE}'.")

if __name__ == "__main__":
    main()