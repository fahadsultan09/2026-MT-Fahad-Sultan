import xml.etree.ElementTree as ET
import re
import os

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "../data/raw/Jabeur_All_hits_fixed_v3.xml"
OUTPUT_FILE = "../data/processed/fahad_sultan_output2.xml"
GAP_SECONDS = 4

# -----------------------------
# FUNCTION 1: CLEAN XML
# -----------------------------
def clean_xml_content(content: str) -> str:
    content = re.sub(r'\b(start|end)\s+of\s+point\s+\d+\b', '', content, flags=re.IGNORECASE)
    content = re.sub(r'^\s*\d{2}:\d{2}:\d{2}\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n\s*\n', '\n', content)
    return content.strip()

# -----------------------------
# HELPER: TIME → SECONDS
# -----------------------------
def time_to_seconds(t: str) -> int:
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

# -----------------------------
# FUNCTION 2: PROCESS XML
# -----------------------------
def process_xml(xml_string: str, clean: bool = True):
    if clean:
        xml_string = clean_xml_content(xml_string)

    root = ET.fromstring(xml_string)
    game = root.find(".//game")
    if game is None:
        raise ValueError("No <game> element found in XML")

    # Build a flat list of elements in order
    flat = list(game.iter())

    # Map each hit → its following player siblings
    hit_to_players = {}
    hits = []

    for i, elem in enumerate(flat):
        if elem.tag == "hit":
            hits.append(elem)
            players = []

            # Look ahead until next hit
            for next_elem in flat[i+1:]:
                if next_elem.tag == "hit":
                    break
                if next_elem.tag == "player":
                    players.append(next_elem)

            hit_to_players[elem] = players

    # Save original point attributes
    original_points = game.findall("point")
    points_attrs = [p.attrib for p in original_points]

    # Group hits by time gap
    new_points = []
    current_point = []
    prev_time = None

    for h in hits:
        t = time_to_seconds(h.attrib["time"])
        if prev_time is None:
            current_point.append(h)
        else:
            if t - prev_time > GAP_SECONDS:
                new_points.append(current_point)
                current_point = []
            current_point.append(h)
        prev_time = t

    if current_point:
        new_points.append(current_point)

    # Remove old points
    for p in game.findall("point"):
        game.remove(p)

    # Create new points
    for i, point_hits in enumerate(new_points, start=1):
        attrs = points_attrs[i-1].copy() if i-1 < len(points_attrs) else {}
        attrs["id"] = str(i)
        point_elem = ET.SubElement(game, "point", attrs)

        hit_counter = 1  # reset for each point

        for h in point_hits:
            hit_attribs = h.attrib.copy()
            hit_attribs["id"] = str(hit_counter)

            hit_elem = ET.SubElement(point_elem, "hit", hit_attribs)
            hit_elem.text = h.text

            hit_counter += 1

    return root

# -----------------------------
# FUNCTION 3: INDENT XML
# -----------------------------
def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

# -----------------------------
# MAIN
# -----------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    print(f"Reading XML from: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        xml_data = f.read()

    print("Processing XML...")
    root = process_xml(xml_data, clean=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

    print(f"Saving XML to: {OUTPUT_FILE}")
    indent(root)
    tree = ET.ElementTree(root)
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

    print("Done!")

if __name__ == "__main__":
    main()
