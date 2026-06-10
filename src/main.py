import os
import xml.etree.ElementTree as ET

from config_loader import load_config

from xml_utils import (
    clean_xml_content,
    read_xml,
    write_xml,
    rebuild_points
)

from cleaners.common import clean_common
from cleaners.point_1_3 import clean_point_1_3
from cleaners.point_4_10 import clean_point_4_10
from cleaners.point_11_20 import clean_point_11_20
from cleaners.point_21_40 import clean_point_21_40

configuration = load_config()

INPUT_FILE = configuration["INPUT_FILE"]
OUTPUT_FILE = configuration["OUTPUT_FILE"]
GAP_SECONDS = configuration["GAP_SECONDS"]
COURT_LIMIT = configuration["COURT_LIMIT"]
PLAYABLE_LIMIT = configuration["PLAYABLE_LIMIT"]

def apply_cleaning(point_index, point_hits):

    # first apply common rules
    point_hits = clean_common(
        point_hits,
        COURT_LIMIT,
        PLAYABLE_LIMIT
    )

    if point_hits is None:
        return None

    # then apply range-specific edge cases
    if 1 <= point_index <= 3:
        return clean_point_1_3(point_hits,COURT_LIMIT)

    elif 4 <= point_index <= 10:
        return clean_point_4_10(point_hits)

    elif 11 <= point_index <= 20:
        return clean_point_11_20(point_hits)
    
    elif 21 <= point_index <= 40:
        return clean_point_21_40(point_hits,COURT_LIMIT,PLAYABLE_LIMIT)

    return point_hits

def process_xml(xml_string: str):

    xml_string = clean_xml_content(xml_string)

    root = ET.fromstring(xml_string)

    game = root.find(".//game")

    if game is None:
        raise ValueError("No <game> element found")

    original_points = game.findall("point")

    points_attrs = [p.attrib.copy() for p in original_points]

    rebuilt_points = rebuild_points(
        game,
        GAP_SECONDS
    )

    # remove original points
    for p in game.findall("point"):
        game.remove(p)

    new_point_id = 1

    for i, point_hits in enumerate(rebuilt_points, start=1):

        cleaned_hits = apply_cleaning(
            i,
            point_hits
        )

        if cleaned_hits is None:
            print(f"Skipping point {i}")
            continue

        if len(cleaned_hits) < 2:
            print(f"Skipping point {i}: less than 2 hits")
            continue

        attrs = (
            points_attrs[i - 1].copy()
            if i - 1 < len(points_attrs)
            else {}
        )

        attrs["id"] = str(new_point_id)

        point_elem = ET.SubElement(
            game,
            "point",
            attrs
        )

        for hit_id, h in enumerate(cleaned_hits, start=1):

            hit_attrs = h.attrib.copy()

            hit_attrs["id"] = str(hit_id)

            ET.SubElement(
                point_elem,
                "hit",
                hit_attrs
            )

        new_point_id += 1

    return root


def main():

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    print(f"Reading XML from: {INPUT_FILE}")

    xml_data = read_xml(INPUT_FILE)

    print("Processing XML...")

    root = process_xml(xml_data)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    print(f"Saving XML to: {OUTPUT_FILE}")

    write_xml(root, OUTPUT_FILE)

    print("Done!")


if __name__ == "__main__":
    main()