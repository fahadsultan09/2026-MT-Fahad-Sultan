import xml.etree.ElementTree as ET
import re
import copy


def clean_xml_content(content: str) -> str:

    content = re.sub(
        r'\b(start|end)\s+of\s+point\s+\d+\b',
        '',
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r'^\s*\d{2}:\d{2}:\d{2}\s*$',
        '',
        content,
        flags=re.MULTILINE
    )

    content = re.sub(r'\n\s*\n', '\n', content)

    return content.strip()


def time_to_seconds(t: str) -> int:

    h, m, s = map(int, t.split(":"))

    return h * 3600 + m * 60 + s


def read_xml(path):

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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


def write_xml(root, output_path):

    indent(root)

    tree = ET.ElementTree(root)

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True
    )


def rebuild_points(game, gap_seconds):

    hits = []

    for hit in game.iter("hit"):
        hits.append(copy.deepcopy(hit))

    new_points = []

    current_point = []
    prev_time = None

    for h in hits:

        t = time_to_seconds(h.attrib["time"])

        if prev_time is not None and t - prev_time > gap_seconds:

            new_points.append(current_point)

            current_point = []

        current_point.append(h)

        prev_time = t

    if current_point:
        new_points.append(current_point)

    return new_points