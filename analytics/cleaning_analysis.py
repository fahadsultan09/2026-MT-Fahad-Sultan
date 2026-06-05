import re
import csv
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


ORIGINAL_XML = "../data/raw/Jabeur_All_hits_fixed_v3.xml"
CLEANED_XML = "../data/processed/Jabeur_scored2.xml"

OUTPUT_DIR = "../visualizations"
COURT_LIMIT = 12.5
PLAYABLE_LIMIT = 11.885


def clean_xml_text(text):
    text = re.sub(r"\b(start|end)\s+of\s+point\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d{2}:\d{2}:\d{2}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_xml(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    text = clean_xml_text(text)
    return ET.fromstring(text)


def hit_key(hit):
    return (
        hit.attrib.get("time", ""),
        hit.attrib.get("x", "")
    )


def classify_removed_hit(hit):
    y = float(hit.attrib["y"])

    if abs(y) >= COURT_LIMIT:
        return "Hard outlier"

    if abs(y) > PLAYABLE_LIMIT:
        return "Outside playable area"

    return "Other removed hit"


def collect_hits_by_game(root):
    data = {}

    for set_elem in root.findall(".//set"):
        set_id = set_elem.attrib.get("id", "unknown")

        for game in set_elem.findall("game"):
            game_id = game.attrib.get("id", "unknown")
            group_key = f"Set {set_id} / Game {game_id}"

            hits = []
            for hit in game.findall(".//hit"):
                hits.append(hit)

            data[group_key] = hits

    return data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    original_root = parse_xml(ORIGINAL_XML)
    cleaned_root = parse_xml(CLEANED_XML)

    original_games = collect_hits_by_game(original_root)
    cleaned_games = collect_hits_by_game(cleaned_root)

    rows = []

    for game_key, original_hits in original_games.items():
        cleaned_hits = cleaned_games.get(game_key, [])

        cleaned_keys = Counter(hit_key(h) for h in cleaned_hits)

        removed_categories = Counter()
        removed_count = 0

        for hit in original_hits:
            key = hit_key(hit)

            if cleaned_keys[key] > 0:
                cleaned_keys[key] -= 1
            else:
                removed_count += 1
                category = classify_removed_hit(hit)
                removed_categories[category] += 1

        before = len(original_hits)
        after = len(cleaned_hits)
        removed_percentage = (removed_count / before * 100) if before else 0

        rows.append({
            "game": game_key,
            "hits_before": before,
            "hits_after": after,
            "hits_removed": removed_count,
            "removed_percentage": removed_percentage,
            "hard_outlier": removed_categories["Hard outlier"],
            "outside_playable": removed_categories["Outside playable area"],
            "other_removed": removed_categories["Other removed hit"],
        })

    csv_path = os.path.join(OUTPUT_DIR, "cleaning_summary_by_game.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "game",
            "hits_before",
            "hits_after",
            "hits_removed",
            "removed_percentage",
            "hard_outlier",
            "outside_playable",
            "other_removed",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    games = [r["game"] for r in rows]
    hard_outlier = [r["hard_outlier"] for r in rows]
    outside_playable = [r["outside_playable"] for r in rows]
    other_removed = [r["other_removed"] for r in rows]

    # Chart 1: removed hit types by game
    plt.figure(figsize=(14, 7))

    plt.bar(games, hard_outlier, label="Hard outlier")
    plt.bar(
        games,
        outside_playable,
        bottom=hard_outlier,
        label="Outside playable area"
    )

    bottom_values = [
        hard_outlier[i] + outside_playable[i]
        for i in range(len(games))
    ]

    plt.bar(
        games,
        other_removed,
        bottom=bottom_values,
        label="Other removed hit"
    )

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Game")
    plt.ylabel("Removed hits")
    plt.title("Removed Hits by Error Type and Game")
    plt.legend()
    plt.tight_layout()

    bar_chart_path = os.path.join(OUTPUT_DIR, "removed_hits_by_game.png")
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()

    # Chart 2: percentage removed by game
    plt.figure(figsize=(14, 6))

    plt.bar(
        games,
        [r["removed_percentage"] for r in rows]
    )

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Game")
    plt.ylabel("Removed hits (%)")
    plt.title("Percentage of Hits Removed per Game")
    plt.tight_layout()

    percentage_chart_path = os.path.join(OUTPUT_DIR, "removed_percentage_by_game.png")
    plt.savefig(percentage_chart_path, dpi=300)
    plt.close()

    # Chart 3: overall retained vs removed
    total_before = sum(r["hits_before"] for r in rows)
    total_removed = sum(r["hits_removed"] for r in rows)
    total_retained = total_before - total_removed

    plt.figure(figsize=(7, 7))

    plt.pie(
        [total_retained, total_removed],
        labels=["Retained hits", "Removed hits"],
        autopct="%1.1f%%"
    )

    plt.title("Overall Hit Retention After Cleaning")
    plt.tight_layout()

    pie_chart_path = os.path.join(OUTPUT_DIR, "overall_retained_vs_removed.png")
    plt.savefig(pie_chart_path, dpi=300)
    plt.close()

    print("Saved:")
    print(csv_path)
    print(bar_chart_path)
    print(percentage_chart_path)
    print(pie_chart_path)


if __name__ == "__main__":
    main()