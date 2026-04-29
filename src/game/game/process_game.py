import xml.etree.ElementTree as ET

def parse_score(value: str) -> int:
    if value == "AD":
        return 4
    try:
        return int(value)
    except ValueError:
        return 0

def tennis_state(a, b):
    if a >= 3 and b >= 3:
        if a == b:
            return "Deuce"
        elif a == b + 1:
            return "Adv A"
        elif b == a + 1:
            return "Adv B"
    mapping = ["0", "15", "30", "40"]
    return f"{mapping[min(a,3)]}-{mapping[min(b,3)]}"

def is_game_over(a, b):
    if a >= 4 and a - b >= 2:
        return "A"
    if b >= 4 and b - a >= 2:
        return "B"
    return None

def full_game_breakdown(path):
    tree = ET.parse(path)
    root = tree.getroot()

    games = []
    current_game = {
        "game_number": 1,
        "winner": None,
        "rallies": []
    }

    prevA = prevB = None
    game_over = False

    for point in root.findall(".//point"):
        pid = int(point.get("id"))
        a = parse_score(point.get("score_A"))
        b = parse_score(point.get("score_B"))

        score_label = tennis_state(a, b)

        # detect winner ONLY if not already closed
        if not game_over:
            winner = is_game_over(a, b)
            if winner:
                current_game["winner"] = winner
                game_over = True

        # detect new game start
        if prevA is not None and prevB is not None:
            if a == 0 and b == 0 and (prevA != 0 or prevB != 0):

                # infer winner from previous score if not already set
                if current_game["winner"] is None:
                    if prevA > prevB:
                        current_game["winner"] = "A"
                    else:
                        current_game["winner"] = "B"

                games.append(current_game)

                current_game = {
                    "game_number": current_game["game_number"] + 1,
                    "winner": None,
                    "rallies": []
                }

                game_over = False

        current_game["rallies"].append({
            "rally_id": pid,
            "score": score_label
        })

        prevA, prevB = a, b

    games.append(current_game)
    return games

# -----------------------------
#  PRINT BREAKDOWN
# -----------------------------
def print_breakdown(path):
    games = full_game_breakdown(path)

    for g in games:
        print(f"\nGAME {g['game_number']} — Winner: {g['winner']}")
        for r in g["rallies"]:
            print(f"   Rally {r['rally_id']}: {r['score']}")


# -----------------------------
#  RUN
# -----------------------------


if __name__ == "__main__":
    print_breakdown("../../../data/processed/fahad_sultan_updated2.xml")
