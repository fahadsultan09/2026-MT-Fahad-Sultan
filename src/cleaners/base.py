import copy

def get_y(hit) -> float:
    return float(hit.attrib["y"])


def set_y(hit, value: float):
    hit.attrib["y"] = f"{value:.3f}"


def is_between_opposite_sides(hits, index):

    if index == 0 or index == len(hits) - 1:
        return False

    prev_y = get_y(hits[index - 1])
    curr_y = get_y(hits[index])
    next_y = get_y(hits[index + 1])

    return (
        prev_y > 0 and curr_y < 0 and next_y > 0
    ) or (
        prev_y < 0 and curr_y > 0 and next_y < 0
    )


def all_same_side(hits):

    if not hits:
        return True

    return (
        all(get_y(h) > 0 for h in hits)
        or
        all(get_y(h) < 0 for h in hits)
    )

def mostly_same_side(hits, threshold=0.70):
    if not hits:
        return True

    positive = 0
    negative = 0

    for h in hits:
        y = get_y(h)

        if y > 0:
            positive += 1
        elif y < 0:
            negative += 1

    total = len(hits)

    return positive / total >= threshold or negative / total >= threshold

def get_sign(hit):
    y = get_y(hit)

    if y > 0:
        return 1

    if y < 0:
        return -1

    return 0


def remove_clear_middle_outliers(hits, court_limit=12.5):
    cleaned = []

    for i, h in enumerate(hits):
        y = get_y(h)

        is_last = i == len(hits) - 1

        # remove impossible middle hit
        # but keep last hit because it can represent rally-ending out ball
        if abs(y) >= court_limit and not is_last:
            continue

        cleaned.append(copy.deepcopy(h))

    return cleaned


def normalize_boundary_y(hits, court_limit=12.5):
    cleaned = []

    for i, h in enumerate(hits):
        h = copy.deepcopy(h)
        y = get_y(h)

        is_last = i == len(hits) - 1

        # 12.xxx -> 11.xxx
        # -12.xxx -> -11.xxx
        if 12.0 <= abs(y) < court_limit and not is_last:
            decimal_part = abs(y) - 12
            new_y = 11 + decimal_part

            if y < 0:
                new_y = -new_y

            set_y(h, new_y)

        cleaned.append(h)

    return cleaned


def keep_second_last_out_ball_remove_last(hits, playable_limit=11.885):
    if len(hits) < 2:
        return hits

    second_last_index = len(hits) - 2
    second_last_y = get_y(hits[second_last_index])

    if abs(second_last_y) > playable_limit:
        return hits[:-1]

    return hits

def all_same_side(hits):
    if not hits:
        return False

    all_positive = all(get_y(h) > 0 for h in hits)
    all_negative = all(get_y(h) < 0 for h in hits)

    return all_positive or all_negative