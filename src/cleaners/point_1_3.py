import copy

from cleaners.base import (
    get_y,
    set_y,
    is_between_opposite_sides,
    mostly_same_side
)


def clean_point_1_3(point_hits, court_limit):

    if mostly_same_side(point_hits, threshold=0.70):
        return None

    cleaned = []

    for i, h in enumerate(point_hits):

        h = copy.deepcopy(h)

        y = get_y(h)

        is_last = i == len(point_hits) - 1

        # remove outlier
        if abs(y) >= court_limit and not is_last:

            if is_between_opposite_sides(point_hits, i):
                cleaned.append(h)

            continue

        # 12.xxx -> 11.xxx
        if 12.0 <= abs(y) < court_limit and not is_last:

            decimal_part = abs(y) - 12

            new_y = 11 + decimal_part

            if y < 0:
                new_y = -new_y

            set_y(h, new_y)

        cleaned.append(h)

    # + - - + average
    # - + + - average
    i = 0

    while i < len(cleaned) - 3:

        y1 = get_y(cleaned[i])
        y2 = get_y(cleaned[i + 1])
        y3 = get_y(cleaned[i + 2])
        y4 = get_y(cleaned[i + 3])

        if y1 > 0 and y2 < 0 and y3 < 0 and y4 > 0:

            avg = (y2 + y3) / 2

            set_y(cleaned[i + 1], avg)

            del cleaned[i + 2]

            continue

        if y1 < 0 and y2 > 0 and y3 > 0 and y4 < 0:

            avg = (y2 + y3) / 2

            set_y(cleaned[i + 1], avg)

            del cleaned[i + 2]

            continue

        i += 1

    return cleaned