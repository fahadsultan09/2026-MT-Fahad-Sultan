import copy

from cleaners.base import (
    get_y,
    mostly_same_side
)


def short_rally_with_outlier(point_hits, court_limit=12.5):
    if len(point_hits) > 3:
        return False

    return any(abs(get_y(h)) >= court_limit for h in point_hits)


def remove_single_outliers(point_hits, court_limit=12.5):
    cleaned = []

    for i, h in enumerate(point_hits):
        y = get_y(h)
        is_last = i == len(point_hits) - 1

        if abs(y) >= court_limit and not is_last:
            continue

        cleaned.append(copy.deepcopy(h))

    return cleaned


def trim_after_late_out_of_play(point_hits, playable_limit=11.885):
    n = len(point_hits)

    if n < 3:
        return point_hits

    for idx in [n - 3, n - 2]:
        y = get_y(point_hits[idx])

        if abs(y) > playable_limit:
            current_sign = 1 if y > 0 else -1
            next_hits = point_hits[idx + 1:]

            if all((get_y(h) > 0) == (current_sign > 0) for h in next_hits):
                return point_hits[:idx + 1]

    return point_hits

def valid_two_hit_finish(point_hits, playable_limit=11.885):
    if len(point_hits) != 2:
        return False

    first_y = abs(get_y(point_hits[0]))
    second_y = abs(get_y(point_hits[1]))

    return (
        first_y <= playable_limit
        and second_y > playable_limit
    )

def clean_point_21_40(point_hits, court_limit=12.5, playable_limit=11.885):

    # remove single-hit points
    if len(point_hits) < 2:
        return None
    
    if valid_two_hit_finish(point_hits,playable_limit):
        return point_hits

    # remove short rallies with clear outlier
    # if short_rally_with_outlier(point_hits, court_limit):
    #     return None

    # remove long rallies mostly one side
    if len(point_hits) >= 10 and mostly_same_side(point_hits, threshold=0.75):
        return None

    cleaned = remove_single_outliers(point_hits, court_limit)

    cleaned = trim_after_late_out_of_play(cleaned, playable_limit)

    return cleaned