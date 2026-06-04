import copy

from cleaners.base import get_y

def keep_second_last_out_ball_remove_last(point_hits, playable_limit=11.885):

    if len(point_hits) < 2:
        return point_hits

    second_last_index = len(point_hits) - 2

    second_last_y = get_y(point_hits[second_last_index])

    # if second-last is outside playable area,
    # keep it and remove only the final hit
    if abs(second_last_y) > playable_limit:
        return point_hits[:-1]

    return point_hits

def all_same_side(point_hits):
    if not point_hits:
        return True

    return (
        all(get_y(h) > 0 for h in point_hits)
        or
        all(get_y(h) < 0 for h in point_hits)
    )

def invalid_after_first_valid_hit(point_hits, court_limit=12.5, playable_limit=11.885):
    if len(point_hits) < 4:
        return False

    first_y = get_y(point_hits[0])

    if abs(first_y) > playable_limit:
        return False

    remaining = point_hits[1:]

    outlier_count = sum(abs(get_y(h)) >= court_limit for h in remaining)

    after_outliers = [
        h for h in remaining
        if abs(get_y(h)) < court_limit
    ]

    if not after_outliers:
        return outlier_count >= 2

    negative_count = sum(get_y(h) < 0 for h in after_outliers)
    positive_count = sum(get_y(h) > 0 for h in after_outliers)

    same_side_after_outliers = (
        negative_count == len(after_outliers)
        or positive_count == len(after_outliers)
    )

    return outlier_count >= 2 and same_side_after_outliers


def clean_point_11_20(point_hits, court_limit=12.5, playable_limit=11.885):
    """
    Rules for points 11-20.
    """

    if all_same_side(point_hits):
        return None

    if invalid_after_first_valid_hit(
        point_hits,
        court_limit,
        playable_limit
    ):
        return None

    hits = [copy.deepcopy(h) for h in point_hits]

    cleaned = []
    i = 0

    while i < len(hits):
        y = get_y(hits[i])

        second_last_index = len(hits) - 2

        # Remove huge middle outlier,
        # but do NOT remove second-last out ball
        if (
            abs(y) >= court_limit
            and i < len(hits) - 1
            and i != second_last_index
        ):
            i += 1

            # keep first following out-ball
            if i < len(hits) and abs(get_y(hits[i])) > playable_limit:
                cleaned.append(hits[i])
                i += 1

                # remove trailing out/noise hits
                while (
                    i < len(hits)
                    and abs(get_y(hits[i])) > playable_limit
                ):
                    i += 1

            continue

        cleaned.append(hits[i])
        i += 1

    cleaned = keep_second_last_out_ball_remove_last(
        cleaned,
        playable_limit
    )

    return cleaned