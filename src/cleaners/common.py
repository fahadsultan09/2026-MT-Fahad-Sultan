from cleaners.base import (
    all_same_side,
    normalize_boundary_y,
    keep_second_last_out_ball_remove_last
)


def clean_common(point_hits, court_limit=12.5, playable_limit=11.885):

    # remove all-same-side only if 3 or more hits
    # keep valid 2-hit finish points
    if len(point_hits) >= 3 and all_same_side(point_hits):
        return None

    point_hits = normalize_boundary_y(
        point_hits,
        court_limit
    )

    point_hits = keep_second_last_out_ball_remove_last(
        point_hits,
        playable_limit
    )

    return point_hits