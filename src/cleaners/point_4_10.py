#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy

from cleaners.base import get_y, set_y


def sign_of_hit(hit):
    y = get_y(hit)

    if y > 0:
        return 1
    if y < 0:
        return -1
    return 0


def short_fault_rally(point_hits, playable_limit=11.885):
    """
    Remove very short rallies that likely represent
    a first-serve fault rather than a completed point.

    Conditions:
    - 3 hits or fewer
    - last hit outside playable area
    """

    if len(point_hits) > 3:
        return False

    last_y = get_y(point_hits[-1])

    if abs(last_y) > playable_limit:
        return True

    return False


def average_same_side_runs_between_opposites(point_hits):
    hits = [copy.deepcopy(h) for h in point_hits]

    result = []
    i = 0

    while i < len(hits):
        current_sign = sign_of_hit(hits[i])

        if current_sign == 0:
            result.append(hits[i])
            i += 1
            continue

        run_start = i
        run = [hits[i]]
        i += 1

        while i < len(hits) and sign_of_hit(hits[i]) == current_sign:
            run.append(hits[i])
            i += 1

        prev_sign = (sign_of_hit(hits[run_start - 1]) if run_start
                     > 0 else None)
        next_sign = (sign_of_hit(hits[i]) if i < len(hits) else None)

        if len(run) >= 2 and prev_sign is not None and next_sign \
            is not None and prev_sign == next_sign and prev_sign \
            != current_sign:
            avg_y = sum(get_y(h) for h in run) / len(run)

            averaged_hit = run[0]
            set_y(averaged_hit, avg_y)

            result.append(averaged_hit)
        else:

            result.extend(run)

    return result


def trim_after_late_out_of_play(point_hits, playable_limit=11.885):
    n = len(point_hits)

    if n < 3:
        return point_hits

    for idx in [n - 3, n - 2]:
        y = get_y(point_hits[idx])

        if abs(y) > playable_limit:
            current_sign = (1 if y > 0 else -1)

            # check if next hits remain same side

            next_hits = point_hits[idx + 1:]

            if all((get_y(h) > 0) == (current_sign > 0) for h in
                   next_hits):
                return point_hits[:idx + 1]

    return point_hits

def remove_clear_middle_outliers(point_hits, court_limit=12.5):
    cleaned = []

    for i, h in enumerate(point_hits):
        y = get_y(h)
        is_last = i == len(point_hits) - 1

        if abs(y) >= court_limit and not is_last:
            continue

        cleaned.append(copy.deepcopy(h))

    return cleaned


def all_hits_outside_same_side(point_hits, playable_limit=11.885):

    if not point_hits:
        return False

    first_y = get_y(point_hits[0])

    first_sign = (1 if first_y > 0 else -1)

    for h in point_hits:

        y = get_y(h)

        current_sign = (1 if y > 0 else -1)

        # different side

        if current_sign != first_sign:
            return False

        # still playable

        if abs(y) <= playable_limit:
            return False

    return True

def remove_single_outliers(point_hits, court_limit=12.5):
    cleaned = []

    for i, h in enumerate(point_hits):
        y = get_y(h)

        is_last = i == len(point_hits) - 1

        # remove only non-last clear outliers
        if abs(y) >= court_limit and not is_last:
            continue

        cleaned.append(copy.deepcopy(h))

    return cleaned

def clean_point_4_10(point_hits,court_limit=12.5,playable_limit=11.885):

    # Rule:
    # remove short fault rallies
    if short_fault_rally(
        point_hits,
        playable_limit
    ):
        return None

    # Rule:
    # remove fully invalid same-side out rallies
    if all_hits_outside_same_side(
        point_hits,
        playable_limit
    ):
        return None

    # Rule:
    # remove obvious outliers
    cleaned = remove_single_outliers(
        point_hits,
        court_limit
    )

    # Rule:
    # additional middle outlier cleanup
    cleaned = remove_clear_middle_outliers(
        cleaned,
        court_limit
    )

    # Rule:
    # merge consecutive same-side hits between opposites
    cleaned = average_same_side_runs_between_opposites(
        cleaned
    )

    # Rule:
    # trim trailing noise after rally-ending out ball
    cleaned = trim_after_late_out_of_play(
        cleaned,
        playable_limit
    )

    return cleaned