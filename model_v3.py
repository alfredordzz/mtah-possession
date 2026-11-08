"""Possession model v3: v2 plus the context of ball-track gaps.

What changed from v2, and why
    v2 never used ball visibility, on the assumption that the competition tracking loses the ball
    in play. An automatic check of the provided tracking (ball_gap_stats.py) showed the opposite:
    few, long gaps (median ~11 s), missing 70% of the time when players walk and 11% when they
    run - the signature of stoppages, as in the validation games. v3 therefore feeds the dead-ball
    model with gap context (is the ball seen, for how long has it been gone, where was it last
    seen), and the team model with the team that had the ball when it vanished and the team that
    has it when it reappears. All weights are still learned on external games only.

Everything else (logistic regressions, Viterbi, per-frame scores) is shared with model_v2.
"""

import numpy as np

import model_v2 as base
from model_v2 import A, D, FPS, H, X, Logistic, emissions, viterbi  # noqa: F401  (re-exported)

common = base.common
nanmean_window = base.nanmean_window


def gap_context(c):
    """Per-frame descriptors of the ball-track gap a frame sits in (zeros where the ball is seen)."""
    n = c["n"]
    seen = c["seen"]
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, t, -1))
    nxt = np.minimum.accumulate(np.where(seen, t, n)[::-1])[::-1]
    since = np.where(last >= 0, t - last, n) / FPS
    until = np.where(nxt < n, nxt - t, n) / FPS
    since = np.where(seen, 0.0, np.minimum(since, 120.0))
    until = np.where(seen, 0.0, np.minimum(until, 120.0))
    length = np.where(seen, 0.0, since + until)

    ball = c["_ball"]
    lx = np.where(last >= 0, ball[np.clip(last, 0, n - 1), 0], np.nan)
    ly = np.where(last >= 0, ball[np.clip(last, 0, n - 1), 1], np.nan)
    L, W = c["length"], c["width"]
    with np.errstate(invalid="ignore"):
        off = (lx < -0.5) | (lx > L + 0.5) | (ly < -0.5) | (ly > W + 0.5)
        edge = np.minimum.reduce([lx, L - lx, ly, W - ly])
        near_line = (edge < 5) & ~off
        box = ((lx < 16.5) | (lx > L - 16.5)) & (ly > 13.85) & (ly < W - 13.85)
    flags = [np.where(seen, 0.0, np.nan_to_num(f.astype(float))) for f in (off, near_line, box)]
    return dict(since=since, until=until, length=length, last=last, nxt=nxt,
                off=flags[0], near_line=flags[1], box=flags[2])


def features_dead(c, gap):
    """One model for every frame: v2's player and ball features plus visibility and gap context."""
    _, with_ball = base.features_dead(c)
    seen = c["seen"].astype(float)
    cols = [seen]
    for before, after in ((25, 25), (75, 75), (125, 0), (0, 125)):
        cols.append(nanmean_window(seen, before, after))
    for key in ("since", "until", "length"):
        cols.append(np.log1p(gap[key]))
    cols += [gap["off"], gap["near_line"], gap["box"]]
    # interaction: a long gap while the players are slow is the strongest stoppage signal
    slow = nanmean_window(c["slow_share"], 50, 50)
    cols.append(np.log1p(gap["length"]) * np.nan_to_num(slow))
    return np.nan_to_num(np.hstack([with_ball, np.stack(cols, axis=1)]))


def features_team(c, pos, slot_team, ball, gap):
    """v2 team features; the no-ball set also gets the team that had / regains the ball."""
    no_ball, with_ball = base.features_team(c, pos, slot_team, ball)
    with np.errstate(invalid="ignore"):
        adv = np.clip(c["d_away"] - c["d_home"], -8, 8)
    adv = np.where(np.isfinite(adv) & c["seen"], adv, np.nan)
    adv_s = nanmean_window(adv, 12, 0)          # last ~half second before each frame
    adv_n = nanmean_window(adv, 0, 12)          # first ~half second after
    n = c["n"]
    before = np.where(gap["last"] >= 0, adv_s[np.clip(gap["last"], 0, n - 1)], 0.0)
    after = np.where(gap["nxt"] < n, adv_n[np.clip(gap["nxt"], 0, n - 1)], 0.0)
    w_before = 1.0 / (1.0 + gap["since"] / 5.0)  # the older the sighting, the less it says
    w_after = 1.0 / (1.0 + gap["until"] / 5.0)
    extra = np.stack([np.nan_to_num(before), np.nan_to_num(after),
                      np.nan_to_num(before) * w_before, np.nan_to_num(after) * w_after,
                      np.log1p(gap["length"])], axis=1)
    return np.hstack([no_ball, extra]), with_ball
