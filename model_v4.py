"""Possession model v4: v3 plus touch detection and sustained-control signals for the team call.

Why
    On held-out validation games v3 knew who had the ball well away from changes of possession
    (~95% of frames right) but only ~67% in the second around a change, and it recovered few of
    the short possessions (< 3 s, about a third of all possessions). Possession is the team of the
    last *controlled* touch, so v4 looks for touches directly:

    touch     the ball's velocity changes sharply while a player is within reach; the touch
              belongs to that player's team
    control   how long, without interruption, one team has had a player glued to the ball

Both are turned into per-frame features (last / next touch team and how recent, touch balance in
short windows, signed control run lengths) for the two team models. The dead-ball side is v3's.
All weights are still learned on external games only.
"""

import numpy as np

import model_v2 as base
import model_v3 as v3
from model_v2 import A, D, FPS, H, X, Logistic, emissions, viterbi  # noqa: F401  (re-exported)

common = base.common
gap_context = v3.gap_context
features_dead = v3.features_dead
nanmean_window = base.nanmean_window

TOUCH_DV = 2.5      # m/s change of ball velocity that counts as a contact
TOUCH_SMOOTH = 4    # frames: centred smoothing of the ball track before differentiating
# Chosen on the validation games for noise invariance: with these settings the detector finds
# 16-24 touches per minute whether the ball positions carry 0, 0.15, 0.3 or 0.6 m of noise
# (unsmoothed it ranged from 17 to 106), so its meaning does not depend on the tracker's jitter.
# The competition tracking turned out smoother than that, with players further from the ball
# (touch_stats.py). Thresholds and reaches were then re-chosen on validation games rebuilt with a
# tracker profile matching those statistics: velocity change 2.5 m/s, touch reach 4 m, control
# reach 3 m. The values below are those; possession_v4.py also reads them from the params file.
TOUCH_REACH = 4.0   # m: a player this close at that moment made the touch
CONTROL_REACH = 3.0  # m: a player this close (and closer than any rival) controls the ball


def ball_velocity(ball, k=2):
    n = len(ball)
    v = np.full((n, 2), np.nan)
    if n > 2 * k:
        v[k:-k] = (ball[2 * k:] - ball[:-2 * k]) / (2 * k) * FPS
    return v


def smooth_ball(ball, half):
    out = np.stack([nanmean_window(ball[:, k], half, half) for k in range(2)], axis=1)
    out[np.isnan(ball[:, 0])] = np.nan
    return out


def detect_touches(c):
    """(team code per frame or -1, touch mask). One touch per local burst of velocity change."""
    ball = smooth_ball(c["_ball"].astype(np.float64), TOUCH_SMOOTH)
    v = ball_velocity(ball, k=3)
    n = len(ball)
    dv = np.full(n, np.nan)
    k = 4
    dv[k:-k] = np.hypot(v[2 * k:, 0] - v[:-2 * k, 0], v[2 * k:, 1] - v[:-2 * k, 1])
    near = np.minimum(c["d_home"], c["d_away"])
    cand = (dv > TOUCH_DV) & (near < TOUCH_REACH)
    cand = np.nan_to_num(cand).astype(bool)
    # keep the frame of maximum velocity change inside each burst
    touch = np.zeros(n, dtype=bool)
    edges = np.diff(np.concatenate(([0], cand.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        touch[a + int(np.nanargmax(dv[a:b]))] = True
    team = np.where(touch, np.where(c["d_home"] <= c["d_away"], H, A), -1)
    return team, touch


def carry(values, valid, forward=True):
    """Last (forward) or next (backward) valid value at each frame, and its distance in frames."""
    n = len(values)
    t = np.arange(n)
    if forward:
        idx = np.maximum.accumulate(np.where(valid, t, -1))
        ok = idx >= 0
        dist = np.where(ok, t - idx, n)
    else:
        idx = np.minimum.accumulate(np.where(valid, t, n)[::-1])[::-1]
        ok = idx < n
        dist = np.where(ok, idx - t, n)
    out = np.where(ok, values[np.clip(idx, 0, n - 1)], 0.0)
    return out, dist


def run_lengths(mask, backward=True):
    """Length of the uninterrupted True run ending (backward) or starting (forward) at each frame."""
    m = mask.astype(np.int64)
    if not backward:
        return run_lengths(mask[::-1], True)[::-1]
    out = np.zeros(len(m), dtype=np.int64)
    run = 0
    for i, val in enumerate(m):
        run = run + 1 if val else 0
        out[i] = run
    return out


def control_features(c):
    team, touch = detect_touches(c)
    sign = np.where(team == H, 1.0, np.where(team == A, -1.0, 0.0))
    last, d_last = carry(sign, touch, forward=True)
    nxt, d_next = carry(sign, touch, forward=False)
    rec_last = 1.0 / (1.0 + d_last / FPS)
    rec_next = 1.0 / (1.0 + d_next / FPS)
    bal_past = nanmean_window(sign, 75, 0) * 76
    bal_future = nanmean_window(sign, 0, 75) * 76

    seen = c["seen"]
    with np.errstate(invalid="ignore"):
        h_ctrl = seen & (c["d_home"] < CONTROL_REACH) & (c["d_home"] < c["d_away"])
        a_ctrl = seen & (c["d_away"] < CONTROL_REACH) & (c["d_away"] < c["d_home"])
    h_back, a_back = run_lengths(h_ctrl), run_lengths(a_ctrl)
    h_fwd, a_fwd = run_lengths(h_ctrl, backward=False), run_lengths(a_ctrl, backward=False)
    ctrl_now = np.log1p(h_back + h_fwd) - np.log1p(a_back + a_fwd)
    # most recent sustained control (>= 0.4 s) by either team, carried forward through gaps
    sustained = np.where((h_back >= 10), 1.0, np.where(a_back >= 10, -1.0, 0.0))
    last_ctrl, d_ctrl = carry(sustained, sustained != 0, forward=True)
    next_ctrl, d_nctrl = carry(np.where(h_fwd >= 10, 1.0, np.where(a_fwd >= 10, -1.0, 0.0)),
                               (h_fwd >= 10) | (a_fwd >= 10), forward=False)

    cols = [last * rec_last, nxt * rec_next, last, nxt,
            np.log1p(np.minimum(d_last, 3000) / FPS), np.log1p(np.minimum(d_next, 3000) / FPS),
            bal_past, bal_future, ctrl_now,
            last_ctrl / (1.0 + np.minimum(d_ctrl, 3000) / FPS / 3.0),
            next_ctrl / (1.0 + np.minimum(d_nctrl, 3000) / FPS / 3.0)]
    return np.nan_to_num(np.stack(cols, axis=1)), int(touch.sum())


def features_team(c, pos, slot_team, ball, gap):
    no_ball, with_ball = v3.features_team(c, pos, slot_team, ball, gap)
    ctrl, _ = control_features(c)
    return np.hstack([no_ball, ctrl]), np.hstack([with_ball, ctrl])
