"""Possession model v8: v6 plus team shape (formation) and counter-press signals, team-symmetric.

Every feature is computed for both teams the same way and entered as home minus away, so the model
learns what an attacking or a pressing shape looks like from the external games, not anything about
a particular club.

    shape          depth and width of each team, compactness (mean distance to the team centroid),
                   share of players in the opponent half and ahead of the ball, distance from the
                   team centroid to the ball
    counter-press  how fast each team's players converge on the ball: change over the last 2 s of
                   the share of its players within 10 m, and its mean speed towards the ball
Counts are shares of the team's visible players, so missing players do not bias them.

Why
    A ceiling analysis on the external games showed that ~90% of what v6 still loses is in the team
    call, not in the dead-ball call. Team shape says who is attacking even when the ball is lost by
    the tracker or contested: the attacking team stretches and pushes players past the ball, the
    defending team closes up behind it, and after a loss the nearest players converge on the ball.

On the 8 external games (leave one game out) v8 beat v6 on 28 of the 32 game x tracker-variant pairs
and on all 8 games of the hardest competition-like variant: mean spellF1 +0.015 (smooth 0.618 ->
0.638, smooth with extra ball gaps 0.601 -> 0.619). Discarded on the way, same protocol: touch
outcome, carries and pressure counts (v7, +0.006, 24 of 32 pairs).
"""

import warnings

import numpy as np

import model_v6 as v6
from model_v2 import A, D, FPS, H, X, Logistic, emissions, viterbi  # noqa: F401

common = v6.common
gap_context = v6.gap_context
features_dead = v6.features_dead
nanmean_window = v6.nanmean_window


def team_shape(pos_t, sign, ball, length):
    """Per-frame shape descriptors of one team. sign = +1 if it attacks towards x = length."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = pos_t[..., 0] * sign[:, None] + np.where(sign[:, None] < 0, length, 0.0)   # 0 = own goal
        y = pos_t[..., 1]
        depth = np.nanpercentile(x, 90, axis=1) - np.nanpercentile(x, 10, axis=1)
        width = np.nanpercentile(y, 90, axis=1) - np.nanpercentile(y, 10, axis=1)
        cx, cy = np.nanmean(pos_t[..., 0], axis=1), np.nanmean(pos_t[..., 1], axis=1)
        compact = np.nanmean(np.hypot(pos_t[..., 0] - cx[:, None], pos_t[..., 1] - cy[:, None]), axis=1)
        visible = np.sum(~np.isnan(x), axis=1).astype(float)
        visible = np.where(visible > 0, visible, np.nan)
        # shares of the visible players, so that players missing from the tracking do not bias them
        in_opp = np.nansum(x > length / 2, axis=1) / visible
        bx = ball[:, 0] * sign + np.where(sign < 0, length, 0.0)
        ahead = np.nansum(x > bx[:, None], axis=1) / visible
        to_ball = np.hypot(cx - ball[:, 0], cy - ball[:, 1])
    return [depth, width, compact, in_opp, ahead, to_ball]


def convergence(pos_t, ball, speed_k=5):
    """(change over 2 s in the share of players within 10 m of the ball, mean speed towards it)."""
    n = len(ball)
    d = np.hypot(pos_t[..., 0] - ball[:, None, 0], pos_t[..., 1] - ball[:, None, 1])
    visible = np.sum(~np.isnan(d), axis=1).astype(float)
    near = np.nansum(d < 10.0, axis=1) / np.where(visible > 0, visible, np.nan)
    near = np.where(np.isnan(ball[:, 0]), np.nan, near)
    lag = 2 * FPS
    change = np.full(n, np.nan)
    change[lag:] = near[lag:] - near[:-lag]
    dd = np.full(d.shape, np.nan)
    dd[speed_k:-speed_k] = (d[2 * speed_k:] - d[:-2 * speed_k]) / (2 * speed_k) * FPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        closing = -np.nanmean(np.where(d < 20.0, dd, np.nan), axis=1)      # m/s towards the ball
    return [change, closing]


def v8_block(c, pos, slot_team):
    ball = c["_ball"].astype(np.float64)
    pos = pos.astype(np.float64)
    sh = c["sign_home"]
    h, a = pos[:, slot_team == H], pos[:, slot_team == A]
    hs, as_ = team_shape(h, sh, ball, c["length"]), team_shape(a, -sh, ball, c["length"])
    hc, ac = convergence(h, ball), convergence(a, ball)
    cols = []
    for x, y in zip(hs, as_):
        diff = x - y
        cols += [nanmean_window(diff, 25, 25), nanmean_window(diff, 100, 100)]
    for x, y in zip(hc, ac):
        diff = x - y
        cols += [nanmean_window(diff, 12, 12), nanmean_window(diff, 50, 0), nanmean_window(diff, 0, 50)]
    return np.nan_to_num(np.stack(cols, axis=1))


def features_team(c, pos, slot_team, ball, gap):
    no_ball, with_ball = v6.features_team(c, pos, slot_team, ball, gap)
    b = v8_block(c, pos, slot_team)
    return np.hstack([no_ball, b]), np.hstack([with_ball, b])
