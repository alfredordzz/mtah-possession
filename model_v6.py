"""Possession model v6: v4 plus pass and interception signals for the team call.

Why
    v4 recovers few short possessions (< 3 s): the ball changes team in a second or two and the
    per-frame evidence (who is nearest, who has stayed near) is too weak and too late for the
    decoder to justify two switches. Most of those changes happen through the ball's flight: a
    pass that the other team intercepts, a clearance, a loose ball. The flight itself is visible in
    the ball track, so v6 reads it directly:

    flight       the ball travels fast between two consecutive touches, with no player of either
                 team in reach on the way; the touch that starts it belongs to the sender's team
                 and the touch that ends it to the receiver's team
    turnover     sender and receiver are different teams: evidence that possession changes at the
                 reception (and not before)
    target       while the ball is moving fast, which team has the player nearest to where the ball
                 will be in 0.5 s and 1 s (who is going to receive it)
    direction    whether the ball is moving towards one team's attacking goal

Touches are assigned to the nearer team over a short window around the contact, not only at the
contact frame, because the competition ball track lags the players by a few frames.
All new thresholds are physical (m/s, m, s), set once from football sense and never tuned on the
competition match; the weights are learned on external games only.

On the 8 external games (leave one game out, training/train_final.py) v6 beat v4 on 30 of the 32
game x tracker-variant pairs, by +0.028 to +0.031 spellF1 on average in every variant.
A check of the detector on those games (flights against the true possession, which is allowed
there because it is external data): ~70% of true changes of possession fall within a second of a
detected flight; same-team flights are right 86% of the time, "turnover" flights only ~32% (the
team of a touch is right ~77% of the time with a smooth tracker). The regressions weigh that
uncertainty; nothing is decided by a single flight.
"""

import numpy as np

import model_v4 as v4
from model_v2 import A, D, FPS, H, X, Logistic, emissions, viterbi  # noqa: F401  (re-exported)

common = v4.common
gap_context = v4.gap_context
features_dead = v4.features_dead
nanmean_window = v4.nanmean_window

ASSIGN_WIN = 6        # frames either side of a touch used to decide which team made it
FLIGHT_SPEED = 6.0    # m/s: mean ball speed between two touches that makes it a pass / flight
FLIGHT_MIN_M = 6.0    # m: minimum distance travelled by a flight
FLIGHT_MAX_S = 4.0    # s: longest gap between two touches still treated as one flight
DECAY_S = 2.0         # s: how long the evidence of a reception lasts after it
LOOKAHEAD_S = (0.5, 1.0)


def window_min(x, k):
    """Minimum of x over [t - k, t + k] (inf outside)."""
    n = len(x)
    pad = np.concatenate([np.full(k, np.inf), x, np.full(k, np.inf)])
    return np.min(np.stack([pad[i:i + n] for i in range(2 * k + 1)]), axis=0)


def touch_teams(c, touch):
    """Sign (+1 H, -1 A) and confidence (0..1) of each touch, from distances around the contact."""
    dh = window_min(np.where(np.isfinite(c["d_home"]), c["d_home"], np.inf), ASSIGN_WIN)
    da = window_min(np.where(np.isfinite(c["d_away"]), c["d_away"], np.inf), ASSIGN_WIN)
    with np.errstate(invalid="ignore"):
        margin = np.clip(np.nan_to_num(da - dh, nan=0.0, posinf=8.0, neginf=-8.0), -8, 8)
    sign = np.where(touch, np.where(dh <= da, 1.0, -1.0), 0.0)
    conf = np.where(touch, np.clip(np.abs(margin) / 3.0, 0, 1), 0.0)
    return sign, conf


def flights(c, touch, sign, conf, speed):
    """List of (start, end, sender sign, receiver sign, confidence) between consecutive touches."""
    idx = np.flatnonzero(touch)
    ball = c["_ball"]
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > FLIGHT_MAX_S * FPS or b - a < 3:
            continue
        if not c["seen"][a:b + 1].all():
            continue
        dist = float(np.hypot(*(ball[b] - ball[a])))
        mspeed = np.nanmean(speed[a:b + 1])
        if dist < FLIGHT_MIN_M or not mspeed > FLIGHT_SPEED:
            continue
        out.append((a, b, sign[a], sign[b], min(conf[a], conf[b])))
    return out


def projected_advantage(c, pos, slot_team, ball_s, v, tau):
    """d_away - d_home at the point the ball will reach in tau seconds (players held still)."""
    target = ball_s + v * tau
    diff = pos - target[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    dh = dist[:, slot_team == H].min(axis=1)
    da = dist[:, slot_team == A].min(axis=1)
    with np.errstate(invalid="ignore"):
        adv = np.clip(da - dh, -8, 8)
    return np.where(np.isfinite(adv), adv, 0.0)


def pass_features(c, pos, slot_team):
    n = c["n"]
    _, touch = v4.detect_touches(c)
    sign, conf = touch_teams(c, touch)
    ball_s = v4.smooth_ball(c["_ball"].astype(np.float64), v4.TOUCH_SMOOTH)
    v = v4.ball_velocity(ball_s, k=3)
    speed = np.hypot(v[:, 0], v[:, 1])

    t = np.arange(n)
    recv = np.zeros(n)          # receiver sign, during the flight and decaying after it
    turn = np.zeros(n)          # same, only for flights that change team
    in_flight = np.zeros(n)
    fl = flights(c, touch, sign, conf, speed)
    decay = int(DECAY_S * FPS * 3)
    for a, b, s_from, s_to, cf in fl:
        in_flight[a:b] = 1.0
        w_in = cf * np.ones(b - a)
        e = min(n, b + decay)
        w_after = cf * np.exp(-(t[b:e] - b) / (DECAY_S * FPS))
        recv[a:b] += s_to * w_in
        recv[b:e] += s_to * w_after
        if s_from != s_to:
            turn[a:b] += s_to * w_in
            turn[b:e] += s_to * w_after
    recv = np.clip(recv, -1, 1)
    turn = np.clip(turn, -1, 1)
    # completed passes: signed by the team, over the flight and the second after it
    keep = np.zeros(n)
    for a, b, s_from, s_to, cf in fl:
        if s_from == s_to:
            keep[a:min(n, b + FPS)] = s_from * cf
    # last / next turnover (reception frame), carried with recency
    rec_frame = np.zeros(n, dtype=bool)
    rec_sign = np.zeros(n)
    for a, b, s_from, s_to, cf in fl:
        if s_from != s_to and cf > 0.2:
            rec_frame[b] = True
            rec_sign[b] = s_to
    last_t, d_last = v4.carry(rec_sign, rec_frame, forward=True)
    next_t, d_next = v4.carry(rec_sign, rec_frame, forward=False)

    fast = np.clip((speed - 4.0) / 6.0, 0, 1)
    fast = np.nan_to_num(fast)
    proj = []
    for tau in LOOKAHEAD_S:
        proj.append(projected_advantage(c, pos, slot_team, np.nan_to_num(ball_s), np.nan_to_num(v), tau) * fast)
    with np.errstate(invalid="ignore"):
        fwd = np.nan_to_num(np.clip(v[:, 0] * c["sign_home"], -25, 25)) * c["seen"]
    cols = [recv, turn, keep, in_flight, nanmean_window(recv, 12, 12),
            last_t / (1.0 + d_last / FPS), next_t / (1.0 + d_next / FPS),
            proj[0], proj[1], nanmean_window(proj[1], 12, 12),
            nanmean_window(fwd, 25, 25), nanmean_window(fwd, 75, 75)]
    return np.nan_to_num(np.stack(cols, axis=1)), len(fl)


def features_team(c, pos, slot_team, ball, gap):
    no_ball, with_ball = v4.features_team(c, pos, slot_team, ball, gap)
    pf, _ = pass_features(c, pos, slot_team)
    return np.hstack([no_ball, pf]), np.hstack([with_ball, pf])
