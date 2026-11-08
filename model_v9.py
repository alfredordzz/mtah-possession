"""Possession model v9: v8 plus collective movement reversal, trained with weight on the changes.

Why
    The ceiling and error analyses on the external games showed that what v8 still loses is the
    team call around changes of possession, and that re-arranging the same evidence (semi-Markov
    decoding, moving boundaries, a second-stage judge of short possessions, a context-dependent
    switch cost) does not help. Two things did:

    reversal   when possession changes, both teams turn round within a second or two: the team that
               lost the ball runs towards its own goal, the team that won it starts moving up. For
               each team, the mean velocity of its players towards the goal it attacks, and how it
               changes between just before and just after each frame (0.5, 1 and 2 s); home minus
               away. It is visible in the players even when the ball track is blurred or missing.
    weighting  the two team regressions are fitted with frames within 1 s of a change of possession
               counted TRANSITION_WEIGHT times (training/train_final.py). Neither idea alone moved
               much (+0.006 and +0.009); together the regressions learn to use the reversal where it
               matters.

On the 8 external games (leave one game out) v9 beat v8 on all 32 game x tracker-variant pairs:
mean spellF1 +0.019 (smooth 0.635 -> 0.654, smooth with extra ball gaps 0.620 -> 0.635). Weights
of 2 and 5 were also tried (+0.014, 27 of 32; +0.020, 28 of 32); 3 was kept.
"""

import warnings

import numpy as np

import model_v8 as v8
from model_v2 import A, D, FPS, H, X, Logistic, emissions, viterbi  # noqa: F401  (re-exported)

common = v8.common
gap_context = v8.gap_context
features_dead = v8.features_dead
nanmean_window = v8.nanmean_window

TRANSITION_WEIGHT = 3.0   # fitting weight of frames within TRANSITION_WINDOW of a change of possession
TRANSITION_WINDOW = 25    # frames (1 s)
REVERSAL_WINDOWS = (12, 25, 50)


def forward_velocity(pos_t, sign, k=3):
    """Mean over visible players of each player's velocity towards the goal the team attacks (m/s)."""
    v = np.full(pos_t.shape[:2], np.nan)
    v[k:-k] = (pos_t[2 * k:, :, 0] - pos_t[:-2 * k, :, 0]) / (2 * k) * FPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(v, axis=1) * sign


def reversal_block(c, pos, slot_team):
    pos = pos.astype(np.float64)
    sh = c["sign_home"]
    vh = forward_velocity(pos[:, slot_team == H], sh)
    va = forward_velocity(pos[:, slot_team == A], -sh)
    w = nanmean_window
    cols = []
    for k in REVERSAL_WINDOWS:
        dh = w(vh, 0, k) - w(vh, k, 0)          # home speeds up towards its attack after t
        da = w(va, 0, k) - w(va, k, 0)
        cols += [dh - da, dh, -da]
    cols += [w(vh - va, 6, 6), w(vh - va, 0, 25), w(vh - va, 25, 0)]
    return np.nan_to_num(np.stack(cols, axis=1))


def features_team(c, pos, slot_team, ball, gap):
    no_ball, with_ball = v8.features_team(c, pos, slot_team, ball, gap)
    r = reversal_block(c, pos, slot_team)
    return np.hstack([no_ball, r]), np.hstack([with_ball, r])


def transition_weights(gt, possession_codes=(H, A)):
    """Fitting weights: TRANSITION_WEIGHT within TRANSITION_WINDOW frames of an H<->A change, else 1."""
    gt = np.asarray(gt)
    h, a = possession_codes
    near = np.zeros(len(gt), dtype=bool)
    alive = (gt == h) | (gt == a)
    for t in np.flatnonzero((gt[1:] != gt[:-1]) & alive[1:] & alive[:-1]) + 1:
        near[max(0, t - TRANSITION_WINDOW):t + TRANSITION_WINDOW] = True
    return np.where(near, TRANSITION_WEIGHT, 1.0)
