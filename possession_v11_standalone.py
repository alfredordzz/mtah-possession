"""v11 in a single file: possession for the competition match, from the raw tracking to the CSV.

    python possession_v11_standalone.py                  # writes evaluation/results/v11_standalone.csv
    python possession_v11_standalone.py --out BAR-ATH.csv
    python possession_v11_standalone.py --rebuild-cache  # re-read the raw tracking even if cached

v11 is v10 with one change: the sequence model is the average of three networks instead of one. The
three are trained identically on the 8 external games and differ only in their random seed (0, 1, 2);
their probabilities are averaged frame by frame before the usual 50/50 average with the judges. Every
other step (clues, judges, Smart Tagging, Viterbi) is the same code and the same numbers as v10.
Validated on external data only: +0.013 spellF1 over v10 on 8 games x 4 variants (26/32), repeated
with two other noise seeds (28/32, 24/32), and on 20 SkillCorner games (+0.010, 16/20).

Needs only numpy and
    model_v9_1_params.json       the two judges and the touch/flight thresholds (8 external games)
    model_v11_seq.npz            the three networks' weights and input scaling (same 8 games);
                                 built with training/train_seq.py and training/build_v11_seq.py
    games.py, starter/, evaluation/   the organisers' loaders and checks, unchanged
    data/                        the provided tracking, metadata and Smart Tagging
The first run reads the raw tracking once into cache/atd_tactical.npz (a minute or two); later runs
reuse it. Nothing from the ground truth or the video is read.

Prints counts and format checks only: no comparison against the public ground truth.

Pipeline
    0. read the tracking   raw ATD -> metres, speeds, team of each slot; checks against DATA.md
    1. common arrays       distances to the ball, speeds, attack direction per half
    2. dead-ball judge      player speeds + ball position/speed + ball-gap context -> logit P(D)
    3. team judge           positions, touches, flights, shape, counter-press, reversal -> logit P(H)
    4. Smart Tagging        +/-0.25 on the team logit where a tagged phase names a team
    5. sequence model       three bidirectional GRUs at 5 Hz over the same 168 features; their
                            probabilities averaged -> log P(H, A, D)
    6. average              (judges + sequence model) / 2, in log-probability, frame by frame
    7. Viterbi per half     switch cost H<->A 16, dead cost 35; D forced before the tagged kickoff
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_names, team_of_each_track, write_submission  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
import possession_metrics as pm  # noqa: E402

CACHE_PATH = os.path.join(ROOT, "cache", "atd_tactical.npz")
PARAMS_PATH = os.path.join(ROOT, "model_v9_1_params.json")
SEQ_PATH = os.path.join(ROOT, "model_v11_seq.npz")
OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v11_standalone.csv")

H, A, D, X = 0, 1, 2, 3
FPS = 25

# Physical thresholds. These are the defaults; main() overwrites them from model_v9_1_params.json,
# so the params file is the single source of truth.
TOUCH_DV = 2.5        # m/s change of ball velocity that counts as a contact
TOUCH_SMOOTH = 4      # frames: centred smoothing of the ball track before differentiating
TOUCH_REACH = 4.0     # m: a player this close at that moment made the touch
CONTROL_REACH = 3.0   # m: a player this close (and closer than any rival) controls the ball
ASSIGN_WIN = 6        # frames either side of a touch used to decide which team made it
FLIGHT_SPEED = 6.0    # m/s: mean ball speed between two touches that makes it a flight
FLIGHT_MIN_M = 6.0    # m: minimum distance travelled by a flight
FLIGHT_MAX_S = 4.0    # s: longest gap between two touches still treated as one flight
DECAY_S = 2.0         # s: how long the evidence of a reception lasts after it
LOOKAHEAD_S = (0.5, 1.0)
REVERSAL_WINDOWS = (12, 25, 50)

# Smart Tagging exists only for this match, so its weight cannot be learned; it is a small nudge
# (checked on external games with simulated phases: 0.25-0.5 neutral, 1 or more hurts).
TAG_LOGIT = 0.25
REVERSAL_RANGE = "p10 -2.0 to -1.8, p90 1.8 to 2.1 m/s"
OFFENSIVE = {"build up", "progression", "creating chances", "attacking transition", "long ball",
             "ball in final third", "ball in the box", "players in the box", "set pieces"}
DEFENSIVE = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}


def apply_params(params):
    """Read the touch and flight thresholds from the params file."""
    global TOUCH_DV, TOUCH_REACH, TOUCH_SMOOTH, CONTROL_REACH
    global ASSIGN_WIN, FLIGHT_SPEED, FLIGHT_MIN_M, FLIGHT_MAX_S, DECAY_S, LOOKAHEAD_S
    t = params.get("touch", {})
    TOUCH_DV = t.get("dv", TOUCH_DV)
    TOUCH_REACH = t.get("reach", TOUCH_REACH)
    TOUCH_SMOOTH = int(t.get("smooth", TOUCH_SMOOTH))
    CONTROL_REACH = t.get("control_reach", CONTROL_REACH)
    f = params.get("flight") or {}
    ASSIGN_WIN = int(f.get("assign_win", ASSIGN_WIN))
    FLIGHT_SPEED = f.get("speed", FLIGHT_SPEED)
    FLIGHT_MIN_M = f.get("min_m", FLIGHT_MIN_M)
    FLIGHT_MAX_S = f.get("max_s", FLIGHT_MAX_S)
    DECAY_S = f.get("decay_s", DECAY_S)
    LOOKAHEAD_S = tuple(f.get("lookahead_s", LOOKAHEAD_S))



# == 0. read the raw tracking once (was cache_atd.py) ===============================================

FIELDS_PER_PLAYER = 3  # x, y, speed


def check(name, ok, detail=""):
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", name,
                               "  ({})".format(detail) if detail else ""))
    return ok


def read_raw(raw_path, n_frames, n_slots):
    """Parse every line into fixed-size arrays indexed by video frame (frame_id - 1).
    raw line: "<frame_id>:<player>;<player>;...:<ball>", each group "x,y,speed", NaN if missing."""
    players = np.full((n_frames, n_slots, FIELDS_PER_PLAYER), np.nan, dtype=np.float32)
    ball_rows = [None] * n_frames
    bad_lines = 0
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").rstrip("\r").split(":")
            if len(parts) != 3:
                bad_lines += 1
                continue
            frame = int(parts[0]) - 1
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            if values.size != n_slots * FIELDS_PER_PLAYER:
                bad_lines += 1
                continue
            players[frame] = values.reshape(n_slots, FIELDS_PER_PLAYER)
            ball_rows[frame] = np.array(parts[2].split(","), dtype=np.float32)
            seen[frame] = True

    n_ball_fields = max((row.size for row in ball_rows if row is not None), default=2)
    ball = np.full((n_frames, n_ball_fields), np.nan, dtype=np.float32)
    for frame, row in enumerate(ball_rows):
        if row is not None:
            ball[frame, :row.size] = row
    return players, ball, seen, bad_lines


def build_cache():
    """Raw ATD -> cache/atd_tactical.npz in metres (ball, pos, speed, slot_team), with the checks
    against the numbers published in DATA.md / FAQ.md. Slot order and team come from the metadata
    XML, with the same rule as starter/baseline.py: the first team is home (H)."""
    t0 = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n = source_frames()
    pitch_l, pitch_w = GAMES[GAME]["pitch"]
    slot_team = team_of_each_track(meta_path)
    s = len(slot_team)
    print("reading the raw tracking once into {} (a minute or two)...".format(os.path.relpath(CACHE_PATH, ROOT)))
    print("  slots: {} total, {} home, {} away".format(s, int((slot_team == 0).sum()), int((slot_team == 1).sum())))
    players, ball_raw, seen, bad_lines = read_raw(raw_path, n, s)

    check("every video frame has exactly one record", bool(seen.all()), "{} of {} frames".format(int(seen.sum()), n))
    check("no malformed lines", bad_lines == 0, "{} bad".format(bad_lines))

    scale = np.array([pitch_l, pitch_w], dtype=np.float32)
    pos = players[:, :, :2] * scale
    speed = players[:, :, 2].copy()
    ball = ball_raw[:, :2] * scale
    ball_extra = ball_raw[:, 2:]

    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    ball_ok = ~np.isnan(ball).any(axis=1)
    tracked = (~np.isnan(pos).any(axis=2)).sum(axis=1)

    pct_all, pct_graded = 100 * ball_ok.mean(), 100 * ball_ok[graded].mean()
    check("ball present on ~56.9% of all frames", abs(pct_all - 56.9) < 0.5, "{:.1f}%".format(pct_all))
    check("ball present on ~59.3% of graded frames", abs(pct_graded - 59.3) < 0.5, "{:.1f}%".format(pct_graded))
    check("~20.5 tracked objects per graded frame", abs(tracked[graded].mean() - 20.5) < 0.5,
          "{:.1f}".format(tracked[graded].mean()))
    check("graded frame count is 158,857", int(graded.sum()) == 158857, str(int(graded.sum())))
    inside = ((ball[ball_ok, 0] > -5) & (ball[ball_ok, 0] < pitch_l + 5)
              & (ball[ball_ok, 1] > -5) & (ball[ball_ok, 1] < pitch_w + 5)).mean()
    check("ball coordinates look like metres", inside > 0.98, "{:.1%} inside padded pitch".format(inside))
    sp = speed[graded][~np.isnan(speed[graded])]
    check("player speeds look like m/s", 0.5 < np.median(sp) < 4 and np.percentile(sp, 99.9) < 12,
          "median {:.1f}, p99.9 {:.1f}".format(np.median(sp), np.percentile(sp, 99.9)))
    centre = np.array([pitch_l / 2, pitch_w / 2], dtype=np.float32)
    for label, frame in (("1st half kickoff", 11185), ("2nd half kickoff", 113742)):
        window = slice(frame - 25, frame + 1)
        visible = ball_ok[window]
        if visible.any():
            dist = np.min(np.linalg.norm(ball[window][visible] - centre, axis=1))
            check("{}: ball within 3 m of centre spot".format(label), dist < 3.0, "{:.1f} m".format(dist))
        else:
            check("{}: ball visible in the second before".format(label), False, "ball missing")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(CACHE_PATH, ball=ball, ball_extra=ball_extra, pos=pos, speed=speed, slot_team=slot_team)
    print("  saved in {:.0f}s (derived match data: keep cache/ out of git)\n".format(time.time() - t0))


# == 0b. helpers ====================================================================================

def nanmean_window(x, before, after):
    """Mean of x over [t - before, t + after], ignoring NaN. x is 1-D or 2-D (rows = time)."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n = len(x)
    valid = ~np.isnan(x)
    cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(np.where(valid, x, 0.0), axis=0)])
    cc = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(valid, axis=0)])
    i = np.arange(n)
    lo = np.clip(i - before, 0, n)
    hi = np.clip(i + after + 1, 0, n)
    count = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, (cs[hi] - cs[lo]) / np.maximum(count, 1), np.nan)
    return out[:, 0] if squeeze else out


def robust_z(x, mask):
    """Standardise by the match's own median and IQR inside mask (per-match normalisation)."""
    ref = x[mask & np.isfinite(x)]
    if len(ref) < 10:
        return np.zeros_like(x)
    med = np.median(ref)
    iqr = np.subtract(*np.percentile(ref, [75, 25])) or 1.0
    return (x - med) / iqr


def in_periods(n, periods):
    mask = np.zeros(n, dtype=bool)
    for first, last in periods:
        mask[first:last + 1] = True
    return mask


def ball_kinematics(ball):
    """Ball speed (m/s) from positions over +/-2 frames; NaN where not seen."""
    n = len(ball)
    speed = np.full(n, np.nan)
    k = 2
    if n > 2 * k:
        d = np.hypot(ball[2 * k:, 0] - ball[:-2 * k, 0], ball[2 * k:, 1] - ball[:-2 * k, 1])
        speed[k:-k] = d / (2 * k) * FPS
    return speed


def attack_signs(pos, slot_team, periods, length):
    """+1 if a team attacks towards x = length in that period, else -1. From mean positions."""
    signs = []
    for first, last in periods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hx = np.nanmean(pos[first:last + 1][:, slot_team == H, 0])
            ax = np.nanmean(pos[first:last + 1][:, slot_team == A, 0])
        s_home = 1.0 if hx < ax else -1.0
        signs.append((s_home, -s_home))
    return signs


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


def window_min(x, k):
    """Minimum of x over [t - k, t + k] (inf outside)."""
    n = len(x)
    pad = np.concatenate([np.full(k, np.inf), x, np.full(k, np.inf)])
    return np.min(np.stack([pad[i:i + n] for i in range(2 * k + 1)]), axis=0)


# == 1. common arrays (v2) and ball-gap context (v3) ===============================================

def common(ball, pos, speed, slot_team, periods, pitch):
    """Arrays shared by both judges."""
    n = len(ball)
    length, width = pitch
    graded = in_periods(n, periods)
    seen = ~np.isnan(ball[:, 0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med_speed = np.nanmedian(speed, axis=1)
        mean_speed = np.nanmean(speed, axis=1)
        sorted_speed = np.sort(np.where(np.isnan(speed), -np.inf, speed), axis=1)[:, ::-1]
        top3 = np.where(np.isfinite(sorted_speed[:, :3]), sorted_speed[:, :3], np.nan)
        top3_speed = np.nanmean(top3, axis=1)
        slow_share = np.nanmean(np.where(np.isnan(speed), np.nan, speed < 1.0), axis=1)

    diff = pos - ball[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, slot_team == H].min(axis=1)
    d_away = dist[:, slot_team == A].min(axis=1)

    bspeed = ball_kinematics(ball)
    signs = attack_signs(pos, slot_team, periods, length)
    sign_home = np.zeros(n)
    for (first, last), (sh, _) in zip(periods, signs):
        sign_home[first:last + 1] = sh

    return dict(n=n, graded=graded, seen=seen, _ball=ball, med_speed=med_speed, mean_speed=mean_speed,
                top3_speed=top3_speed, slow_share=slow_share, d_home=d_home, d_away=d_away,
                bspeed=bspeed, sign_home=sign_home, length=length, width=width)


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


# == 2. dead-ball judge features (v2 + v3) =========================================================

def features_dead(c, gap):
    """One matrix for every frame: player speeds, ball position/speed, visibility, gap context."""
    g = c["graded"]
    cols = []
    for name in ("med_speed", "mean_speed", "top3_speed"):
        z = robust_z(c[name], g)
        for before, after in ((12, 12), (50, 50), (125, 0), (0, 125)):
            cols.append(nanmean_window(z, before, after))
    slow = c["slow_share"]
    for before, after in ((25, 25), (75, 75)):
        cols.append(nanmean_window(slow, before, after))
    m = robust_z(c["med_speed"], g)
    cols.append(nanmean_window(m, 0, 25) - nanmean_window(m, 50, -25))
    players = np.nan_to_num(np.stack(cols, axis=1))

    length, width = c["length"], c["width"]
    bx, by = c["_ball"][:, 0], c["_ball"][:, 1]
    with np.errstate(invalid="ignore"):
        outside = np.maximum.reduce([-bx, bx - length, -by, by - width])
        out = (outside > 0.5).astype(float)
        edge = np.clip(np.minimum.reduce([bx, length - bx, by, width - by]), 0, 15)
    bs = np.log(np.clip(c["bspeed"], 0.1, 40))
    still_1s = nanmean_window(np.where(np.isnan(c["bspeed"]), np.nan, c["bspeed"] < 1.0), 25, 25)
    nearest = np.minimum(c["d_home"], c["d_away"])
    ball_cols = [out, nanmean_window(out, 25, 25), edge, bs, nanmean_window(bs, 12, 12),
                 still_1s, np.log(np.clip(nearest, 0.2, 40))]
    with_ball = np.nan_to_num(np.hstack([players, np.stack(ball_cols, axis=1)]))

    seen = c["seen"].astype(float)
    gcols = [seen]
    for before, after in ((25, 25), (75, 75), (125, 0), (0, 125)):
        gcols.append(nanmean_window(seen, before, after))
    for key in ("since", "until", "length"):
        gcols.append(np.log1p(gap[key]))
    gcols += [gap["off"], gap["near_line"], gap["box"]]
    slow_w = nanmean_window(c["slow_share"], 50, 50)
    gcols.append(np.log1p(gap["length"]) * np.nan_to_num(slow_w))
    return np.nan_to_num(np.hstack([with_ball, np.stack(gcols, axis=1)]))


# == 3. team judge features ========================================================================

# -- 3a. positions and nearest player (v2) --------------------------------------------------------

def team_base(c, pos, slot_team, ball):
    """(no_ball, with_ball) features, oriented so that positive favours H."""
    length = c["length"]
    sh = c["sign_home"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hx = pos[:, slot_team == H, 0]
        ax = pos[:, slot_team == A, 0]
        hy = pos[:, slot_team == H, 1]
        ay = pos[:, slot_team == A, 1]
        h_adv = np.nanmean(hx, axis=1) * sh
        a_adv = np.nanmean(ax, axis=1) * -sh
        h_line = np.nanpercentile(hx * sh[:, None], 15, axis=1)
        a_line = np.nanpercentile(ax * -sh[:, None], 15, axis=1)
        h_spread = np.nanstd(hx, axis=1) + np.nanstd(hy, axis=1)
        a_spread = np.nanstd(ax, axis=1) + np.nanstd(ay, axis=1)
    h_vel = np.gradient(nanmean_window(h_adv, 12, 12)) * FPS
    a_vel = np.gradient(nanmean_window(a_adv, 12, 12)) * FPS

    raw = [h_adv - a_adv, h_line - a_line, h_spread - a_spread, h_vel - a_vel]
    cols = []
    for r in raw:
        for before, after in ((25, 25), (100, 100)):
            cols.append(nanmean_window(r, before, after))
    no_ball = np.nan_to_num(np.stack(cols, axis=1))

    with np.errstate(invalid="ignore"):
        adv = np.clip(c["d_away"] - c["d_home"], -8, 8)
    adv = np.where(np.isfinite(adv) & c["seen"], adv, np.nan)
    near = np.minimum(c["d_home"], c["d_away"])
    close = np.where(c["seen"], np.clip((4 - near) / 3, 0, 1), np.nan)
    ball_or = np.where(c["seen"], ball[:, 0] * sh - (length / 2) * sh, np.nan)
    bcols = [adv, nanmean_window(adv, 6, 6), nanmean_window(adv, 25, 25),
             adv * close, nanmean_window(adv * close, 12, 12), ball_or]
    with_ball = np.nan_to_num(np.hstack([no_ball, np.stack(bcols, axis=1)]))
    return no_ball, with_ball


# -- 3b. who had / regains the ball across a gap (v3; only in the no-ball set) --------------------

def gap_team_block(c, gap):
    with np.errstate(invalid="ignore"):
        adv = np.clip(c["d_away"] - c["d_home"], -8, 8)
    adv = np.where(np.isfinite(adv) & c["seen"], adv, np.nan)
    adv_s = nanmean_window(adv, 12, 0)
    adv_n = nanmean_window(adv, 0, 12)
    n = c["n"]
    before = np.where(gap["last"] >= 0, adv_s[np.clip(gap["last"], 0, n - 1)], 0.0)
    after = np.where(gap["nxt"] < n, adv_n[np.clip(gap["nxt"], 0, n - 1)], 0.0)
    w_before = 1.0 / (1.0 + gap["since"] / 5.0)
    w_after = 1.0 / (1.0 + gap["until"] / 5.0)
    return np.stack([np.nan_to_num(before), np.nan_to_num(after),
                     np.nan_to_num(before) * w_before, np.nan_to_num(after) * w_after,
                     np.log1p(gap["length"])], axis=1)


# -- 3c. touches and control (v4) -----------------------------------------------------------------

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
    touch = np.zeros(n, dtype=bool)
    edges = np.diff(np.concatenate(([0], cand.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        touch[a + int(np.nanargmax(dv[a:b]))] = True
    team = np.where(touch, np.where(c["d_home"] <= c["d_away"], H, A), -1)
    return team, touch


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


# -- 3d. ball flights: passes, interceptions, where the ball is going (v6) ------------------------

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
    _, touch = detect_touches(c)
    sign, conf = touch_teams(c, touch)
    ball_s = smooth_ball(c["_ball"].astype(np.float64), TOUCH_SMOOTH)
    v = ball_velocity(ball_s, k=3)
    speed = np.hypot(v[:, 0], v[:, 1])

    t = np.arange(n)
    recv = np.zeros(n)
    turn = np.zeros(n)
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
    keep = np.zeros(n)
    for a, b, s_from, s_to, cf in fl:
        if s_from == s_to:
            keep[a:min(n, b + FPS)] = s_from * cf
    rec_frame = np.zeros(n, dtype=bool)
    rec_sign = np.zeros(n)
    for a, b, s_from, s_to, cf in fl:
        if s_from != s_to and cf > 0.2:
            rec_frame[b] = True
            rec_sign[b] = s_to
    last_t, d_last = carry(rec_sign, rec_frame, forward=True)
    next_t, d_next = carry(rec_sign, rec_frame, forward=False)

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


# -- 3e. team shape and counter-press (v8) --------------------------------------------------------

def team_shape(pos_t, sign, ball, length):
    """Per-frame shape descriptors of one team. sign = +1 if it attacks towards x = length."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = pos_t[..., 0] * sign[:, None] + np.where(sign[:, None] < 0, length, 0.0)
        y = pos_t[..., 1]
        depth = np.nanpercentile(x, 90, axis=1) - np.nanpercentile(x, 10, axis=1)
        width = np.nanpercentile(y, 90, axis=1) - np.nanpercentile(y, 10, axis=1)
        cx, cy = np.nanmean(pos_t[..., 0], axis=1), np.nanmean(pos_t[..., 1], axis=1)
        compact = np.nanmean(np.hypot(pos_t[..., 0] - cx[:, None], pos_t[..., 1] - cy[:, None]), axis=1)
        visible = np.sum(~np.isnan(x), axis=1).astype(float)
        visible = np.where(visible > 0, visible, np.nan)
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
        closing = -np.nanmean(np.where(d < 20.0, dd, np.nan), axis=1)
    return [change, closing]


def shape_block(c, pos, slot_team):
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


# -- 3f. collective movement reversal (v9) --------------------------------------------------------

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
        dh = w(vh, 0, k) - w(vh, k, 0)
        da = w(va, 0, k) - w(va, k, 0)
        cols += [dh - da, dh, -da]
    cols += [w(vh - va, 6, 6), w(vh - va, 0, 25), w(vh - va, 25, 0)]
    return np.nan_to_num(np.stack(cols, axis=1))


# -- 3g. the two team matrices, columns in the order the weights expect ---------------------------

def features_team(c, pos, slot_team, ball, gap):
    """(no_ball, with_ball): the first for frames without a ball position, the second with one."""
    no_ball, with_ball = team_base(c, pos, slot_team, ball)
    no_ball = np.hstack([no_ball, gap_team_block(c, gap)])
    ctrl, _ = control_features(c)
    pf, _ = pass_features(c, pos, slot_team)
    shape = shape_block(c, pos, slot_team)
    rev = reversal_block(c, pos, slot_team)
    return (np.hstack([no_ball, ctrl, pf, shape, rev]),
            np.hstack([with_ball, ctrl, pf, shape, rev]))


# == 4. the two judges, emissions and Viterbi ======================================================

class Logistic:
    """A fitted logistic regression read from the params file: standardise, then a dot product."""

    def __init__(self, d):
        self.mu, self.sd, self.beta = np.array(d["mu"]), np.array(d["sd"]), np.array(d["beta"])

    def logit(self, Xm):
        Z = np.hstack([np.ones((len(Xm), 1)), (Xm - self.mu) / self.sd])
        return Z @ self.beta


def emissions(logit_dead, logit_home, temp_dead=1.0, temp_team=1.0):
    """log P(H), log P(A), log P(D) from the two logits."""
    ld = logit_dead * temp_dead
    lt = logit_home * temp_team
    log_d = -np.logaddexp(0, -ld)
    log_alive = -np.logaddexp(0, ld)
    log_h = -np.logaddexp(0, -lt)
    log_a = -np.logaddexp(0, lt)
    return np.stack([log_alive + log_h, log_alive + log_a, log_d], axis=1)


def viterbi(emis, switch_cost, dead_cost):
    """emis: (n, 3) log-scores for H, A, D. Returns the best path with change penalties."""
    n = len(emis)
    trans = np.array([[0.0, -switch_cost, -dead_cost],
                      [-switch_cost, 0.0, -dead_cost],
                      [-dead_cost, -dead_cost, 0.0]])
    back = np.zeros((n, 3), dtype=np.int8)
    score = emis[0].copy()
    for t in range(1, n):
        cand = score[:, None] + trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emis[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def tag_team_logit(n, home_name, away_name):
    """+1 per frame where a phase points to H, -1 where it points to A; plus kickoff frames."""
    names = {home_name.strip().lower(): H, away_name.strip().lower(): A}
    out = np.zeros(n)
    kickoffs = []
    for event in load_tags():
        code = (event["code"] or "").strip().lower()
        team = names.get((event["tags"].get("Team") or "").strip().lower())
        start, stop = max(0, event["start_frame"]), min(n, event["end_frame"])
        if code == "set pieces" and (event["tags"].get("Type") or "").strip().lower() == "kick off (start)":
            kickoffs.append(start)
        if team is None or stop <= start:
            continue
        if code in OFFENSIVE:
            sign = 1.0 if team == H else -1.0
        elif code in DEFENSIVE:
            sign = -1.0 if team == H else 1.0
        else:
            continue
        out[start:stop] += sign
    return np.clip(out, -1, 1), sorted(kickoffs)



# == 5. sequence model: a bidirectional GRU over the same features ==================================

SEQ_STEP = 5      # the GRU reads every 5th frame (5 Hz)
SEQ_HID = 64


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gru(x, W, U, b):
    """One GRU direction over x (T, d) -> (T, SEQ_HID)."""
    h = np.zeros(SEQ_HID)
    out = np.empty((len(x), SEQ_HID))
    xw = x @ W + b
    Uzr, Uh = U[:, :2 * SEQ_HID], U[:, 2 * SEQ_HID:]
    for t in range(len(x)):
        zr = _sigmoid(xw[t, :2 * SEQ_HID] + h @ Uzr)
        z, r = zr[:SEQ_HID], zr[SEQ_HID:]
        h = (1 - z) * h + z * np.tanh(xw[t, 2 * SEQ_HID:] + (r * h) @ Uh)
        out[t] = h
    return out


def seq_log_probs(P, x):
    """Two bidirectional GRU layers -> log P(H), log P(A), log P(D) per step."""
    h = np.maximum(x @ P["Win"] + P["bin"], 0.0)
    for layer in (0, 1):
        f = _gru(h, P["W%df" % layer], P["U%df" % layer], P["b%df" % layer])
        b = _gru(h[::-1], P["W%db" % layer], P["U%db" % layer], P["b%db" % layer])[::-1]
        h = np.concatenate([f, b], -1)
    lo = h @ P["Wout"] + P["bout"]
    m = lo.max(-1, keepdims=True)
    return lo - m - np.log(np.exp(lo - m).sum(-1, keepdims=True))


def seq_emissions(path, dall, tn, tb, seen, periods):
    """Per-frame log-probabilities from the sequence model, frames outside the periods left at 0.
    The file holds several networks (prefixes n0__, n1__, ...); each one runs on its own standardised
    input (training mean and spread; columns that never varied in training ignored; values clipped to
    +/-6 spreads) and their probabilities are averaged."""
    z = np.load(path)
    nets = sorted({k.split("__")[0] for k in z.files})
    s = seen.astype(np.float32)[:, None]
    X = np.hstack([dall.astype(np.float32), tn.astype(np.float32), tb.astype(np.float32) * s, s])
    prob = np.zeros((len(X), 3))
    for net in nets:
        P = {k.split("__", 1)[1]: z[k].astype(np.float64) for k in z.files
             if k.startswith(net + "__") and k.split("__", 1)[1] not in ("mu", "sd")}
        mu, sd = z[net + "__mu"].astype(np.float64), z[net + "__sd"].astype(np.float64)
        e = np.zeros((len(X), 3))
        for first, last in periods:
            idx = np.arange(first, last + 1, SEQ_STEP)
            x = np.clip((X[idx].astype(np.float64) - mu) / sd, -6.0, 6.0)
            x[:, sd < 1e-3] = 0.0
            lp = seq_log_probs(P, x.astype(np.float32).astype(np.float64))
            j = np.minimum(np.round((np.arange(first, last + 1) - first) / SEQ_STEP).astype(int), len(lp) - 1)
            e[first:last + 1] = lp[j]
        prob += np.exp(e)
    return np.log(prob / len(nets) + 1e-12)


# == 6-7. main ====================================================================================

def main(out_path=None, rebuild_cache=False):
    out_path = out_path or OUT_PATH
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN rows are expected
    if rebuild_cache or not os.path.exists(CACHE_PATH):
        build_cache()
    n = source_frames()
    data = np.load(CACHE_PATH)
    ball, pos, speed, slot_team = data["ball"], data["pos"], data["speed"], data["slot_team"]
    params = json.load(open(PARAMS_PATH))
    models = {k: Logistic(v) for k, v in params["models"].items()}
    dec = params["decode"]
    apply_params(params)

    blocks = half_video_blocks(GAME)
    periods = [(first, last) for _, first, last in blocks]
    home_name, away_name = team_names(require(["atd_meta"])[0])

    # 1-3: features
    c = common(ball, pos, speed, slot_team, periods, GAMES[GAME]["pitch"])
    gap = gap_context(c)
    dall = features_dead(c, gap)
    tn, tb = features_team(c, pos, slot_team, ball, gap)
    seen = c["seen"]
    for name, m, mat in (("dead", models["dead"], dall), ("team_ball", models["team_ball"], tb),
                         ("team_noball", models["team_noball"], tn)):
        if mat.shape[1] != len(m.mu):
            sys.exit("feature count mismatch for {}: {} columns, params expect {}".format(
                name, mat.shape[1], len(m.mu)))

    # 2-4: judges and Smart Tagging
    ld = models["dead"].logit(dall)
    lt = np.where(seen, models["team_ball"].logit(tb), models["team_noball"].logit(tn))
    tags, kickoffs = tag_team_logit(n, home_name, away_name)
    lt = lt + TAG_LOGIT * tags

    # 5-6: sequence model, averaged (in log space) with the judges
    emis = emissions(ld, lt, dec["temp_dead"], dec["temp_team"])
    emis = (emis + seq_emissions(SEQ_PATH, dall, tn, tb, seen, periods)) / 2
    print("model v11 = judges from {} + average of {} networks from {}, averaged".format(
        os.path.relpath(PARAMS_PATH, ROOT), len({k.split("__")[0] for k in np.load(SEQ_PATH).files}),
        os.path.relpath(SEQ_PATH, ROOT)))

    # 7: Viterbi per half
    codes = np.full(n, X, dtype=np.int8)
    print("decoding (switch cost {}, dead cost {})".format(dec["switch_cost"], dec["dead_cost"]))
    for first, last in periods:
        e = emis[first:last + 1].copy()
        kick = [k for k in kickoffs if first <= k <= last + 1]
        forced = (kick[0] - first) if kick else 0
        if forced > 0:
            e[:forced, H] = e[:forced, A] = -1e9
        codes[first:last + 1] = viterbi(e, dec["switch_cost"], dec["dead_cost"])
        print("  half from frame {}: {} frames forced D before the tagged kickoff".format(first, forced))

    letters = np.array(["H", "A", "D", "X"])
    write_submission(out_path, letters[codes])
    table = pm.read_label_csv(out_path)
    reason = pm.validate_submission(table, n)
    print("\nvalidation: {}".format("OK" if reason is None else "FAIL " + reason))

    # sanity aggregates only (no ground truth)
    graded = c["graded"]
    stats = pm.possession_stats(pm.label_codes(table), [(a, b + 1) for a, b in periods])
    print("\nfull match (no ground truth, sanity only)")
    print("  {} possessions, mean {:.1f} s, home share {:.1%}".format(
        stats["n_spells"], stats["mean_duration_s"], stats["home_share"]))
    print("  dead ball {:.1%} of graded frames (model's frame-level P(D) > 0.5 on {:.1%})".format(
        np.mean(codes[graded] == D), np.mean(ld[graded] > 0)))
    print("  team evidence leans home on {:.1%} of graded frames".format(np.mean(lt[graded] > 0)))
    missing = graded & ~seen
    print("  ball missing on {:.1%} of graded frames; of those, labelled D: {:.1%}".format(
        np.mean(missing[graded]), np.mean(codes[missing] == D)))
    _, touch = detect_touches(c)
    minutes = graded.sum() / FPS / 60
    sign, conf = touch_teams(c, touch)
    ball_s = smooth_ball(c["_ball"].astype(np.float64), TOUCH_SMOOTH)
    v = ball_velocity(ball_s, k=3)
    fl = flights(c, touch, sign, conf, np.hypot(v[:, 0], v[:, 1]))
    turn = sum(1 for f in fl if f[2] != f[3])
    print("  touches {:.1f} per minute (validation 13-23); flights {:.1f} per minute, {:.0%} between "
          "different teams (validation 7.5-10, ~33%)".format(
              touch[graded].sum() / minutes, len(fl) / minutes, turn / max(len(fl), 1)))
    vh = forward_velocity(pos[:, slot_team == H].astype(np.float64), c["sign_home"])
    va = forward_velocity(pos[:, slot_team == A].astype(np.float64), -c["sign_home"])
    print("  players' speed towards the attacked goal, p10 / p90: home {:.1f} / {:.1f} m/s, away {:.1f} / {:.1f} m/s "
          "(validation games: {})".format(*np.nanpercentile(vh[graded], [10, 90]),
                                          *np.nanpercentile(va[graded], [10, 90]), REVERSAL_RANGE))
    print("\nwrote {}".format(os.path.relpath(out_path, ROOT)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="output CSV (default evaluation/results/v11_standalone.csv)")
    parser.add_argument("--rebuild-cache", action="store_true", help="re-read the raw tracking even if cache/ exists")
    args = parser.parse_args()
    main(args.out and os.path.abspath(args.out), args.rebuild_cache)
