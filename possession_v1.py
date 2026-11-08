"""Step 2 (v1): the first full possession pipeline.

    python possession_v1.py

Reads cache/atd_tactical.npz (from cache_atd.py) and the Smart Tagging, then:

  1. features      who is nearest the ball, how fast the players move, is the ball off the pitch
  2. scores        per frame, how much the evidence favours H, A and D
  3. decoding      Viterbi per half with a cost for every change of state, so a possession only
                   changes hands when the evidence for it is sustained (possession is "sticky")
  4. output        evaluation/results/v1.csv: H/A/D on graded frames, X elsewhere

Every rule is generic (distances, speeds, durations). The only game-specific inputs are the
published half boundaries and the Smart Tagging kickoff tags. The dead-ball speed threshold is
learned without labels from this match (two-cluster split of player speeds), which RULES.md allows.

Output is aggregate scores and counts only, safe to share.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_names, write_submission  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
import possession_metrics as pm  # noqa: E402

CACHE_PATH = os.path.join(ROOT, "cache", "atd_tactical.npz")
OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v1.csv")
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BASELINE = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]

# All tunable numbers in one place. Chosen by football reasoning, not by the public window.
PARAMS = {
    "ball_gap_fill": 10,       # frames: interpolate ball gaps up to 0.4 s
    "team_per_metre": 0.4,     # team evidence per metre of (away dist - home dist) to the ball
    "team_clip_m": 6.0,        # cap on that distance advantage
    "near_m": 3.0,             # full weight when the nearest player is within this of the ball
    "speed_smooth_s": 2.0,     # window for the collective player speed
    "dead_scale": 0.25,        # m/s of collective speed per unit of dead-ball evidence
    "dead_clip": 6.0,
    "out_margin_m": 1.0,       # ball this far outside the lines counts as out
    "out_bonus": 4.0,          # extra dead-ball evidence while the ball is out
    "tag_weight": 0.3,         # Smart Tagging team evidence per frame
    "switch_cost": 30.0,       # cost of an H <-> A change
    "dead_cost": 15.0,         # cost of entering or leaving D
}

OFFENSIVE = {"build up", "progression", "creating chances", "attacking transition", "long ball",
             "ball in final third", "ball in the box", "players in the box", "set pieces"}
DEFENSIVE = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}

H, A, D = 0, 1, 2


# -- features ---------------------------------------------------------------------------------

def fill_short_gaps(ball, max_gap):
    """Linearly interpolate ball gaps of at most max_gap frames that have a ball on both sides."""
    n = len(ball)
    ok = ~np.isnan(ball[:, 0])
    idx = np.flatnonzero(ok)
    out = ball.copy()
    if len(idx) < 2:
        return out, ok
    t = np.arange(n)
    for k in range(2):
        out[:, k] = np.interp(t, idx, ball[idx, k])
    prev = np.maximum.accumulate(np.where(ok, t, -1))
    nxt = np.minimum.accumulate(np.where(ok, t, n)[::-1])[::-1]
    gap = nxt - prev - 1
    fill = (~ok) & (prev >= 0) & (nxt < n) & (gap <= max_gap)
    out[~ok & ~fill] = np.nan
    return out, ok | fill


def moving_mean(x, half_window):
    """Centred moving mean that ignores NaN."""
    valid = ~np.isnan(x)
    values = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(values)))
    cc = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half_window)
    hi = np.minimum(len(x), i + half_window + 1)
    count = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cs[hi] - cs[lo]) / np.maximum(count, 1), np.nan)


def two_means_threshold(x, iters=100):
    """Unsupervised split of a 1-D sample into two clusters; returns the midpoint."""
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [20, 80])
    for _ in range(iters):
        thr = 0.5 * (lo + hi)
        a, b = x[x < thr], x[x >= thr]
        if not len(a) or not len(b):
            break
        new_lo, new_hi = a.mean(), b.mean()
        if abs(new_lo - lo) < 1e-9 and abs(new_hi - hi) < 1e-9:
            break
        lo, hi = new_lo, new_hi
    return 0.5 * (lo + hi)


def team_evidence(ball, ball_ok, pos, slot_team, p):
    """Positive favours H, negative favours A. Zero where the ball is unknown."""
    diff = pos - ball[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, slot_team == 0].min(axis=1)
    d_away = dist[:, slot_team == 1].min(axis=1)
    nearest = np.minimum(d_home, d_away)

    advantage = np.clip(d_away - d_home, -p["team_clip_m"], p["team_clip_m"])
    advantage = np.where(np.isfinite(advantage), advantage, 0.0)
    weight = np.clip((2 * p["near_m"] - nearest) / p["near_m"], 0.0, 1.0)
    weight = np.where(np.isfinite(nearest), weight, 0.0)
    out = p["team_per_metre"] * advantage * weight
    return np.where(ball_ok, out, 0.0)


def dead_evidence(speed, ball_raw, graded, p):
    """Positive favours D. From collective player speed and the ball leaving the pitch."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        collective = np.nanmedian(speed, axis=1)
    smooth = moving_mean(collective, int(p["speed_smooth_s"] * FPS / 2))

    log_speed = np.log(np.clip(smooth[graded], 0.05, None))
    threshold = float(np.exp(two_means_threshold(log_speed)))

    dead = np.clip((threshold - smooth) / p["dead_scale"], -p["dead_clip"], p["dead_clip"])
    dead = np.where(np.isnan(dead), 0.0, dead)

    m = p["out_margin_m"]
    bx, by = ball_raw[:, 0], ball_raw[:, 1]
    with np.errstate(invalid="ignore"):
        out = (bx < -m) | (bx > PITCH_L + m) | (by < -m) | (by > PITCH_W + m)
    dead = dead + np.where(out, p["out_bonus"], 0.0)
    return dead, threshold, float(out[graded].mean())


def tag_evidence(n, home_name, away_name, p):
    """Smart Tagging as soft team evidence, plus the kickoff tags."""
    names = {home_name.strip().lower(): H, away_name.strip().lower(): A}
    out = np.zeros(n)
    kickoffs = []
    used = 0
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
        out[start:stop] += sign * p["tag_weight"]
        used += 1
    return out, sorted(kickoffs), used


# -- decoding ---------------------------------------------------------------------------------

def viterbi(scores, switch_cost, dead_cost):
    """Best H/A/D path for one half given per-frame scores (n, 3) and change costs."""
    n = len(scores)
    trans = np.array([[0.0, -switch_cost, -dead_cost],
                      [-switch_cost, 0.0, -dead_cost],
                      [-dead_cost, -dead_cost, 0.0]])
    back = np.zeros((n, 3), dtype=np.int8)
    score = scores[0].copy()
    for t in range(1, n):
        cand = score[:, None] + trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# -- main -------------------------------------------------------------------------------------

def main(p=PARAMS):
    n = source_frames()
    data = np.load(CACHE_PATH)
    ball_raw, pos, speed, slot_team = data["ball"], data["pos"], data["speed"], data["slot_team"]
    meta_path = require(["atd_meta"])[0]
    home_name, away_name = team_names(meta_path)

    blocks = half_video_blocks(GAME)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    ball, ball_ok = fill_short_gaps(ball_raw, p["ball_gap_fill"])
    team = team_evidence(ball, ball_ok, pos, slot_team, p)
    dead, speed_threshold, out_share = dead_evidence(speed, ball_raw, graded, p)
    tags, kickoffs, n_tags = tag_evidence(n, home_name, away_name, p)
    team = team + tags

    scores = np.stack([-dead / 2 + team / 2, -dead / 2 - team / 2, dead / 2], axis=1)

    print("features")
    print("  ball usable after short-gap fill: {:.1f}% of graded frames".format(100 * ball_ok[graded].mean()))
    print("  dead-ball speed threshold (unsupervised): {:.2f} m/s".format(speed_threshold))
    print("  ball off the pitch: {:.1f}% of graded frames".format(100 * out_share))
    print("  Smart Tagging phases used as team evidence: {}".format(n_tags))

    labels = np.full(n, "X", dtype="<U1")
    letters = np.array(["H", "A", "D"])
    print("decoding")
    for half, first, last in blocks:
        s = scores[first:last + 1].copy()
        kick = [k for k in kickoffs if first <= k <= last + 1]
        forced = (kick[0] - first) if kick else 0
        if forced > 0:
            s[:forced, H] = s[:forced, A] = -1e9
        path = viterbi(s, p["switch_cost"], p["dead_cost"])
        labels[first:last + 1] = letters[path]
        print("  half {}: {} frames forced D before the tagged kickoff".format(half, forced))

    write_submission(OUT_PATH, labels)
    table = pm.read_label_csv(OUT_PATH)
    reason = pm.validate_submission(table, n)
    print("\nvalidation: {}".format("OK" if reason is None else "FAIL " + reason))

    codes = pm.label_codes(table)
    runs = [(first, last + 1) for _, first, last in blocks]
    stats = pm.possession_stats(codes, runs)
    d_share = float(np.mean(codes[graded] == pm.D))
    print("\nfull match (no ground truth, sanity only)")
    print("  {} possessions, mean {:.1f} s, home share {:.1%}, dead ball {:.1%} of graded frames".format(
        stats["n_spells"], stats["mean_duration_s"], stats["home_share"], d_share))

    gt = pm.label_codes(pm.read_label_csv(GT_PUBLIC))
    print("\npublic 5 minutes (pipeline check only, NOT a tuning signal)")
    for name, path in (("baseline_smooth50", BASELINE), ("v1", OUT_PATH)):
        if not os.path.isfile(path):
            continue
        r = pm.evaluate(gt, pm.label_codes(pm.read_label_csv(path)))
        print("  {:<18} spellF1 {:.4f}  macroF1 {:.4f}  possessions {:>3} (truth {})  mean {:.1f} s".format(
            name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
            r["mean_spell_duration_s"]))


if __name__ == "__main__":
    main()
