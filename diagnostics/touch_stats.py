"""Automatic check: why does the touch detector fire less on this match? Aggregates only.

    python diagnostics/touch_stats.py                 # the competition match (cache/atd_tactical.npz)
    python diagnostics/touch_stats.py some_game.npz   # any array file with ball/pos/speed/slot_team/periods

Prints percentiles and counts only (ball speed and velocity change while the ball is seen, how far
the nearest player is, how many touches each threshold would give), so the output is safe to share.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
import model_v2 as M2  # noqa: E402
import model_v4 as M4  # noqa: E402


def load(path):
    z = np.load(path)
    if "periods" in z:
        periods = [tuple(int(v) for v in row) for row in z["periods"]]
        pitch = (105.0, 68.0)
    else:
        from games import GAME, GAMES, half_video_blocks
        periods = [(a, b) for _, a, b in half_video_blocks(GAME)]
        pitch = GAMES[GAME]["pitch"]
    return z["ball"], z["pos"], z["speed"], z["slot_team"], periods, pitch


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "cache", "atd_tactical.npz")
    ball, pos, speed, slot_team, periods, pitch = load(path)
    c = M2.common(ball, pos, speed, slot_team, periods, pitch)
    g, seen = c["graded"], c["seen"]
    minutes = g.sum() / M4.FPS / 60

    raw = ball.astype(np.float64)
    sm = M4.smooth_ball(raw, M4.TOUCH_SMOOTH)
    v_raw = M4.ball_velocity(raw, k=2)
    v_sm = M4.ball_velocity(sm, k=3)
    sp_raw = np.hypot(v_raw[:, 0], v_raw[:, 1])
    sp_sm = np.hypot(v_sm[:, 0], v_sm[:, 1])
    n = len(ball)
    dv = np.full(n, np.nan)
    k = 4
    dv[k:-k] = np.hypot(v_sm[2 * k:, 0] - v_sm[:-2 * k, 0], v_sm[2 * k:, 1] - v_sm[:-2 * k, 1])
    step = np.full(n, np.nan)
    step[1:] = np.hypot(np.diff(raw[:, 0]), np.diff(raw[:, 1]))
    near = np.minimum(c["d_home"], c["d_away"])

    m = g & seen
    pct = lambda x: " / ".join("{:.2f}".format(v) for v in np.nanpercentile(x[m], [10, 50, 90, 99]))
    print("ball seen on {:.1%} of graded frames ({:.0f} graded minutes)".format(seen[g].mean(), minutes))
    print("percentiles p10 / p50 / p90 / p99 over graded frames with the ball seen:")
    print("  frame-to-frame ball step (m)         ", pct(step))
    print("  ball speed, raw (m/s)                ", pct(sp_raw))
    print("  ball speed, smoothed (m/s)           ", pct(sp_sm))
    print("  velocity change over 8 frames (m/s)  ", pct(dv))
    print("  nearest player to the ball (m)       ", pct(near))
    exact_repeat = np.mean((step[m] == 0))
    print("  share of seen frames where the ball does not move at all: {:.1%}".format(exact_repeat))

    print("\ncandidate touches per graded minute for other thresholds (velocity change, reach):")
    for thr in (2.0, 3.0, 5.0):
        row = []
        for reach in (2.0, 3.0, 5.0):
            cand = np.nan_to_num((dv > thr) & (near < reach)).astype(bool) & g
            e = np.diff(np.concatenate(([0], cand.astype(np.int8), [0])))
            row.append("{:5.1f}".format((e == 1).sum() / minutes))
        print("  dv > {:.0f} m/s   reach 2 / 3 / 5 m: {}".format(thr, " / ".join(row)))
    big = np.nan_to_num(dv > 5.0).astype(bool) & g
    print("\nwhere sharp velocity changes happen (dv > 5 m/s): nearest player p50 {:.1f} m, "
          "share with a player within 2 m {:.1%}".format(np.nanmedian(near[big]), np.mean(near[big] < 2.0)))
    _, touch = M4.detect_touches(c)
    print("detector as used by v4: {:.1f} touches per graded minute".format(touch[g].sum() / minutes))


if __name__ == "__main__":
    main()
