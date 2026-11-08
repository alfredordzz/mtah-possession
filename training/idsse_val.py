"""Build validation games from the IDSSE arrays (idsse_convert.py output).

    python idsse_val.py idsse_npz idsse_cache

Writes four variants per match, matching the Metrica validation caches:
    <out>/noisy/idsse_<ID>.npz        noisy tracker profile, ball gaps tied to stoppages
    <out>/noisy_hard/idsse_<ID>.npz   same + extra long in-play ball gaps
    <out>/smooth/...  <out>/smooth_hard/...   the competition-like tracker profile

Two things differ from the raw DFL data and are made explicit here:

1. Controlled possession. The DFL BallPossession flag flips at nearly every contact (45% of its
   possession spells last under 3 s). The challenge counts a change only when the other team gains
   *controlled* possession; while the ball is contested it stays with the last team. So an alive
   spell shorter than CONTROL_S that interrupts the previous team's possession is given back to it.
   CONTROL_S = 1.5 s gives possession counts and durations in line with the event-derived
   possessions of the Metrica games (~330 per match, ~9-11 s mean).

2. Ball gaps. TRACAB data is corrected: the ball is always there. The competition tracking loses
   it mostly during stoppages (ball_gap_stats.py: few, long gaps; missing 70% of the time when
   players walk, 11% when they run). Gaps are simulated accordingly: most of each stoppage, plus a
   small share of long in-play gaps.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import val_data  # noqa: E402

FPS = 25
CONTROL_S = 1.5
DEAD_GAP_PROB = 0.9
INPLAY_MISS = {"": 0.06, "_hard": 0.14}
INPLAY_GAP_S = 8.0


def load(path):
    z = np.load(path)
    pos = z["pos_cm"].astype(np.float64) / 100.0
    pos[pos < -300] = np.nan                      # off the pitch / not tracked (sentinel)
    ball = z["ball_cm"][:, :2].astype(np.float64) / 100.0
    ball[ball < -300] = np.nan
    periods = [tuple(int(v) for v in row) for row in z["periods"]]
    return pos, ball, z["slot_team"].astype(np.int8), z["gt"].astype(np.int8), periods


def controlled(gt, periods, min_s=CONTROL_S):
    g = gt.copy()
    for a, e in periods:
        w = g[a:e + 1]
        changed = True
        while changed:
            changed = False
            br = np.flatnonzero(np.diff(w) != 0) + 1
            b = np.concatenate(([0], br, [len(w)]))
            for i in range(1, len(b) - 1):
                k, prev = w[b[i]], w[b[i] - 1]
                if k != 2 and prev != 2 and prev != k and (b[i + 1] - b[i]) / FPS < min_s:
                    w[b[i]:b[i + 1]] = prev
                    changed = True
        g[a:e + 1] = w
    return g


def ball_gaps(gt, periods, in_play_miss, rng):
    """True where the simulated tracker loses the ball."""
    n = len(gt)
    lost = np.zeros(n, dtype=bool)
    for a, e in periods:
        w = gt[a:e + 1]
        br = np.flatnonzero(np.diff(w) != 0) + 1
        b = np.concatenate(([0], br, [len(w)]))
        for i in range(len(b) - 1):
            s, t = a + b[i], a + b[i + 1]
            if w[b[i]] == 2 and (t - s) >= 3 * FPS and rng.random() < DEAD_GAP_PROB:
                lo = s + int(rng.uniform(0, 1.5) * FPS)
                hi = t - int(rng.uniform(0, 1.0) * FPS)
                lost[lo:hi] = True
    alive = gt != 2
    extra = val_data.markov_mask(n, in_play_miss, INPLAY_GAP_S * FPS, rng)
    return lost | (extra & alive)


def main(src, out):
    files = sorted(f for f in os.listdir(src) if f.endswith(".npz"))
    for profile in ("noisy", "smooth"):
        for suffix, miss in INPLAY_MISS.items():
            folder = os.path.join(out, profile + suffix)
            os.makedirs(folder, exist_ok=True)
            rng = np.random.default_rng(2027 + len(suffix) + (0 if profile == "noisy" else 7))
            for f in files:
                pos, ball, slot_team, gt_raw, periods = load(os.path.join(src, f))
                gt = controlled(gt_raw, periods)
                lost = ball_gaps(gt, periods, miss, rng)
                clean = ball.copy()
                clean[lost] = np.nan
                dball, dpos, speed = val_data.degrade(clean, pos, rng, profile=profile)
                np.savez_compressed(os.path.join(folder, f), ball=dball.astype(np.float32),
                                    pos=dpos.astype(np.float32), speed=speed, slot_team=slot_team,
                                    periods=np.array(periods), gt=gt,
                                    clean_ball=ball.astype(np.float32),
                                    restarts=np.zeros((0, 3), np.int64), goals=np.zeros((0, 2), np.int64))
                seen = ~np.isnan(dball[:, 0])
                print("{}/{}: ball visible {:.1%} (alive {:.1%}, dead {:.1%}), D {:.1%}".format(
                    profile + suffix, f, seen.mean(), seen[gt != 2].mean(), seen[gt == 2].mean(),
                    np.mean(gt == 2)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
