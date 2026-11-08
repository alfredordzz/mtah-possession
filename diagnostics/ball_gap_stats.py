"""Automatic check: statistics of the gaps in the ball track. Prints aggregates only.

    python diagnostics/ball_gap_stats.py

Reads cache/atd_tactical.npz (from cache_atd.py). Uses only the provided tracking, no labels, and
prints only counts, shares and percentiles, so the output is safe to share.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]
L, W = GAMES[GAME]["pitch"]
BUCKETS = [(0, 0.4), (0.4, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)]


def runs(mask):
    """[(start, stop)] of True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def zone(x, y):
    if np.isnan(x):
        return "unknown"
    if x < 0 or x > L or y < 0 or y > W:
        return "off the pitch"
    if (x < 16.5 or x > L - 16.5) and 13.85 < y < W - 13.85:
        return "penalty box"
    if y < 5 or y > W - 5 or x < 5 or x > L - 5:
        return "near a line (<5 m)"
    return "open play area"


def main():
    z = np.load(os.path.join(ROOT, "cache", "atd_tactical.npz"))
    ball, pos, speed = z["ball"], z["pos"], z["speed"]
    extra = z["ball_extra"] if "ball_extra" in z else None

    print("ball gaps in graded frames (aggregates only)\n")
    all_gaps = []
    for half, first, last in half_video_blocks(GAME):
        seen = ~np.isnan(ball[first:last + 1, 0])
        gaps = [(a + first, b + first) for a, b in runs(~seen)]
        all_gaps += [(a, b, half) for a, b in gaps]
        lengths = np.array([b - a for a, b in gaps]) / FPS
        print("half {}: ball visible {:.1%}, {} gaps, median {:.1f} s, longest {:.0f} s".format(
            half, seen.mean(), len(gaps), np.median(lengths) if len(lengths) else 0,
            lengths.max() if len(lengths) else 0))

    lengths = np.array([b - a for a, b, _ in all_gaps]) / FPS
    total = lengths.sum()
    print("\ngap length distribution (both halves)")
    print("  {:<12} {:>8} {:>18}".format("length", "gaps", "share of missing time"))
    for lo, hi in BUCKETS:
        m = (lengths > lo) & (lengths <= hi)
        label = "> {:g} s".format(lo) if hi > 1e8 else "{:g}-{:g} s".format(lo, hi)
        print("  {:<12} {:>8} {:>17.1%}".format(label, int(m.sum()), lengths[m].sum() / total))
    print("  percentiles of gap length: p25 {:.1f} s, p50 {:.1f} s, p75 {:.1f} s, p95 {:.1f} s".format(
        *np.percentile(lengths, [25, 50, 75, 95])))

    # Where was the ball just before it vanished?
    print("\nwhere the ball was last seen before each gap (share of missing time)")
    by_zone = {}
    crowd = []
    for a, b, _ in all_gaps:
        if a == 0:
            continue
        x, y = ball[a - 1]
        key = zone(x, y)
        by_zone[key] = by_zone.get(key, 0) + (b - a)
        d = np.hypot(pos[a - 1, :, 0] - x, pos[a - 1, :, 1] - y)
        crowd.append((int(np.nansum(d < 3.0)), b - a))
    for key, frames in sorted(by_zone.items(), key=lambda kv: -kv[1]):
        print("  {:<20} {:>6.1%}".format(key, frames / (total * FPS)))

    print("\nplayers within 3 m of the ball just before the gap (share of missing time)")
    crowd = np.array(crowd)
    for k, label in ((0, "0 players"), (1, "1 player"), (2, "2 players"), (3, "3 or more")):
        m = crowd[:, 0] >= 3 if k == 3 else crowd[:, 0] == k
        print("  {:<12} {:>6.1%}".format(label, crowd[m, 1].sum() / crowd[:, 1].sum()))

    # Is the ball missing more when the players run or when they stand?
    graded = np.zeros(len(ball), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(speed, axis=1)
    ok = graded & np.isfinite(med)
    q1, q2 = np.percentile(med[ok], [33.3, 66.7])
    missing = np.isnan(ball[:, 0])
    print("\nball missing rate by how fast the players move (median player speed, terciles)")
    for label, m in (("slow  (< {:.1f} m/s)".format(q1), med < q1),
                     ("medium", (med >= q1) & (med < q2)),
                     ("fast  (>= {:.1f} m/s)".format(q2), med >= q2)):
        print("  {:<22} {:>6.1%}".format(label, missing[ok & m].mean()))

    if extra is not None and extra.size and extra.shape[1] > 0:
        v = extra[graded & ~missing, 0]
        v = v[np.isfinite(v)]
        if len(v):
            print("\nthird ball field (unknown meaning) where the ball is seen: "
                  "p5 {:.2f}, median {:.2f}, p95 {:.2f}, max {:.2f}".format(
                      *np.percentile(v, [5, 50, 95]), v.max()))


if __name__ == "__main__":
    main()
