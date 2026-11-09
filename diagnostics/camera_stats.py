"""How the tactical camera moves, and whether it follows the ball. Aggregates only.

    python diagnostics/camera_stats.py

From the per-frame homography (pitch -> image), the point of the pitch at the centre of the image
(where the camera looks) and the width of pitch visible across the middle of the image (zoom). Then,
on graded frames:
    * whether the camera moves at all (spread of its aim point, its speed)
    * how far its aim point is from the tracked ball, at lags from -2 s to +2 s
    * whether its horizontal pan follows the ball's horizontal motion
    * how it behaves when players walk (likely stoppages) vs run, and while the ball is missing
Nothing here reads labels.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAMES, GAME, half_video_blocks  # noqa: E402
import load_homography as LH  # noqa: E402

W_PX, H_PX = 1920, 1080
FPS = 25


def to_pitch(Hm, u, v):
    """Image pixel -> normalised pitch coordinates, per frame (inverse homography)."""
    inv = np.linalg.inv(Hm)
    p = inv @ np.array([u, v, 1.0])
    return p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]


def main():
    L, Wd = GAMES[GAME]["pitch"]
    mats, err, valid = LH.load()
    n = len(valid)
    cache = np.load(os.path.join(ROOT, "cache", "atd_tactical.npz"))
    ball = cache["ball"].astype(np.float64)
    speed = cache["speed"].astype(np.float64)
    graded = np.zeros(n, dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True

    ok = valid & graded
    Hm = np.where(np.isfinite(mats), mats, np.eye(3))
    cx, cy = to_pitch(Hm, W_PX / 2, H_PX / 2)
    lx, ly = to_pitch(Hm, 0.0, H_PX / 2)
    rx, ry = to_pitch(Hm, float(W_PX), H_PX / 2)
    aim = np.stack([cx * L, cy * Wd], axis=1)
    aim[~ok] = np.nan
    zoom = np.hypot((rx - lx) * L, (ry - ly) * Wd)
    zoom[~ok] = np.nan
    on_pitch = ok & (aim[:, 0] > -10) & (aim[:, 0] < L + 10) & (aim[:, 1] > -20) & (aim[:, 1] < Wd + 20)

    print("graded frames {}, valid homography {:.1%}, aim point on or near the pitch {:.1%}".format(
        graded.sum(), valid[graded].mean(), on_pitch[graded].mean()))
    m = on_pitch
    print("aim point x (m): p5 / p50 / p95  {:.1f} / {:.1f} / {:.1f}   (pitch length {:.1f})".format(
        *np.nanpercentile(aim[m, 0], [5, 50, 95]), L))
    print("aim point y (m): p5 / p50 / p95  {:.1f} / {:.1f} / {:.1f}   (pitch width {:.1f})".format(
        *np.nanpercentile(aim[m, 1], [5, 50, 95]), Wd))
    print("visible width across the image middle (m): p5 / p50 / p95  {:.1f} / {:.1f} / {:.1f}".format(
        *np.nanpercentile(zoom[m], [5, 50, 95])))
    v = np.full(n, np.nan)
    v[5:-5] = np.hypot(*(aim[10:] - aim[:-10]).T) / 10 * FPS
    print("aim point speed (m/s): p50 / p90 / p99  {:.2f} / {:.2f} / {:.2f}".format(*np.nanpercentile(v[m], [50, 90, 99])))

    seen = ~np.isnan(ball[:, 0])
    print("\ndistance aim point -> tracked ball, by lag (camera at t vs ball at t - lag):")
    best = None
    for lag in (-50, -25, -12, 0, 12, 25, 50):
        b = np.full_like(ball, np.nan)
        if lag >= 0:
            b[lag:] = ball[:n - lag]
        else:
            b[:lag] = ball[-lag:]
        d = np.hypot(*(aim - b).T)
        mm = m & np.isfinite(d)
        p50 = np.median(d[mm])
        best = (p50, lag) if best is None or p50 < best[0] else best
        print("  lag {:+4d} frames ({:+.1f} s): p50 {:.1f} m, p90 {:.1f} m, x only p50 {:.1f} m".format(
            lag, lag / FPS, p50, np.percentile(d[mm], 90), np.median(np.abs(aim[mm, 0] - b[mm, 0]))))
    print("  closest at lag {:+d} frames".format(best[1]))

    bvx = np.full(n, np.nan)
    bvx[5:-5] = (ball[10:, 0] - ball[:-10, 0]) / 10 * FPS
    avx = np.full(n, np.nan)
    avx[5:-5] = (aim[10:, 0] - aim[:-10, 0]) / 10 * FPS
    mm = m & np.isfinite(bvx) & np.isfinite(avx)
    print("\ncorrelation of horizontal pan speed with ball horizontal speed (same frame): {:.2f}".format(
        np.corrcoef(avx[mm], bvx[mm])[0, 1]))
    fast = mm & (np.abs(bvx) > 8)
    print("when the ball moves fast along the pitch (>8 m/s): pan in the same direction {:.0%} of frames".format(
        np.mean(np.sign(avx[fast]) == np.sign(bvx[fast]))))

    with np.errstate(invalid="ignore"):
        med_speed = np.nanmedian(speed, axis=1)
    walk = m & (med_speed < 1.0)
    run = m & (med_speed > 2.5)
    print("\nplayers walking (median < 1 m/s): aim speed p50 {:.2f} m/s, visible width p50 {:.1f} m".format(
        np.nanmedian(v[walk]), np.nanmedian(zoom[walk])))
    print("players running (median > 2.5 m/s): aim speed p50 {:.2f} m/s, visible width p50 {:.1f} m".format(
        np.nanmedian(v[run]), np.nanmedian(zoom[run])))
    print("ball missing: aim speed p50 {:.2f} m/s; ball seen: {:.2f} m/s".format(
        np.nanmedian(v[m & ~seen]), np.nanmedian(v[m & seen])))


if __name__ == "__main__":
    main()
