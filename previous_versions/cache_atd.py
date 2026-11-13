"""Step 1: read the tactical ATD once and cache it as compact numpy arrays, in metres.

Run from the root of the competition repository (next to games.py):

    python cache_atd.py

Reads the raw file directly (numpy only, no kloppy), with the same format and the same
slot-to-team rule as starter/baseline.py:

    raw line:  "<frame_id>:<player>;<player>;...:<ball>"   each group "x,y,speed", NaN if missing
    teams:     slot order and team come from the metadata XML; the first team is home (H)

Output: cache/atd_tactical.npz  (derived match data: keep cache/ in .gitignore, never commit it)

    ball       (N, 2)    float32  metres, NaN where the ball is missing
    ball_extra (N, k)    float32  any further ball fields in the raw file (e.g. speed), as given
    pos        (N, S, 2) float32  metres, NaN where a slot is not tracked
    speed      (N, S)    float32  m/s, as given by the ATD
    slot_team  (S,)      int8     0 = home (H), 1 = away (A)

Row i is video frame i (submission row i). The script prints only PASS/FAIL checks against the
numbers published in DATA.md / FAQ.md, plus counts, so its output is safe to share.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # previous_versions/
ROOT = os.path.dirname(HERE)                          # repository root (data, cache, starter, evaluation)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_names, team_of_each_track  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "atd_tactical.npz")
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
FIELDS_PER_PLAYER = 3  # x, y, speed


def check(name, ok, detail=""):
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", name,
                               "  ({})".format(detail) if detail else ""))
    return ok


def read_raw(raw_path, n_frames, n_slots):
    """Parse every line into fixed-size arrays indexed by video frame (frame_id - 1)."""
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


def main():
    t0 = time.time()
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n = source_frames()
    slot_team = team_of_each_track(meta_path)
    s = len(slot_team)
    home, away = team_names(meta_path)

    print("teams: H = {}, A = {}".format(home, away))
    print("slots: {} total, {} home, {} away".format(s, int((slot_team == 0).sum()),
                                                     int((slot_team == 1).sum())))
    print("reading raw tracking (a minute or two)...")
    players, ball_raw, seen, bad_lines = read_raw(raw_path, n, s)
    print("  done in {:.0f}s, {} ball fields per frame".format(time.time() - t0, ball_raw.shape[1]))

    print("\nalignment")
    check("every video frame has exactly one record", bool(seen.all()),
          "{} of {} frames".format(int(seen.sum()), n))
    check("no malformed lines", bad_lines == 0, "{} bad".format(bad_lines))

    scale = np.array([PITCH_L, PITCH_W], dtype=np.float32)
    pos = players[:, :, :2] * scale
    speed = players[:, :, 2].copy()
    ball = ball_raw[:, :2] * scale
    ball_extra = ball_raw[:, 2:]

    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    ball_ok = ~np.isnan(ball).any(axis=1)
    tracked = (~np.isnan(pos).any(axis=2)).sum(axis=1)

    print("\nsanity checks against the published numbers")
    pct_all = 100 * ball_ok.mean()
    pct_graded = 100 * ball_ok[graded].mean()
    check("ball present on ~56.9% of all frames", abs(pct_all - 56.9) < 0.5, "{:.1f}%".format(pct_all))
    check("ball present on ~59.3% of graded frames", abs(pct_graded - 59.3) < 0.5,
          "{:.1f}%".format(pct_graded))
    mean_tracked = tracked[graded].mean()
    check("~20.5 tracked objects per graded frame", abs(mean_tracked - 20.5) < 0.5,
          "{:.1f}".format(mean_tracked))
    check("graded frame count is 158,857", int(graded.sum()) == 158857, str(int(graded.sum())))

    inside = ((ball[ball_ok, 0] > -5) & (ball[ball_ok, 0] < PITCH_L + 5)
              & (ball[ball_ok, 1] > -5) & (ball[ball_ok, 1] < PITCH_W + 5)).mean()
    check("ball coordinates look like metres", inside > 0.98, "{:.1%} inside padded pitch".format(inside))

    sp = speed[graded][~np.isnan(speed[graded])]
    check("player speeds look like m/s", 0.5 < np.median(sp) < 4 and np.percentile(sp, 99.9) < 12,
          "median {:.1f}, p99.9 {:.1f}".format(np.median(sp), np.percentile(sp, 99.9)))

    centre = np.array([PITCH_L / 2, PITCH_W / 2], dtype=np.float32)
    for label, frame in (("1st half kickoff", 11185), ("2nd half kickoff", 113742)):
        window = slice(frame - 25, frame + 1)
        visible = ball_ok[window]
        if visible.any():
            dist = np.min(np.linalg.norm(ball[window][visible] - centre, axis=1))
            check("{}: ball within 3 m of centre spot".format(label), dist < 3.0, "{:.1f} m".format(dist))
        else:
            check("{}: ball visible in the second before".format(label), False, "ball missing")

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(OUT_PATH, ball=ball, ball_extra=ball_extra, pos=pos, speed=speed,
                        slot_team=slot_team)
    print("\nsaved {} ({:.0f} MB) in {:.0f}s total".format(
        os.path.relpath(OUT_PATH, ROOT), os.path.getsize(OUT_PATH) / 1e6, time.time() - t0))
    print("reminder: add cache/ to .gitignore")


if __name__ == "__main__":
    main()
