"""Convert the IDSSE open dataset (7 Bundesliga matches, DFL, CC BY 4.0) to compact arrays.

    python -m pip install kloppy
    python idsse_convert.py C:\\BarcaHub\\idsse C:\\BarcaHub\\idsse_npz

Source: Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer",
Scientific Data (2025), https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0).

For every match it writes <out>/idsse_<MATCHID>.npz with:
    pos_cm    (N, S, 2) int16   player positions in cm, origin at a pitch corner; -32768 = missing
    ball_cm   (N, 3)    int16   ball x, y, z in cm
    slot_team (S,)      int8    0 home, 1 away
    gt        (N,)      int8    0 home possession, 1 away possession, 2 dead ball (from the DFL
                                tracking's own BallPossession / BallStatus flags)
    periods   (P, 2)    int64   first and last row of each half
    pitch     (2,)      float32 pitch length and width in metres

This is public data about OTHER matches, so the output files can be shared and used for training.
"""

import glob
import os
import sys
import time

import numpy as np

MATCH_IDS = ("J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9")
MISSING = -32768


def find_files(root, match_id):
    files = [f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if os.path.isfile(f) and match_id in os.path.basename(f)]
    low = {f: os.path.basename(f).lower() for f in files}
    meta = [f for f in files if "matchinformation" in low[f] or "meta" in low[f]]
    pos = [f for f in files if "position" in low[f] or "tracking" in low[f]]
    return files, meta, pos


def to_cm(v):
    out = np.where(np.isnan(v), MISSING, np.round(v * 100))
    return np.clip(out, -32767, 32767).astype(np.int16)


def convert(meta_path, pos_path, out_path):
    from kloppy import sportec
    from kloppy.domain import BallState

    t0 = time.time()
    ds = sportec.load_tracking(meta_data=meta_path, raw_data=pos_path, only_alive=False)
    md = ds.metadata
    home, away = md.teams[0], md.teams[1]
    players = [(p, 0) for p in home.players] + [(p, 1) for p in away.players]
    slot = {p.player_id: i for i, (p, _) in enumerate(players)}
    slot_team = np.array([t for _, t in players], dtype=np.int8)

    xdim, ydim = md.pitch_dimensions.x_dim, md.pitch_dimensions.y_dim
    length = float(md.pitch_dimensions.pitch_length or 105.0)
    width = float(md.pitch_dimensions.pitch_width or 68.0)

    def mx(x):
        return (x - xdim.min) / (xdim.max - xdim.min) * length

    def my(y):
        return (y - ydim.min) / (ydim.max - ydim.min) * width

    n = len(ds.records)
    pos = np.full((n, len(players), 2), np.nan, dtype=np.float64)
    ball = np.full((n, 3), np.nan, dtype=np.float64)
    gt = np.full(n, 2, dtype=np.int8)
    period_ids = np.zeros(n, dtype=np.int64)
    last_owner = 0
    for i, fr in enumerate(ds.records):
        period_ids[i] = fr.period.id
        bc = fr.ball_coordinates
        if bc is not None and bc.x is not None:
            ball[i, 0], ball[i, 1] = mx(bc.x), my(bc.y)
            ball[i, 2] = getattr(bc, "z", np.nan) or 0.0
        for player, pdata in fr.players_data.items():
            k = slot.get(player.player_id)
            c = pdata.coordinates
            if k is not None and c is not None and c.x is not None:
                pos[i, k, 0], pos[i, k, 1] = mx(c.x), my(c.y)
        if fr.ball_state == BallState.DEAD:
            gt[i] = 2
        else:
            team = fr.ball_owning_team
            if team is not None:
                last_owner = 0 if team.team_id == home.team_id else 1
            gt[i] = last_owner

    periods = []
    for p in np.unique(period_ids):
        rows = np.flatnonzero(period_ids == p)
        periods.append((int(rows[0]), int(rows[-1])))

    np.savez_compressed(out_path, pos_cm=to_cm(pos), ball_cm=to_cm(ball), slot_team=slot_team,
                        gt=gt, periods=np.array(periods), pitch=np.array([length, width], np.float32))
    seen = ~np.isnan(ball[:, 0])
    print("  {} frames, {} player slots, pitch {:.1f} x {:.1f} m, ball present {:.1%}, "
          "H {:.1%} A {:.1%} D {:.1%}, {:.0f}s, {:.1f} MB".format(
              n, len(players), length, width, seen.mean(), np.mean(gt == 0), np.mean(gt == 1),
              np.mean(gt == 2), time.time() - t0, os.path.getsize(out_path) / 1e6))


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for mid in MATCH_IDS:
        files, meta, pos = find_files(root, mid)
        print("{}: {} files found".format(mid, len(files)))
        if not meta or not pos:
            print("  could not tell metadata and positions apart; files:")
            for f in files:
                print("   ", os.path.relpath(f, root))
            continue
        print("  metadata:  {}\n  positions: {}".format(os.path.basename(meta[0]), os.path.basename(pos[0])))
        try:
            convert(meta[0], pos[0], os.path.join(out, "idsse_{}.npz".format(mid)))
        except Exception as error:  # report and carry on with the next match
            print("  FAILED: {}: {}".format(type(error).__name__, error))


if __name__ == "__main__":
    main()
