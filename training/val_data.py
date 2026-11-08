"""Validation set from Metrica's open sample games (other matches, public data).

    python val_data.py <sample-data/data dir> <out dir> [extra_in_play_ball_miss gap_seconds [noisy|smooth]]

For each of the three games it writes <out>/val_game{1,2,3}.npz with:

    ball, pos, speed, slot_team, periods   degraded tracking in the competition's array format
    clean_ball                             the undegraded ball, for diagnostics only
    gt                                     per-frame H/A/D/X codes derived from the event data

Ground truth from events (human-annotated data about OTHER games, which RULES.md allows):
  * each period is D until its kick-off
  * the ball dies at BALL OUT, FAULT RECEIVED, a shot that goes out or in, an offside or a hand
    ball, and comes back at the next SET PIECE or goal-kick PASS
  * while alive, the team of the last PASS / CARRY / RECOVERY / SHOT / SET PIECE has the ball;
    BALL LOST and CHALLENGE do not move possession on their own (the next recovery does)

Degradation, to look like the automatic tracking of the competition match:
  * the ball keeps the sample data's own gaps only (v3: the competition tracking turned out to lose
    the ball in long stretches tied to stoppages, like the sample data, not in short in-play runs).
    An extra in-play dropout chain is available (ball_miss > 0, mean gap ball_gap_s, more likely
    when players crowd it); position noise on what remains
  * players drop out in runs (~7% missing), with position noise; speed is recomputed from the
    noisy positions, as a tracker would
"""

import csv
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
PITCH_L, PITCH_W = 105.0, 68.0
H, A, D, X = 0, 1, 2, 3

OWNER_TYPES = {"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE"}


# -- loading -----------------------------------------------------------------------------------

def load_csv_game(folder, n):
    """Games 1 and 2: separate home/away CSVs, ball in the home file."""
    def read(side):
        path = os.path.join(folder, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(n, side))
        with open(path) as f:
            rows = list(csv.reader(f))
        header = rows[2]
        data = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r] for r in rows[3:]])
        return header, data

    hh, hd = read("Home")
    ah, ad = read("Away")
    period = hd[:, 0].astype(int)
    frames = hd[:, 1].astype(int)

    def players(header, data):
        cols = [i for i, name in enumerate(header) if name.startswith("Player")]
        return np.stack([np.stack([data[:, c], data[:, c + 1]], axis=1) for c in cols], axis=1)

    home, away = players(hh, hd), players(ah, ad)
    bcol = hh.index("Ball")
    ball = np.stack([hd[:, bcol], hd[:, bcol + 1]], axis=1)
    pos = np.concatenate([home, away], axis=1)
    slot_team = np.array([H] * home.shape[1] + [A] * away.shape[1], dtype=np.int8)

    events = []
    with open(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(n))) as f:
        for r in csv.DictReader(f):
            events.append({"team": H if r["Team"] == "Home" else A, "type": r["Type"],
                           "sub": (r["Subtype"] or "").upper(),
                           "start": int(r["Start Frame"]), "end": int(r["End Frame"] or r["Start Frame"])})
    return frames, period, ball, pos, slot_team, events


def load_epts_game(folder):
    """Game 3: FIFA EPTS with several format specs (substitutions change the player order)."""
    meta = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    local = meta.find(".//Score").get("idLocalTeam")
    player_team = {p.get("id"): (H if p.get("teamId") == local else A) for p in meta.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    params = {p.findtext("Name"): p.findtext("Value") for p in meta.iter("ProviderParameter")}

    pids = sorted(player_team)
    index = {pid: i for i, pid in enumerate(pids)}
    slot_team = np.array([player_team[p] for p in pids], dtype=np.int8)

    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        order = []
        for ref in spec.iter("PlayerChannelRef"):
            cid = ref.get("playerChannelId")
            if cid.endswith("_x"):
                order.append(index[channel_player[cid]])
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))

    lines = open(os.path.join(folder, "Sample_Game_3_tracking.txt")).read().splitlines()
    n = len(lines)
    frames = np.zeros(n, dtype=int)
    ball = np.full((n, 2), np.nan)
    pos = np.full((n, len(pids), 2), np.nan)
    for i, line in enumerate(lines):
        f, pl, bl = line.split(":")
        f = int(f)
        frames[i] = f
        order = next(o for s, e, o in specs if s <= f <= e)
        vals = np.array(pl.replace(";", ",").split(","), dtype=float).reshape(-1, 2)
        pos[i, order] = vals
        ball[i] = np.array(bl.split(","), dtype=float)[:2]
    h1e = int(params["first_half_end"])
    period = np.where(frames <= h1e, 1, 2)

    events = []
    for e in json.load(open(os.path.join(folder, "Sample_Game_3_events.json")))["data"]:
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = " ".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        events.append({"team": H if e["team"]["id"] == local else A, "type": e["type"]["name"],
                       "sub": (sub or "").upper(), "start": int(e["start"]["frame"]),
                       "end": int(e["end"]["frame"] or e["start"]["frame"])})
    return frames, period, ball, pos, slot_team, events


# -- ground truth ------------------------------------------------------------------------------

def is_death(e):
    t, s = e["type"], e["sub"]
    if t in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return True
    if t == "BALL LOST" and ("OFFSIDE" in s or "HAND BALL" in s):
        return True
    return False


def is_restart(e):
    return e["type"] == "SET PIECE" or (e["type"] == "PASS" and "GOAL KICK" in e["sub"])


def ground_truth(frames, period, events):
    """Per-row H/A/D codes; rows are tracking frames in file order (contiguous ids)."""
    n = len(frames)
    base = int(frames[0])
    gt = np.full(n, D, dtype=np.int8)
    events = sorted(events, key=lambda e: e["start"])

    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        first, stop = int(idx[0]), int(idx[-1]) + 1          # row range [first, stop)
        evs = [e for e in events if first <= e["start"] - base < stop]
        alive, owner, cursor = False, None, first
        dead_until = -1

        def fill(upto):
            nonlocal cursor
            upto = min(max(upto, cursor), stop)
            gt[cursor:upto] = owner if alive else D
            cursor = upto

        for e in evs:
            r = e["start"] - base
            fill(r)
            if not alive:
                # events overlapping the stoppage (the action that sent the ball out) do not
                # restart play; the next restart or owning action after it does
                if e["start"] >= dead_until and (is_restart(e) or e["type"] in OWNER_TYPES):
                    alive, owner = True, e["team"]
            elif e["type"] in OWNER_TYPES:
                owner = e["team"]
            if is_death(e) and alive:
                fill(max(e["end"], e["start"]) - base)
                alive = False
                dead_until = max(e["end"], e["start"])
        fill(stop)
    return gt


def restart_list(frames, events):
    """(restarts (k, 3): row, team, kind; goals (m, 2): row, team) from the event data.

    kind: 0 throw-in, 1 free kick or corner, 2 goal kick, 3 kick-off. These are the moments the
    competition's Smart Tagging marks as SET PIECES; they feed the tag simulator used in training.
    """
    base = int(frames[0])
    rows, gl = [], []
    for e in events:
        r = e["start"] - base
        if not 0 <= r < len(frames):
            continue
        if is_restart(e):
            s = e["sub"]
            kind = 0 if "THROW" in s else 2 if "GOAL KICK" in s else 3 if "KICK OFF" in s else 1
            rows.append((r, e["team"], kind))
        if e["type"] == "SHOT" and "GOAL" in e["sub"]:
            gl.append((r, e["team"]))
    return (np.array(sorted(rows), dtype=np.int64).reshape(-1, 3),
            np.array(sorted(gl), dtype=np.int64).reshape(-1, 2))


# -- degradation -------------------------------------------------------------------------------

def markov_mask(n, miss_frac, mean_gap, rng, hazard_boost=None):
    """True where missing. Two-state chain with given stationary share and mean gap length."""
    q = 1.0 / mean_gap                      # missing -> visible
    p0 = q * miss_frac / (1 - miss_frac)   # visible -> missing
    u = rng.random(n)
    boost = np.ones(n) if hazard_boost is None else hazard_boost
    out = np.zeros(n, dtype=bool)
    missing = False
    for t in range(n):
        if missing:
            missing = u[t] >= q
        else:
            missing = u[t] < p0 * boost[t]
        out[t] = missing
    return out


def ou_offsets(n, s, sigma, tau, rng):
    """Slowly varying per-player offsets (Ornstein-Uhlenbeck): std sigma m, correlation tau frames."""
    a = np.exp(-1.0 / tau)
    out = np.zeros((n, s, 2))
    x = rng.normal(0, sigma, (s, 2))
    e = rng.normal(0, sigma * np.sqrt(1 - a * a), (n, s, 2))
    for t in range(n):
        x = a * x + e[t]
        out[t] = x
    return out


def smooth_track(ball, half):
    out = np.full_like(ball, np.nan)
    for k in range(2):
        v = ball[:, k]
        ok = ~np.isnan(v)
        cs = np.concatenate(([0.0], np.cumsum(np.where(ok, v, 0.0))))
        cc = np.concatenate(([0], np.cumsum(ok)))
        i = np.arange(len(v))
        lo, hi = np.clip(i - half, 0, len(v)), np.clip(i + half + 1, 0, len(v))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, k] = (cs[hi] - cs[lo]) / (cc[hi] - cc[lo])
    out[np.isnan(ball[:, 0])] = np.nan
    return out


# Tracker profiles. "noisy" is the original simulation. "smooth" reproduces the summary statistics
# of the competition tracking measured by touch_stats.py (aggregates of the provided input only):
# a smoothed ball track (8-frame velocity change p90 ~5.3 m/s), nearest player ~1.9 m from the
# ball (slow per-player offsets ~0.9 m) and the ball a few frames behind the players.
PROFILES = {
    "noisy": dict(ball_noise=0.3, ball_smooth=0, player_drift=0.0, drift_tau=50, ball_lag=0),
    "smooth": dict(ball_noise=0.2, ball_smooth=10, player_drift=0.9, drift_tau=50, ball_lag=4),
}


def degrade(ball, pos, rng, ball_miss=0.0, ball_gap_s=2.5, player_miss=0.07, player_gap_s=2.0,
            profile="noisy"):
    n, s, _ = pos.shape
    pr = PROFILES[profile]
    missing = np.isnan(ball[:, 0])
    ball = ball + rng.normal(0, pr["ball_noise"], ball.shape)
    if pr["ball_smooth"]:
        ball = smooth_track(ball, pr["ball_smooth"])
    if pr["ball_lag"]:
        lag = pr["ball_lag"]
        ball = np.vstack([np.full((lag, 2), np.nan), ball[:-lag]])
        missing = np.concatenate([np.ones(lag, bool), missing[:-lag]])
    ball[missing] = np.nan
    pos = pos + rng.normal(0, 0.2, pos.shape)
    if pr["player_drift"]:
        pos = pos + ou_offsets(n, s, pr["player_drift"], pr["drift_tau"], rng)

    d = np.hypot(pos[..., 0] - ball[:, None, 0], pos[..., 1] - ball[:, None, 1])
    crowd = np.nansum(d < 3.0, axis=1)
    boost = 0.6 + 0.4 * np.clip(crowd, 0, 4)          # harder to see in a crowd
    boost = boost / boost.mean()
    lost = markov_mask(n, ball_miss, ball_gap_s * FPS, rng, boost) if ball_miss > 0 else np.zeros(n, bool)
    ball = ball.copy()
    ball[lost] = np.nan

    pos = pos.copy()
    for k in range(s):
        m = markov_mask(n, player_miss, player_gap_s * FPS, rng)
        pos[m, k] = np.nan

    step = np.full((n, s), np.nan)
    step[1:] = np.hypot(np.diff(pos[..., 0], axis=0), np.diff(pos[..., 1], axis=0)) * FPS
    k = 5
    valid = ~np.isnan(step)
    cs = np.cumsum(np.where(valid, step, 0), axis=0)
    cc = np.cumsum(valid, axis=0)
    cs = np.vstack([np.zeros((1, s)), cs])
    cc = np.vstack([np.zeros((1, s)), cc])
    lo = np.clip(np.arange(n) - k, 0, n)
    hi = np.clip(np.arange(n) + k + 1, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = (cs[hi] - cs[lo]) / (cc[hi] - cc[lo])
    speed[np.isnan(pos[..., 0])] = np.nan
    return ball, pos, speed.astype(np.float32)


def main(src, out, ball_miss=0.0, ball_gap_s=2.5, profile="noisy"):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(2027)
    loaders = [("val_game1", lambda: load_csv_game(os.path.join(src, "Sample_Game_1"), 1)),
               ("val_game2", lambda: load_csv_game(os.path.join(src, "Sample_Game_2"), 2)),
               ("val_game3", lambda: load_epts_game(os.path.join(src, "Sample_Game_3")))]
    for name, load in loaders:
        frames, period, ball, pos, slot_team, events = load()
        gt = ground_truth(frames, period, events)
        restarts, goals = restart_list(frames, events)
        scale = np.array([PITCH_L, PITCH_W])
        clean_ball = ball * scale
        dball, dpos, speed = degrade(clean_ball, pos * scale, rng, ball_miss=ball_miss, ball_gap_s=ball_gap_s,
                                    profile=profile)
        periods = []
        for p in np.unique(period):
            idx = np.flatnonzero(period == p)
            periods.append((int(idx[0]), int(idx[-1])))
        np.savez_compressed(os.path.join(out, name + ".npz"), ball=dball.astype(np.float32),
                            pos=dpos.astype(np.float32), speed=speed, slot_team=slot_team,
                            periods=np.array(periods), gt=gt, clean_ball=clean_ball.astype(np.float32),
                            restarts=restarts, goals=goals)
        spells = 0
        for a, b in periods:
            g = gt[a:b + 1]
            alive = g != D
            starts = np.flatnonzero(np.diff(np.concatenate(([D], g))) != 0)
            spells += sum(1 for i in starts if g[i] != D)
        print("{}: {} frames, gt H {:.1%} A {:.1%} D {:.1%}, {} possessions (mean {:.1f} s), "
              "ball visible {:.1%} (clean {:.1%})".format(
                  name, len(gt), np.mean(gt == H), np.mean(gt == A), np.mean(gt == D), spells,
                  np.mean(gt != D) * len(gt) / FPS / max(spells, 1),
                  np.mean(~np.isnan(dball[:, 0])), np.mean(~np.isnan(clean_ball[:, 0]))))


if __name__ == "__main__":
    extra = [float(v) for v in sys.argv[3:5]]
    prof = sys.argv[5] if len(sys.argv) > 5 else "noisy"
    main(sys.argv[1], sys.argv[2], *extra, profile=prof)
