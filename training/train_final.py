"""Train the final v4 model on external games only and write model_v4_params.json.

    python training/train_final.py VALIDATION_ROOT [--out model_v4_params.json]

VALIDATION_ROOT is the folder written by training/build_validation.py:
    VALIDATION_ROOT/metrica/{noisy,noisy_hard,smooth,smooth_hard}/val_game{1,2,3}.npz
    VALIDATION_ROOT/idsse/{noisy,noisy_hard,smooth,smooth_hard}/idsse_<MATCH>.npz

Protocol
    1. Leave-one-game-out over the 8 external games: fit on the two "hard" tracker variants of the
       other 7 games, score the held-out game on all four variants with the official metric
       (evaluation/possession_metrics.py), and pick the Viterbi costs with the best mean spellF1.
       The baseline (nearest player + majority vote over +/-50 frames) is scored alongside.
    2. Fit the final regressions on the hard variants of all 8 games and write them, with the
       chosen costs, to the parameter file that possession_v4.py applies to the competition match.

No data from the competition match and none of its ground truth is read here.
"""

import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import model_v2 as M2  # noqa: E402
import model_v4 as M4  # noqa: E402
import possession_metrics as pm  # noqa: E402

VARIANTS = ("noisy", "noisy_hard", "smooth", "smooth_hard")
TRAIN = ("noisy_hard", "smooth_hard")
SWITCH = (5.0, 8.0, 12.0)
DEAD = (25.0, 35.0, 50.0)
PITCH = (105.0, 68.0)
STRIDE = 5          # fit on every 5th frame: neighbouring frames are nearly identical
L2 = 1.0


def load_npz(path):
    z = np.load(path)
    gt = np.full(len(z["gt"]), pm.X, dtype=np.int8)
    periods = [tuple(int(v) for v in row) for row in z["periods"]]
    for a, b in periods:
        gt[a:b + 1] = z["gt"][a:b + 1]
    return dict(ball=z["ball"], pos=z["pos"], speed=z["speed"], slot_team=z["slot_team"],
                periods=periods, gt=gt)


def game_paths(root):
    """{variant: [(name, path)]} with Metrica games first, then IDSSE sorted by match id."""
    out = {}
    for v in VARIANTS:
        rows = [("metrica_{}".format(k), os.path.join(root, "metrica", v, "val_game{}.npz".format(k)))
                for k in (1, 2, 3)]
        folder = os.path.join(root, "idsse", v)
        rows += [(f[:-4], os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if f.endswith(".npz")]
        out[v] = rows
    return out


def prepare(g):
    """Features and labels for one game, kept small (float32, only what fitting needs)."""
    c = M2.common(g["ball"], g["pos"], g["speed"], g["slot_team"], g["periods"], PITCH)
    gap = M4.gap_context(c)
    dall = M4.features_dead(c, gap)
    tn, tb = M4.features_team(c, g["pos"], g["slot_team"], g["ball"], gap)
    return dict(gt=g["gt"], periods=g["periods"], seen=c["seen"],
                dall=dall.astype(np.float32), tn=tn.astype(np.float32), tb=tb.astype(np.float32))


def fit(preps):
    def stack(key, mask_fn, label_fn):
        X, y = [], []
        for p in preps:
            idx = np.flatnonzero(mask_fn(p))[::STRIDE]
            X.append(p[key][idx])
            y.append(label_fn(p)[idx])
        return np.vstack(X), np.concatenate(y)

    graded = lambda p: p["gt"] != pm.X
    alive = lambda p: (p["gt"] == pm.H) | (p["gt"] == pm.A)
    alive_seen = lambda p: alive(p) & p["seen"]
    is_dead = lambda p: (p["gt"] == pm.D).astype(float)
    is_home = lambda p: (p["gt"] == pm.H).astype(float)
    return {"dead": M2.Logistic(L2).fit(*stack("dall", graded, is_dead)),
            "team_noball": M2.Logistic(L2).fit(*stack("tn", alive, is_home)),
            "team_ball": M2.Logistic(L2).fit(*stack("tb", alive_seen, is_home))}


def logits(m, p):
    ld = m["dead"].logit(p["dall"])
    lt = np.where(p["seen"], m["team_ball"].logit(p["tb"]), m["team_noball"].logit(p["tn"]))
    return ld, lt


def decode(p, ld, lt, switch_cost, dead_cost):
    emis = M2.emissions(ld, lt)
    codes = np.full(len(ld), pm.X, dtype=np.int8)
    for first, last in p["periods"]:
        codes[first:last + 1] = M2.viterbi(emis[first:last + 1], switch_cost, dead_cost)
    return codes


def baseline(g, window=50):
    """The repository's reference: nearest tracked player to the ball, majority vote over +/-window."""
    c = M2.common(g["ball"], g["pos"], g["speed"], g["slot_team"], g["periods"], PITCH)
    seen = ~np.isnan(g["ball"][:, 0])
    codes = np.where(~seen, -1, np.where(c["d_home"] <= c["d_away"], 0, 1))
    n = len(codes)
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(n)
    lo, hi = np.maximum(0, i - window), np.minimum(n, i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    lab = np.where(h + a == 0, 2, np.where(h >= a, 0, 1)).astype(np.int8)
    lab[g["gt"] == pm.X] = pm.X
    return lab


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("validation_root")
    parser.add_argument("--out", default=os.path.join(ROOT, "model_v4_params.json"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()

    paths = game_paths(args.validation_root)
    names = [n for n, _ in paths["noisy"]]
    print("{} external games: {}".format(len(names), ", ".join(names)), flush=True)

    preps, base = {v: {} for v in VARIANTS}, {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for n, path in paths[v]:
            g = load_npz(path)
            base[v][n] = pm.evaluate(g["gt"], baseline(g))["spell_f1_mean"]
            preps[v][n] = prepare(g)
    print("features ready ({:.0f}s)".format(time.time() - t0), flush=True)

    held = {v: {} for v in VARIANTS}
    for test in names:
        m = fit([preps[v][n] for v in TRAIN for n in names if n != test])
        for v in VARIANTS:
            held[v][test] = logits(m, preps[v][test])
    best = None
    for sc, dc in itertools.product(SWITCH, DEAD):
        per = {v: {n: pm.evaluate(preps[v][n]["gt"], decode(preps[v][n], *held[v][n], sc, dc))["spell_f1_mean"]
                   for n in names} for v in VARIANTS}
        mean = np.mean([s for v in per for s in per[v].values()])
        if best is None or mean > best[0]:
            best = (mean, sc, dc, per)
    _, sc, dc, per = best

    print("\nheld-out spellF1 (mean over games)      " + "  ".join("{:>11}".format(v) for v in VARIANTS))
    print("  baseline (+/-50 frames)               " + "  ".join("{:>11.4f}".format(np.mean(list(base[v].values()))) for v in VARIANTS))
    print("  v4 (switch {:g}, dead {:g})              ".format(sc, dc) +
          "  ".join("{:>11.4f}".format(np.mean(list(per[v].values()))) for v in VARIANTS))

    m = fit([preps[v][n] for v in TRAIN for n in names])
    payload = {"version": "v4",
               "decode": {"switch_cost": sc, "dead_cost": dc, "temp_dead": 1.0, "temp_team": 1.0},
               "touch": {"dv": M4.TOUCH_DV, "reach": M4.TOUCH_REACH, "smooth": M4.TOUCH_SMOOTH,
                         "control_reach": M4.CONTROL_REACH},
               "validation_spell_f1": {v: float(np.mean(list(per[v].values()))) for v in VARIANTS},
               "models": {k: v.to_dict() for k, v in m.items()},
               "trained_on": names, "train_sets": list(TRAIN),
               "sources": ["Metrica Sports open sample data (games 1-3)",
                           "IDSSE, Bassek et al. 2025, CC BY 4.0 (5 matches)"]}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print("\nwrote {} ({:.0f}s)".format(args.out, time.time() - t0))


if __name__ == "__main__":
    main()
