"""Descriptive statistics of a submission file. Aggregates only; no ground truth is read.

    python diagnostics/describe_submission.py                          # evaluation/results/v12_standalone.csv
    python diagnostics/describe_submission.py evaluation/results/v9_1.csv

Prints what the file says happened in the match: how many possessions, how long, how much dead
ball, how possession changes hands, and how all of that moves through the match. Each line is shown
next to the same measure on the 8 external games (diagnostics/describe_ref.json): their true labels
and what v12 predicts there, held out (v10 and v11 are kept in the reference file too). That is the only yardstick: these numbers say what the model
answered, not whether it is right.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from games import GAME, half_video_blocks  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def segments(c):
    ch = np.flatnonzero(np.diff(c)) + 1
    return [(len(s), int(c[s[0]])) for s in np.split(np.arange(len(c)), ch)] if len(c) else []


def describe(halves):
    """halves: list of code arrays (graded part of each half, X removed)."""
    dur, dead_dur, durs_team = [], [], {0: [], 1: []}
    n = {"home": 0, "away": 0, "dead": 0, "graded": 0, "direct": 0, "via_dead": 0, "kept": 0}
    for c in halves:
        n["graded"] += len(c)
        n["home"] += int((c == 0).sum()); n["away"] += int((c == 1).sum()); n["dead"] += int((c == 2).sum())
        segs = segments(c)
        for L, lab in segs:
            if lab in (0, 1):
                dur.append(L / FPS); durs_team[lab].append(L / FPS)
            else:
                dead_dur.append(L / FPS)
        labs = [lab for _, lab in segs]
        for i in range(1, len(labs)):
            if labs[i] in (0, 1) and labs[i - 1] in (0, 1) and labs[i] != labs[i - 1]:
                n["direct"] += 1
        for i in range(2, len(labs)):
            if labs[i - 1] == 2 and labs[i] in (0, 1) and labs[i - 2] in (0, 1):
                n["via_dead" if labs[i] != labs[i - 2] else "kept"] += 1
    mins = n["graded"] / FPS / 60
    d, dd = np.array(dur), np.array(dead_dur)
    return {"minutes": mins, "n_poss": len(d), "n_home": len(durs_team[0]), "n_away": len(durs_team[1]),
            "poss_per_90": len(d) / mins * 90, "dur_mean": d.mean(), "dur_p10": np.percentile(d, 10),
            "dur_med": np.median(d), "dur_p90": np.percentile(d, 90), "dur_max": d.max(),
            "short_lt3": np.mean(d < 3), "long_ge30": np.mean(d >= 30),
            "dead_share": n["dead"] / n["graded"], "home_share_alive": n["home"] / max(1, n["home"] + n["away"]),
            "stops_per_90": len(dd) / mins * 90, "stop_med": np.median(dd),
            "direct_per_90": n["direct"] / mins * 90, "via_dead_change_per_90": n["via_dead"] / mins * 90,
            "via_dead_same_per_90": n["kept"] / mins * 90,
            "dur_home_med": np.median(durs_team[0]) if durs_team[0] else 0,
            "dur_away_med": np.median(durs_team[1]) if durs_team[1] else 0, "hist": d}


LINES = [
    ("poss_per_90", "posesiones por 90 min", "{:.0f}"),
    ("dur_mean", "duración media (s)", "{:.1f}"),
    ("dur_p10", "duración p10 (s)", "{:.1f}"),
    ("dur_med", "duración mediana (s)", "{:.1f}"),
    ("dur_p90", "duración p90 (s)", "{:.1f}"),
    ("short_lt3", "posesiones de menos de 3 s", "{:.0%}"),
    ("long_ge30", "posesiones de 30 s o más", "{:.0%}"),
    ("dead_share", "balón muerto (parte del tiempo)", "{:.0%}"),
    ("home_share_alive", "local, del tiempo con balón vivo", "{:.0%}"),
    ("stops_per_90", "pausas por 90 min", "{:.0f}"),
    ("stop_med", "duración mediana de una pausa (s)", "{:.1f}"),
    ("direct_per_90", "cambios de equipo en juego, por 90", "{:.0f}"),
    ("via_dead_change_per_90", "cambios de equipo tras una pausa, por 90", "{:.0f}"),
    ("via_dead_same_per_90", "mismo equipo antes y después de pausa, por 90", "{:.0f}"),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evaluation", "results", "v12_standalone.csv")
    codes = pm.label_codes(pm.read_label_csv(path))
    ref = json.load(open(os.path.join(ROOT, "diagnostics", "describe_ref.json")))
    blocks = half_video_blocks(GAME)
    halves = []
    for _, a, b in blocks:
        c = codes[a:b + 1]
        halves.append(c[c != CODE["X"]])

    print("ARCHIVO: {}".format(os.path.relpath(path, ROOT)))
    print("  {} filas; H {:,}  A {:,}  D {:,}  X {:,}".format(len(codes), *[int((codes == k).sum()) for k in range(4)]))
    s = describe(halves)
    print("  {:.1f} minutos dentro de las dos mitades; {} posesiones ({} local, {} visitante); la más larga dura {:.0f} s".format(
        s["minutes"], s["n_poss"], s["n_home"], s["n_away"], s["dur_max"]))
    print("  duración mediana: local {:.1f} s, visitante {:.1f} s\n".format(s["dur_home_med"], s["dur_away_med"]))

    print("1. EL PARTIDO COMPLETO, CONTRA LOS 8 PARTIDOS EXTERNOS")
    print("  {:<46} {:>8}   {:>20}   {:>10}".format("", "este", "externos: verdad", "externos: v12"))
    print("  {:<46} {:>8}   {:>20}   {:>10}".format("", "archivo", "mediana [mín-máx]", "mediana"))
    for key, text, fmt in LINES:
        t, v = ref["truth"][key], ref["v12"][key]
        rng = "{} [{}-{}]".format(fmt.format(t[1]), fmt.format(t[0]), fmt.format(t[2]))
        print("  {:<46} {:>8}   {:>20}   {:>10}".format(text, fmt.format(s[key]), rng, fmt.format(v[1])))

    print("\n2. POR MITAD")
    for (name, _, _), c in zip(blocks, halves):
        h = describe([c])
        print("  {:<14} {:5.1f} min   {:3d} posesiones   mediana {:4.1f} s   muerto {:3.0%}   local {:3.0%}".format(
            str(name), h["minutes"], h["n_poss"], h["dur_med"], h["dead_share"], h["home_share_alive"]))

    print("\n3. CADA 15 MINUTOS DE JUEGO (desde el inicio de cada mitad)")
    for (name, _, _), c in zip(blocks, halves):
        step = 15 * 60 * FPS
        for k in range(0, len(c), step):
            part = c[k:k + step]
            if len(part) < 5 * 60 * FPS:
                continue
            h = describe([part])
            print("  {:<14} min {:>2}-{:<3} {:3d} posesiones   mediana {:4.1f} s   muerto {:3.0%}   local {:3.0%}".format(
                str(name), k // (60 * FPS), (k + len(part)) // (60 * FPS), h["n_poss"], h["dur_med"], h["dead_share"],
                h["home_share_alive"]))

    print("\n4. CUÁNTO DURAN LAS POSESIONES")
    edges = [0, 1, 3, 6, 12, 25, 60, 1e9]
    names = ["< 1 s", "1-3 s", "3-6 s", "6-12 s", "12-25 s", "25-60 s", "60 s o más"]
    cnt, _ = np.histogram(s["hist"], bins=edges)
    for nm, k in zip(names, cnt):
        print("  {:<12} {:4d}  {:5.1%}  {}".format(nm, k, k / max(1, cnt.sum()), "#" * int(round(60 * k / max(1, cnt.max())))))


if __name__ == "__main__":
    main()
