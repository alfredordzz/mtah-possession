"""The 'DNA' of a submission: the match read as a string of letters. Aggregates only; no ground truth is read.

    python diagnostics/possession_dna.py                          # evaluation/results/v13_standalone.csv
    python diagnostics/possession_dna.py evaluation/results/v10_standalone.csv
    python diagnostics/possession_dna.py evaluation/results/v11_standalone.csv --quality evaluation/results/v11_quality.csv   # the fallback

Each run of the same label becomes one letter: L (home has the ball), V (away has the ball), M (dead
ball). The script counts:
    - motifs (short patterns of letters) per 90 minutes: a change of team in play, a possession that
      comes back to the same team after less than 3 s of the rival (recuperacion), a change of team
      across a stoppage, the same team before and after a stoppage, and possessions shorter than 3 s
    - transitions: after an L, how often comes V or M; after an M, how often L or V
    - the most common three-letter words, as a share of all words
and prints them next to the same numbers on the 8 external games (diagnostics/dna_ref.json): their
true labels and what v11 predicts there, held out. Like describe_submission.py, it says what the
file answered, not whether it is right.

It ends with a quality gate: PASS, or FAIL naming the measures out of range. The bands come from
v11 on the external games at normal, slower and faster pace; on 25 kinds of deliberate damage to
the tracking it raised no false alarm on harmless damage and failed the broken pipelines (no ball),
but it does not catch moderate damage: a degraded model still writes a plausible-looking match.
With the quality file written by possession_v11_standalone.py --quality (read by default when it
exists), the gate also fails when too many possessions have a Phred Q below 5: a damaged pipeline
makes the model less sure of itself. Together they caught 22 of 63 harmful runs (all 8 without the
ball, 5 of 8 with 80% less ball, 3 of 8 played backwards) with no false alarm on 74 harmless ones.
"""

import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 25
LET = "LVM"
MOTIFS = [("cambio_en_juego", "cambio de equipo en juego (L V o V L)"),
          ("recuperacion", "recuperación: X, rival < 3 s, X (L V L o V L V)"),
          ("cambio_por_pausa", "cambio de equipo a través de una pausa (L M V)"),
          ("mismo_tras_pausa", "el mismo equipo antes y después de una pausa (L M L)"),
          ("corta", "posesiones de menos de 3 s")]


def tokens(codes, mask_codes=None):
    mask = (mask_codes if mask_codes is not None else codes) != 3
    c = np.where(mask, codes, 3).astype(int)
    ch = np.flatnonzero(np.diff(c)) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(c)]
    return [(int(a), int(b), LET[c[a]]) for a, b in zip(st, en) if c[a] != 3]


def profile(codes, mask_codes=None):
    """Motifs per 90 minutes, transition shares and three-letter word shares of one label array."""
    toks = tokens(codes, mask_codes)
    graded = ((mask_codes if mask_codes is not None else codes) != 3).sum()
    per90 = 90.0 / (graded / FPS / 60)
    m = {k: 0 for k, _ in MOTIFS}
    trans = Counter()
    words = Counter()
    for i, t in enumerate(toks):
        if t[2] != "M" and t[1] - t[0] < 3 * FPS:
            m["corta"] += 1
        if i + 1 < len(toks) and toks[i + 1][0] == t[1]:
            u = toks[i + 1]
            trans[t[2] + u[2]] += 1
            if t[2] != "M" and u[2] != "M":
                m["cambio_en_juego"] += 1
        if i + 2 < len(toks) and toks[i + 1][0] == t[1] and toks[i + 2][0] == toks[i + 1][1]:
            u, w = toks[i + 1], toks[i + 2]
            words[t[2] + u[2] + w[2]] += 1
            if t[2] != "M" and w[2] == t[2] and u[2] not in ("M", t[2]) and u[1] - u[0] < 3 * FPS:
                m["recuperacion"] += 1
            if u[2] == "M" and t[2] != "M" and w[2] != "M":
                m["cambio_por_pausa" if w[2] != t[2] else "mismo_tras_pausa"] += 1
    tr = {}
    for a in LET:
        tot = sum(v for k, v in trans.items() if k[0] == a)
        for b in LET:
            if a != b:
                tr[a + b] = trans[a + b] / tot if tot else 0.0
    nw = sum(words.values())
    live = [(t[1] - t[0]) / FPS for t in toks if t[2] != "M"]
    graded_codes = codes[(mask_codes if mask_codes is not None else codes) != 3]
    return {"motifs_per90": {k: v * per90 for k, v in m.items()}, "trans": tr,
            "words": {k: v / nw for k, v in words.items()} if nw else {}, "letters_per90": len(toks) * per90,
            "dead_share": float(np.mean(graded_codes == 2)) if len(graded_codes) else 0.0,
            "median_poss_s": float(np.median(live)) if live else 0.0}


def flat(p):
    """The scalar measures the quality gate checks, from a profile()."""
    out = {"letters_per90": p["letters_per90"], "dead_share": p["dead_share"], "median_poss_s": p["median_poss_s"]}
    out.update({"motif_" + k: v for k, v in p["motifs_per90"].items()})
    out.update({"after_" + k: v for k, v in p["trans"].items()})
    return out


def main():
    sys.path.insert(0, ROOT)
    for sub in ("starter", "evaluation"):
        sys.path.insert(0, os.path.join(ROOT, sub))
    import possession_metrics as pm  # noqa: E402
    import argparse
    ap = argparse.ArgumentParser(description="DNA of a submission file and the quality gate")
    ap.add_argument("csv", nargs="?", default=os.path.join(ROOT, "evaluation", "results", "v13_standalone.csv"))
    ap.add_argument("--quality", default=os.path.join(ROOT, "evaluation", "results", "v13_quality.csv"),
                    help="per-possession quality file from possession_v11_standalone.py --quality (used if it exists)")
    a = ap.parse_args()
    path = a.csv
    codes = pm.label_codes(pm.read_label_csv(path))
    ref = json.load(open(os.path.join(ROOT, "diagnostics", "dna_ref.json")))
    p = profile(codes)
    rng = lambda r, f: "{} [{}-{}]".format(f.format(r[1]), f.format(r[0]), f.format(r[2]))
    print("ARCHIVO: {}".format(os.path.relpath(path, ROOT)))
    print("  letras (L, V, M) por 90 min: {:.0f}   externos: verdad {}, v11 {:.0f}\n".format(
        p["letters_per90"], rng(ref["truth"]["letters_per90"], "{:.0f}"), ref["v11"]["letters_per90"][1]))
    print("1. MOTIVOS POR 90 MIN")
    print("  {:<56} {:>8}   {:>20}   {:>8}".format("", "archivo", "externos: verdad", "ext. v11"))
    for k, text in MOTIFS:
        print("  {:<56} {:>8.1f}   {:>20}   {:>8.1f}".format(text, p["motifs_per90"][k], rng(ref["truth"]["motifs_per90"][k], "{:.1f}"),
                                                          ref["v11"]["motifs_per90"][k][1]))
    print("\n2. QUÉ VIENE DESPUÉS DE CADA LETRA")
    names = {"L": "local", "V": "visitante", "M": "balón muerto"}
    for k in ("LV", "LM", "VL", "VM", "ML", "MV"):
        print("  después de {:<13} viene {:<13} {:>5.0%}   externos: verdad {}, v11 {:.0%}".format(
            names[k[0]], names[k[1]], p["trans"][k], rng(ref["truth"]["trans"][k], "{:.0%}"), ref["v11"]["trans"][k][1]))
    print("\n3. PALABRAS DE TRES LETRAS MÁS COMUNES (parte de todas)")
    for w, s in sorted(p["words"].items(), key=lambda x: -x[1])[:8]:
        t = ref["truth"]["words"].get(w, [0, 0, 0])
        print("  {}   {:>5.1%}   externos: verdad {}, v11 {:.1%}".format(w, s, rng(t, "{:.1%}"), ref["v11"]["words"].get(w, [0, 0, 0])[1]))

    g = ref["gate"]
    out = [(k, v, g["bands"][k]) for k, v in flat(p).items() if not (g["bands"][k][0] <= v <= g["bands"][k][1])]
    print("\n4. QUALITY GATE")
    for k, v, b in out:
        print("  fuera de rango: {:<22} {:.3f}   permitido {:.3f} a {:.3f}".format(k, v, b[0], b[1]))
    fail = len(out) >= g["k"]
    print("  ADN: {} de {} medidas fuera de rango (falla con {} o más)".format(len(out), len(g["bands"]), g["k"]))
    if os.path.exists(a.quality):
        qs = np.array([float(r.split(",")[4]) for r in open(a.quality).read().splitlines()[1:] if r])
        low = float(np.mean(qs < 5)) if len(qs) else 0.0
        print("  calidad: {:.0%} de las posesiones con Q < 5 (falla arriba de {:.0%}); archivo {}".format(
            low, g["low_q_max"], os.path.relpath(a.quality, ROOT)))
        fail = fail or low > g["low_q_max"]
    else:
        print("  calidad: sin archivo de calidad (corre possession_v11_standalone.py --quality para incluirla)")
    print("  {}".format("FAIL" if fail else "PASS"))


if __name__ == "__main__":
    main()
