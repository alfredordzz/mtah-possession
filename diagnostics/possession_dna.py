"""The 'DNA' of a submission: the match read as a string of letters. Aggregates only; no ground truth is read.

    python diagnostics/possession_dna.py                          # evaluation/results/v11_standalone.csv
    python diagnostics/possession_dna.py evaluation/results/v10_standalone.csv

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
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evaluation", "results", "v11_standalone.csv")
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


if __name__ == "__main__":
    main()
