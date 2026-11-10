"""How far the competition match's features fall outside the range seen in training. Aggregates only.

    python diagnostics/feature_shift.py

Computes v9.1's per-frame features for the competition match (the same functions as
possession_v9_standalone.py) and compares them with their mean and spread over the external training
games (diagnostics/feature_ref_v91.json, built from external data only). A model with non-linear
inputs, like the sequence model under test, can misbehave on values far outside that range.
Prints counts and shares only; no labels are read.

Reference values (same measure): an external validation game 0.06% of frame-columns beyond +/-6 sd,
4 columns with more than 1% beyond; a SkillCorner game 0.05% and 2 columns.
"""

import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import possession_v9_standalone as S  # noqa: E402

NAMES = ([f"pausa_{i}" for i in range(34)] + [f"equipo_sin_balon_{i}" for i in range(66)]
         + [f"equipo_con_balon_{i}" for i in range(67)] + ["balon_visto"])


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ref = json.load(open(os.path.join(ROOT, "diagnostics", "feature_ref_v91.json")))
    mu, sd = np.array(ref["mu"]), np.array(ref["sd"])
    data = np.load(S.CACHE_PATH)
    ball, pos, speed, slot_team = data["ball"], data["pos"], data["speed"], data["slot_team"]
    S.apply_params(json.load(open(S.PARAMS_PATH)))
    periods = [(a, b) for _, a, b in S.half_video_blocks(S.GAME)]
    c = S.common(ball, pos, speed, slot_team, periods, S.GAMES[S.GAME]["pitch"])
    gap = S.gap_context(c)
    dall = S.features_dead(c, gap)
    tn, tb = S.features_team(c, pos, slot_team, ball, gap)
    seen = c["seen"].astype(np.float32)[:, None]
    X = np.hstack([dall, tn, tb * seen, seen])[c["graded"]][::5]
    z = (X - mu) / sd
    const = sd < 1e-3
    out = np.abs(z) > 6
    out[:, const] = False
    share = out.mean(0)
    print("frames calificados (cada 5): {}".format(len(X)))
    print("frame-columnas fuera de +/-6 desviaciones: {:.3%}   (validacion 0.06%, SkillCorner 0.05%)".format(out[:, ~const].mean()))
    print("columnas con mas de 1% de frames fuera: {}   (validacion 4, SkillCorner 2)".format(int(np.sum(share > 0.01))))
    print("|z| medio: {:.2f}   (validacion 0.66, SkillCorner 0.66)".format(np.abs(np.where(const, 0, z)).mean()))
    for i in np.argsort(-share)[:8]:
        if share[i] > 0:
            print("   {:<22} {:.2%} de frames fuera de +/-6".format(NAMES[i], share[i]))
    for i in np.flatnonzero(const):
        print("   {:<22} constante en el entrenamiento; en este partido distinto de cero en {:.1%} de frames".format(
            NAMES[i], np.mean(np.abs(X[:, i] - mu[i]) > 1e-6)))


if __name__ == "__main__":
    main()
