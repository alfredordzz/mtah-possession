"""Are the sequence model's two key clue groups in normal ranges on the competition match? Aggregates only.

    python diagnostics/key_clues_check.py

The stress tests on the 8 external games showed that the network relies above all on two groups of
clues: the collective "about-turn" of the teams (media vuelta) and the ball's position and speed
(pistas del balón). Hiding either from the network drops v10 below v9.1. This script checks, on the
competition match, that both groups (and team shape, for comparison) look like they did in the
external games the model was trained on.

Two checks per group, both compared with the range over 8 external games x 4 tracking variants
(diagnostics/key_clues_ref.json, built from external data only):
    1. how the values look: share beyond +/-3 and +/-6 training spreads, mean distance from the
       training average, largest shift and spread of a single column
    2. how much the network relies on them here: share of graded frames whose v10 label changes when
       the group is hidden from the network (the judges still see everything)

Uses the same functions and weights as possession_v10_standalone.py. Reads no labels and prints
counts and shares only. Takes a few minutes (the network runs four times).
"""

import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import possession_v10_standalone as S  # noqa: E402

REF_PATH = os.path.join(ROOT, "diagnostics", "key_clues_ref.json")
LABELS = {"media_vuelta": "Media vuelta de los equipos", "balon": "Pistas del balón (fuera, borde, velocidad)",
          "forma": "Forma de los equipos (comparación)"}
METRICS = [("beyond3", "valores fuera de ±3", "{:.2%}"), ("beyond6", "valores fuera de ±6 (la red los recorta)", "{:.2%}"),
           ("mean_abs", "distancia media al promedio de entrenamiento", "{:.2f}"),
           ("col_shift", "mayor corrimiento de una columna", "{:.2f}"),
           ("col_spread", "mayor dispersión de una columna", "{:.2f}")]


def verdict(value, r):
    """Inside the external range, a little above it (up to 1.5 times the maximum) or clearly above it."""
    if value <= r["max"]:
        return "normal"
    if value <= 1.5 * r["max"]:
        return "un poco alto"
    return "FUERA DE RANGO"


def features():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if not os.path.exists(S.CACHE_PATH):
        S.build_cache()
    data = np.load(S.CACHE_PATH)
    ball, pos, speed, slot_team = data["ball"], data["pos"], data["speed"], data["slot_team"]
    params = json.load(open(S.PARAMS_PATH))
    S.apply_params(params)
    periods = [(a, b) for _, a, b in S.half_video_blocks(S.GAME)]
    c = S.common(ball, pos, speed, slot_team, periods, S.GAMES[S.GAME]["pitch"])
    gap = S.gap_context(c)
    dall = S.features_dead(c, gap)
    tn, tb = S.features_team(c, pos, slot_team, ball, gap)
    return params, periods, c, dall, tn, tb


def group_stats(Z, cols):
    Zg = Z[:, cols]
    return {"beyond6": float(np.mean(np.abs(Zg) > 6)), "beyond3": float(np.mean(np.abs(Zg) > 3)),
            "mean_abs": float(np.mean(np.abs(Zg))), "col_shift": float(np.max(np.abs(Zg.mean(0)))),
            "col_spread": float(np.max(Zg.std(0)))}


def decode(emis, periods, kickoffs, switch, dead):
    codes = np.full(len(emis), S.X, dtype=np.int8)
    for first, last in periods:
        e = emis[first:last + 1].copy()
        kick = [k for k in kickoffs if first <= k <= last + 1]
        forced = (kick[0] - first) if kick else 0
        if forced > 0:
            e[:forced, S.H] = e[:forced, S.A] = -1e9
        codes[first:last + 1] = S.viterbi(e, switch, dead)
    return codes


def main():
    ref = json.load(open(REF_PATH))
    params, periods, c, dall, tn, tb = features()
    graded, seen = c["graded"], c["seen"]
    z = np.load(S.SEQ_PATH)
    mu, sd = z["mu"].astype(np.float64), z["sd"].astype(np.float64)
    const = sd < 1e-3
    s = seen.astype(np.float32)[:, None]
    X = np.hstack([dall, tn, tb * s, s]).astype(np.float64)
    Z = (X[graded][::5] - mu) / sd
    Z[:, const] = 0

    print("1. COMO SE VEN LAS PISTAS (partido del reto contra 8 partidos externos x 4 variantes)\n")
    groups = {g: [c_ for c_ in ref["_groups"][g] if not const[c_]] for g in LABELS}
    flags = []
    for g, cols in groups.items():
        st = group_stats(Z, cols)
        print("  {} ({} columnas)".format(LABELS[g], len(cols)))
        for key, text, fmt in METRICS:
            r = ref[g][key]
            v = verdict(st[key], r)
            if v != "normal":
                flags.append((g, text, v))
            print("    {:<46} {:>8}   externos {} a {}   {}".format(
                text, fmt.format(st[key]), fmt.format(r["min"]), fmt.format(r["max"]), v))
        print()

    print("2. CUANTO SE APOYA LA RED EN CADA GRUPO AQUI (frames calificados cuyo resultado v10 cambia al esconderlo)\n")
    ld = S.Logistic(params["models"]["dead"]).logit(dall)
    models = {k: S.Logistic(v) for k, v in params["models"].items()}
    lt = np.where(seen, models["team_ball"].logit(tb), models["team_noball"].logit(tn))
    home, away = S.team_names(S.require(["atd_meta"])[0])
    tags, kickoffs = S.tag_team_logit(len(X), home, away)
    lt = lt + S.TAG_LOGIT * tags
    dec = params["decode"]
    e91 = S.emissions(ld, lt, dec["temp_dead"], dec["temp_team"])

    def v10_codes(Xin):
        P = {k: z[k].astype(np.float64) for k in z.files if k not in ("mu", "sd")}
        eg = np.zeros((len(Xin), 3))
        for first, last in periods:
            idx = np.arange(first, last + 1, S.SEQ_STEP)
            x = np.clip((Xin[idx] - mu) / sd, -6.0, 6.0)
            x[:, const] = 0.0
            lp = S.seq_log_probs(P, x.astype(np.float32).astype(np.float64))
            j = np.minimum(np.round((np.arange(first, last + 1) - first) / S.SEQ_STEP).astype(int), len(lp) - 1)
            eg[first:last + 1] = lp[j]
        return decode((e91 + eg) / 2, periods, kickoffs, dec["switch_cost"], dec["dead_cost"])

    base = v10_codes(X)
    for g, cols in groups.items():
        X2 = X.copy()
        X2[:, cols] = mu[cols]
        share = float(np.mean(base[graded] != v10_codes(X2)[graded]))
        r = ref[g]["hide_change"]
        v = "normal" if r["min"] / 1.5 <= share <= r["max"] * 1.5 else ("depende MAS que en los externos" if share > r["max"] * 1.5
                                                                      else "depende MENOS que en los externos")
        if v != "normal":
            flags.append((g, "dependencia de la red", v))
        print("  {:<46} {:>7.2%}   externos {:.2%} a {:.2%}   {}".format(LABELS[g], share, r["min"], r["max"], v))

    print("\nRESUMEN")
    key_flags = [f for f in flags if f[0] in ("media_vuelta", "balon")]
    if not key_flags:
        print("  Las dos pistas clave de la red se ven como en los partidos externos: no hay señal de que la red")
        print("  esté recibiendo algo distinto a lo que aprendió.")
    else:
        for g, text, v in key_flags:
            print("  {}: {} -> {}".format(LABELS[g], text, v))
        serious = [f for f in key_flags if f[2] != "un poco alto"]
        if serious:
            print("  Revisar antes de entregar: la red podría estar leyendo estas pistas distinto que en el entrenamiento.")
        else:
            print("  Solo diferencias menores (hasta 1.5 veces el máximo de los externos): no es señal de alarma por sí sola.")


if __name__ == "__main__":
    main()
