"""Automatic check: how often do the Smart Tagging phases agree with the tracking? Aggregates only.

    python diagnostics/phase_agreement.py

Runs the v4 team model WITHOUT the phase nudge and compares, frame by frame, the team it leans to
with the team each phase tag points to (offensive phases: the tagged team; defensive phases: the
other one). Only frames where the model is confident and the ball is alive by the model's own call
are counted. No labels are involved: it measures agreement between two automatic inputs, which
tells us how reliable the tags are likely to be. Prints shares and counts only.
"""

import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for sub in ("", "previous_versions", "starter"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from baseline import team_names  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
import model_v4 as M  # noqa: E402
from possession_v4 import OFFENSIVE, DEFENSIVE  # noqa: E402


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    z = np.load(os.path.join(ROOT, "cache", "atd_tactical.npz"))
    ball, pos, speed, st = z["ball"], z["pos"], z["speed"], z["slot_team"]
    params = json.load(open(os.path.join(ROOT, "previous_versions", "model_v4_params.json")))
    for k, v in params.get("touch", {}).items():
        setattr(M, {"dv": "TOUCH_DV", "reach": "TOUCH_REACH", "smooth": "TOUCH_SMOOTH",
                    "control_reach": "CONTROL_REACH"}[k], int(v) if k == "smooth" else v)
    models = {k: M.Logistic.from_dict(v) for k, v in params["models"].items()}
    periods = [(a, b) for _, a, b in half_video_blocks(GAME)]
    c = M.common(ball, pos, speed, st, periods, GAMES[GAME]["pitch"])
    gap = M.gap_context(c)
    dall = M.features_dead(c, gap)
    tn, tb = M.features_team(c, pos, st, ball, gap)
    ld = models["dead"].logit(dall)
    lt = np.where(c["seen"], models["team_ball"].logit(tb), models["team_noball"].logit(tn))
    n, g = c["n"], c["graded"]
    alive = g & (ld < 0)
    home, away = team_names(require(["atd_meta"])[0])
    names = {home.strip().lower(): 1.0, away.strip().lower(): -1.0}

    rows = {}
    covered = np.zeros(n, dtype=bool)
    for e in load_tags():
        code = (e["code"] or "").strip().lower()
        team = names.get((e["tags"].get("Team") or "").strip().lower())
        if team is None or code not in OFFENSIVE | DEFENSIVE or code == "set pieces":
            continue
        sign = team if code in OFFENSIVE else -team
        s, t = max(0, e["start_frame"]), min(n, e["end_frame"])
        covered[s:t] = True
        r = rows.setdefault(code, [0, 0.0, 0, 0])
        r[0] += 1
        r[1] += (t - s) / M.FPS
        m = alive[s:t] & (np.abs(lt[s:t]) > 3.0)      # model clearly leans one way
        r[2] += int(np.sum(np.sign(lt[s:t][m]) == sign))
        r[3] += int(m.sum())

    print("phase tags vs the tracking-only team model (aggregates only)\n")
    print("graded frames covered by at least one phase tag: {:.1%}".format(covered[g].mean()))
    print("  of the frames the model calls alive: {:.1%}\n".format(covered[alive].mean()))
    print("  {:<30} {:>5} {:>9} {:>17}".format("code", "tags", "mean dur", "agree (confident)"))
    tot_a = tot_m = 0
    for code, (k, dur, agree, m) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        tot_a, tot_m = tot_a + agree, tot_m + m
        print("  {:<30} {:>5} {:>8.1f}s {:>16.1%}".format(code, k, dur / k, agree / max(m, 1)))
    print("\noverall agreement on confident alive frames: {:.1%}".format(tot_a / max(tot_m, 1)))


if __name__ == "__main__":
    main()
