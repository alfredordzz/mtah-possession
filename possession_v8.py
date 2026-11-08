"""v8: possession for the competition match with the model trained on other games.

    python possession_v8.py

Needs cache/atd_tactical.npz (from cache_atd.py), the Smart Tagging XML, model_v2.py, model_v3.py,
model_v4.py, model_v6.py, model_v8.py and model_v8_params.json (weights learned on 8 external games by
training/train_final.py; nothing in it comes from this match or its ground truth).
Writes evaluation/results/v8.csv and validates it.

Prints counts and format checks only: no comparison against the public ground truth, which the
organisers' rules reserve for checking that the file is well formed.
"""

import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_names, write_submission  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
import possession_metrics as pm  # noqa: E402
import model_v4 as M4  # noqa: E402
import model_v6 as M6  # noqa: E402
import model_v8 as M  # noqa: E402
from model_v8 import FPS  # noqa: E402

CACHE_PATH = os.path.join(ROOT, "cache", "atd_tactical.npz")
PARAMS_PATH = os.path.join(ROOT, "model_v8_params.json")
OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v8.csv")

# Smart Tagging exists only for this match, so its weight cannot be calibrated on the validation
# games. It is kept small: a phase tag nudges the team call, it never decides it on its own.
# Its size was checked on the external games with simulated phases matching the coverage (~79% of
# alive frames), durations and agreement (~73-80%) the real tags showed against the tracking-only
# model (diagnostics/phase_agreement.py): 0.25 to 0.5 is neutral, 1 or more hurts. 0.25 is the
# value that did best when the simulated tags were least reliable.
# (That check was run with v4's team model; v6 and v8 keep the same small value.)
TAG_LOGIT = 0.25
OFFENSIVE = {"build up", "progression", "creating chances", "attacking transition", "long ball",
             "ball in final third", "ball in the box", "players in the box", "set pieces"}
DEFENSIVE = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}


def tag_team_logit(n, home_name, away_name):
    """+1 per frame where a phase points to H, -1 where it points to A; plus kickoff frames."""
    names = {home_name.strip().lower(): M.H, away_name.strip().lower(): M.A}
    out = np.zeros(n)
    kickoffs = []
    for event in load_tags():
        code = (event["code"] or "").strip().lower()
        team = names.get((event["tags"].get("Team") or "").strip().lower())
        start, stop = max(0, event["start_frame"]), min(n, event["end_frame"])
        if code == "set pieces" and (event["tags"].get("Type") or "").strip().lower() == "kick off (start)":
            kickoffs.append(start)
        if team is None or stop <= start:
            continue
        if code in OFFENSIVE:
            sign = 1.0 if team == M.H else -1.0
        elif code in DEFENSIVE:
            sign = -1.0 if team == M.H else 1.0
        else:
            continue
        out[start:stop] += sign
    return np.clip(out, -1, 1), sorted(kickoffs)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN rows are expected
    n = source_frames()
    data = np.load(CACHE_PATH)
    ball, pos, speed, slot_team = data["ball"], data["pos"], data["speed"], data["slot_team"]
    params = json.load(open(PARAMS_PATH))
    models = {k: M.Logistic.from_dict(v) for k, v in params["models"].items()}
    dec = params["decode"]
    touch_cfg = params.get("touch", {})
    M4.TOUCH_DV = touch_cfg.get("dv", M4.TOUCH_DV)
    M4.TOUCH_REACH = touch_cfg.get("reach", M4.TOUCH_REACH)
    M4.TOUCH_SMOOTH = int(touch_cfg.get("smooth", M4.TOUCH_SMOOTH))
    M4.CONTROL_REACH = touch_cfg.get("control_reach", M4.CONTROL_REACH)
    flight_cfg = params.get("flight") or {}
    M6.ASSIGN_WIN = int(flight_cfg.get("assign_win", M6.ASSIGN_WIN))
    M6.FLIGHT_SPEED = flight_cfg.get("speed", M6.FLIGHT_SPEED)
    M6.FLIGHT_MIN_M = flight_cfg.get("min_m", M6.FLIGHT_MIN_M)
    M6.FLIGHT_MAX_S = flight_cfg.get("max_s", M6.FLIGHT_MAX_S)
    M6.DECAY_S = flight_cfg.get("decay_s", M6.DECAY_S)
    M6.LOOKAHEAD_S = tuple(flight_cfg.get("lookahead_s", M6.LOOKAHEAD_S))

    blocks = half_video_blocks(GAME)
    periods = [(first, last) for _, first, last in blocks]
    home_name, away_name = team_names(require(["atd_meta"])[0])

    c = M.common(ball, pos, speed, slot_team, periods, GAMES[GAME]["pitch"])
    gap = M.gap_context(c)
    dall = M.features_dead(c, gap)
    tn, tb = M.features_team(c, pos, slot_team, ball, gap)
    seen = c["seen"]
    ld = models["dead"].logit(dall)
    lt = np.where(seen, models["team_ball"].logit(tb), models["team_noball"].logit(tn))
    tags, kickoffs = tag_team_logit(n, home_name, away_name)
    lt = lt + TAG_LOGIT * tags

    emis = M.emissions(ld, lt, dec["temp_dead"], dec["temp_team"])
    codes = np.full(n, M.X, dtype=np.int8)
    print("decoding (switch cost {}, dead cost {})".format(dec["switch_cost"], dec["dead_cost"]))
    for first, last in periods:
        e = emis[first:last + 1].copy()
        kick = [k for k in kickoffs if first <= k <= last + 1]
        forced = (kick[0] - first) if kick else 0
        if forced > 0:
            e[:forced, M.H] = e[:forced, M.A] = -1e9
        codes[first:last + 1] = M.viterbi(e, dec["switch_cost"], dec["dead_cost"])
        print("  half from frame {}: {} frames forced D before the tagged kickoff".format(first, forced))

    letters = np.array(["H", "A", "D", "X"])
    write_submission(OUT_PATH, letters[codes])
    table = pm.read_label_csv(OUT_PATH)
    reason = pm.validate_submission(table, n)
    print("\nvalidation: {}".format("OK" if reason is None else "FAIL " + reason))

    graded = c["graded"]
    stats = pm.possession_stats(pm.label_codes(table), [(a, b + 1) for a, b in periods])
    print("\nfull match (no ground truth, sanity only)")
    print("  {} possessions, mean {:.1f} s, home share {:.1%}".format(
        stats["n_spells"], stats["mean_duration_s"], stats["home_share"]))
    print("  (on the 8 validation games the model finds 75-90% of the true number of possessions;"
          " merging the shortest ones scores better than guessing them)")
    print("  dead ball {:.1%} of graded frames (model's frame-level P(D) > 0.5 on {:.1%})".format(
        np.mean(codes[graded] == M.D), np.mean(ld[graded] > 0)))
    print("  team evidence leans home on {:.1%} of graded frames".format(np.mean(lt[graded] > 0)))
    missing = graded & ~seen
    print("  ball missing on {:.1%} of graded frames; of those, labelled D: {:.1%}".format(
        np.mean(missing[graded]), np.mean(codes[missing] == M.D)))
    print("  ball seen: labelled D on {:.1%}".format(np.mean(codes[graded & seen] == M.D)))
    _, touch = M4.detect_touches(c)
    minutes = graded.sum() / FPS / 60
    print("  touches detected: {:.1f} per minute of graded time (8 validation games, smooth tracker profile: 13-23)".format(
        touch[graded].sum() / minutes))
    sign, conf = M6.touch_teams(c, touch)
    ball_s = M4.smooth_ball(c["_ball"].astype(np.float64), M4.TOUCH_SMOOTH)
    v = M4.ball_velocity(ball_s, k=3)
    fl = M6.flights(c, touch, sign, conf, np.hypot(v[:, 0], v[:, 1]))
    turn = sum(1 for f in fl if f[2] != f[3])
    print("  ball flights between touches: {:.1f} per minute, {:.0%} between different teams "
          "(validation games: 7.5-10 per minute, ~33%)".format(len(fl) / minutes, turn / max(len(fl), 1)))
    m = graded & seen
    for name, team, sign in (("home", M.H, c["sign_home"]), ("away", M.A, -c["sign_home"])):
        players = pos[:, slot_team == team].astype(np.float64)
        depth, width, compact, _, _, _ = M.team_shape(players, sign, c["_ball"].astype(np.float64), c["length"])
        visible = np.sum(~np.isnan(players[..., 0]), axis=1)
        print("  {} shape, medians: depth {:.1f} m, width {:.1f} m, compactness {:.1f} m, {:.0f} players tracked "
              "(validation games: 28-33, 24-32, 16-19 m, 10)".format(
                  name, np.nanmedian(depth[m]), np.nanmedian(width[m]), np.nanmedian(compact[m]),
                  np.median(visible[graded])))
    print("\nwrote {}".format(os.path.relpath(OUT_PATH, ROOT)))


if __name__ == "__main__":
    main()
