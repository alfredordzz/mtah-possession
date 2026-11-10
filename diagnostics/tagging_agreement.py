"""How a submission lines up with the Smart Tagging phases. Aggregates only, no ground truth.

    python diagnostics/tagging_agreement.py                         # v9.csv (and v9_1.csv if present)
    python diagnostics/tagging_agreement.py evaluation/results/v9.csv

Smart Tagging is automatic and is not ground truth: this is a consistency check, used only to decide
whether an idea is worth simulating on the external games. It must not be used to choose between
versions, since agreeing more with the tagger also means copying its mistakes.

For each submission it prints:
    1. per phase code: the frames inside its intervals labelled H / A / D, and how often the team
       in possession is the one the phase implies (attacking team for offensive codes, the other
       team for defensive ones)
    2. transition starts: how far each tagged start of an ATTACKING TRANSITION is from the nearest
       change of team in the submission, against the same distance from ordinary in-play frames
       (what chance alone gives), and whether the team after that change is the tagged team
    3. set pieces with a team: whether the first team in possession after the tagged interval is
       the tagged team
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import possession_metrics as pm  # noqa: E402

FPS = 25
OFFENSIVE = {"build up", "progression", "creating chances", "attacking transition", "long ball",
             "ball in final third", "ball in the box", "players in the box", "set pieces"}
DEFENSIVE = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}
WINDOWS_S = (1, 2, 3, 5)
AFTER_S = 5          # set pieces: look this long after the tagged interval for the first team on the ball


def team_changes(codes):
    """Frames where the team in possession differs from the previous team in possession (D/X skipped)."""
    alive = np.flatnonzero((codes == pm.H) | (codes == pm.A))
    if len(alive) < 2:
        return np.array([], dtype=int), alive
    team = codes[alive]
    return alive[1:][team[1:] != team[:-1]], alive


def nearest(events, at):
    """Distance in frames from each frame in `at` to the nearest frame in `events`, and that event."""
    if len(events) == 0:
        return np.full(len(at), np.inf), np.full(len(at), -1)
    i = np.clip(np.searchsorted(events, at), 1, len(events) - 1)
    left, right = events[i - 1], events[i]
    pick = np.where(np.abs(at - left) <= np.abs(right - at), left, right)
    if len(events) == 1:
        pick = np.full(len(at), events[0])
    return np.abs(pick - at), pick


def report(codes, events, home, away):
    n = len(codes)
    names = {home.strip().lower(): pm.H, away.strip().lower(): pm.A}
    graded = codes != pm.X

    print("1. PHASES (frames inside the tagged intervals, graded frames only)")
    print("  {:<32} {:>6} {:>8} {:>6} {:>6} {:>6}   {}".format(
        "code", "inst.", "frames", "H", "A", "D", "team in possession = team the phase implies"))
    by_code = {}
    for e in events:
        by_code.setdefault((e["code"] or "").strip().lower(), []).append(e)
    for code, evs in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
        mask = np.zeros(n, dtype=bool)
        expect = np.full(n, -1)
        for e in evs:
            a, b = max(0, e["start_frame"]), min(n, e["end_frame"])
            if b <= a:
                continue
            mask[a:b] = True
            team = names.get((e["tags"].get("Team") or "").strip().lower())
            if team is not None and (code in OFFENSIVE or code in DEFENSIVE):
                expect[a:b] = team if code in OFFENSIVE else (pm.A if team == pm.H else pm.H)
        m = mask & graded
        if not m.any():
            continue
        shares = [np.mean(codes[m] == k) for k in (pm.H, pm.A, pm.D)]
        alive = m & ((codes == pm.H) | (codes == pm.A)) & (expect >= 0)
        agree = "{:.0%} of {} in-play frames".format(np.mean(codes[alive] == expect[alive]), int(alive.sum())) \
            if alive.any() else "(no team or not a phase)"
        print("  {:<32} {:>6} {:>8} {:>6.0%} {:>6.0%} {:>6.0%}   {}".format(code, len(evs), int(m.sum()), *shares, agree))

    changes, alive_idx = team_changes(codes)
    print("\n2. ATTACKING TRANSITION STARTS vs changes of team in the submission ({} changes in total)".format(len(changes)))
    starts = sorted({max(0, e["start_frame"]) for e in by_code.get("attacking transition", [])
                     if 0 <= e["start_frame"] < n and graded[e["start_frame"]]})
    if not starts:
        print("  no tagged attacking transitions inside the graded halves")
    else:
        starts = np.array(starts)
        d, pick = nearest(changes, starts)
        sample = alive_idx[::FPS]                      # ordinary in-play frames, one per second
        d0, _ = nearest(changes, sample)
        print("  {} transition starts; share with a change of team within:".format(len(starts)))
        for w in WINDOWS_S:
            print("    +/-{} s   transitions {:>4.0%}   ordinary in-play frames {:>4.0%}".format(
                w, np.mean(d <= w * FPS), np.mean(d0 <= w * FPS)))
        print("  median distance to the nearest change: transitions {:.1f} s, ordinary frames {:.1f} s".format(
            np.median(d) / FPS, np.median(d0) / FPS))
        team_of = {}
        for e in by_code.get("attacking transition", []):
            t = names.get((e["tags"].get("Team") or "").strip().lower())
            if t is not None:
                team_of[max(0, e["start_frame"])] = t
        ok = [(codes[p] == team_of[s]) for s, p, dd in zip(starts, pick, d)
              if dd <= 3 * FPS and s in team_of and p >= 0]
        if ok:
            print("  of the changes within +/-3 s, the new team is the tagged team in {:.0%} ({} of {})".format(
                np.mean(ok), int(np.sum(ok)), len(ok)))

    print("\n3. SET PIECES WITH A TEAM: first team in possession in the {} s after the tagged interval".format(AFTER_S))
    rows = {}
    for e in by_code.get("set pieces", []):
        team = names.get((e["tags"].get("Team") or "").strip().lower())
        kind = (e["tags"].get("Type") or "?").strip()
        if team is None:
            continue
        a, b = min(n, max(0, e["end_frame"])), min(n, max(0, e["end_frame"]) + AFTER_S * FPS)
        seg = codes[a:b]
        first = seg[(seg == pm.H) | (seg == pm.A)]
        r = rows.setdefault(kind, [0, 0, 0])
        if len(first) == 0:
            r[2] += 1
        elif first[0] == team:
            r[0] += 1
        else:
            r[1] += 1
    if not rows:
        print("  no set pieces with a team")
    for kind, (same, other, none) in sorted(rows.items(), key=lambda kv: -sum(kv[1])):
        print("  {:<24} tagged team {:>3}   other team {:>3}   still dead ball {:>3}".format(kind, same, other, none))


def main():
    from games import GAME, require, source_frames  # noqa: F401
    from baseline import team_names
    from load_smart_tagging import load as load_tags

    paths = sys.argv[1:] or [p for p in (os.path.join(ROOT, "evaluation", "results", "v9.csv"),
                                         os.path.join(ROOT, "evaluation", "results", "v9_1.csv")) if os.path.exists(p)]
    if not paths:
        sys.exit("usage: python diagnostics/tagging_agreement.py submission.csv [another.csv]")
    home, away = team_names(require(["atd_meta"])[0])
    events = load_tags()
    for path in paths:
        codes = pm.label_codes(pm.read_label_csv(path))
        print("=" * 100)
        print(os.path.relpath(path, ROOT))
        print("=" * 100)
        report(codes, events, home, away)
        print()


if __name__ == "__main__":
    main()
