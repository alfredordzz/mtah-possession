"""Descriptive summary of the Smart Tagging XML: which codes and labels it uses. Aggregates only.

    python diagnostics/smart_tagging_stats.py
    python diagnostics/smart_tagging_stats.py --file path/to/other_smart_tagging.xml

Reads the XML directly (standard library only) and prints:
    1. size of the file: instances, time covered, first and last tag
    2. every <code>: how many instances, total and median duration
    3. every label <group>: in how many instances it appears, its values and how often each occurs
       (numeric groups are summarised by range instead of value by value)
    4. which label groups each code carries
    5. code x Team: how many instances of each code name each team
    6. overlap: share of the tagged time covered by 1, 2, 3+ simultaneous instances
Nothing here reads labels of possession or the ground truth, and nothing is written to disk.
"""

import argparse
import os
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MAX_VALUES = 25   # values listed per group before the rest are summarised


def default_path():
    from games import GAME, require
    return require(["smart_tagging"], GAME)[0]


def is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def fmt_s(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return "{}:{:02d}".format(m, s)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", help="Smart Tagging XML (default: the competition match's file)")
    args = parser.parse_args()
    path = args.file or default_path()

    instances = []
    for inst in ET.parse(path).iter("instance"):
        start, end = float(inst.findtext("start")), float(inst.findtext("end"))
        labels = [((lab.findtext("group") or "").strip(), (lab.findtext("text") or "").strip())
                  for lab in inst.findall("label")]
        instances.append({"code": (inst.findtext("code") or "").strip(), "start": start, "end": end,
                          "dur": end - start, "labels": labels})
    if not instances:
        sys.exit("no <instance> elements in {}".format(path))

    # 1. size
    first = min(i["start"] for i in instances)
    last = max(i["end"] for i in instances)
    print("1. FILE")
    print("  {} instances, from {} to {} of video ({:.1f} min span), {} distinct codes".format(
        len(instances), fmt_s(first), fmt_s(last), (last - first) / 60,
        len({i["code"] for i in instances})))

    # 2. codes
    print("\n2. CODES (instances, total duration, median duration)")
    by_code = defaultdict(list)
    for i in instances:
        by_code[i["code"]].append(i["dur"])
    for code, durs in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
        print("  {:<34} {:>5}   total {:>7}   median {:>5.1f} s".format(
            code, len(durs), fmt_s(sum(durs)), statistics.median(durs)))

    # 3. label groups and values
    print("\n3. LABEL GROUPS (instances that carry the group; values and counts)")
    group_inst = Counter()
    group_vals = defaultdict(Counter)
    repeated = Counter()
    for i in instances:
        seen = Counter(g for g, _ in i["labels"])
        for g, n in seen.items():
            group_inst[g] += 1
            if n > 1:
                repeated[g] += 1
        for g, v in i["labels"]:
            group_vals[g][v] += 1
    for g, n in sorted(group_inst.items(), key=lambda kv: -kv[1]):
        vals = group_vals[g]
        extra = ", repeated inside {} instances".format(repeated[g]) if repeated[g] else ""
        print("  {} : in {} instances ({:.0%}), {} distinct values{}".format(
            g, n, n / len(instances), len(vals), extra))
        if vals and all(is_number(v) for v in vals):
            nums = [float(v) for v, c in vals.items() for _ in range(c)]
            print("      numeric: min {:g}, median {:g}, max {:g}".format(min(nums), statistics.median(nums), max(nums)))
            if len(vals) <= MAX_VALUES:
                print("      " + ", ".join("{}: {}".format(v, c) for v, c in sorted(vals.items(), key=lambda kv: float(kv[0]))))
        else:
            for v, c in vals.most_common(MAX_VALUES):
                print("      {:<40} {:>5}".format(v or "(empty)", c))
            if len(vals) > MAX_VALUES:
                print("      ... {} more values".format(len(vals) - MAX_VALUES))

    # 4. which groups each code carries
    print("\n4. LABEL GROUPS PER CODE (share of that code's instances carrying the group)")
    for code, durs in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
        cnt = Counter(g for i in instances if i["code"] == code for g in {g for g, _ in i["labels"]})
        print("  {:<34} ".format(code) + ", ".join("{} {:.0%}".format(g, c / len(durs)) for g, c in cnt.most_common()))

    # 5. code x Team
    teams = sorted({v for i in instances for g, v in i["labels"] if g.lower() == "team"})
    if teams:
        print("\n5. CODE x TEAM (instances)")
        print("  {:<34} ".format("") + "".join("{:>14}".format(t[:13]) for t in teams + ["(no team)"]))
        for code in sorted(by_code, key=lambda c: -len(by_code[c])):
            row = Counter()
            for i in instances:
                if i["code"] == code:
                    t = [v for g, v in i["labels"] if g.lower() == "team"]
                    row[t[0] if t else "(no team)"] += 1
            print("  {:<34} ".format(code) + "".join("{:>14}".format(row[t]) for t in teams + ["(no team)"]))

    # 6. overlap
    edges = sorted([(i["start"], 1) for i in instances] + [(i["end"], -1) for i in instances])
    depth, prev, cover = 0, None, Counter()
    for t, d in edges:
        if prev is not None and depth > 0:
            cover[min(depth, 3)] += t - prev
        depth += d
        prev = t
    tagged = sum(cover.values())
    print("\n6. OVERLAP")
    print("  tagged time {} of a {} span ({:.0%}); of the tagged time: 1 instance {:.0%}, 2 at once {:.0%}, 3+ at once {:.0%}".format(
        fmt_s(tagged), fmt_s(last - first), tagged / max(last - first, 1e-9),
        cover[1] / tagged, cover[2] / tagged, cover[3] / tagged))


if __name__ == "__main__":
    main()
