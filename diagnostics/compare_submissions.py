"""Compare two submission files, e.g. the same version produced on two machines. Aggregates only.

    python diagnostics/compare_submissions.py mine.csv theirs.csv

Prints each file's SHA-256, its label counts and number of possessions, and how many frames differ
and in how many separate stretches. Identical hashes mean the run is reproduced exactly. A handful
of differing frames would point to floating-point differences between machines (different CPU
math libraries), not to a different model.
"""

import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def describe(path):
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    codes = pm.label_codes(pm.read_label_csv(path))
    counts = {k: int(np.sum(codes == v)) for k, v in (("H", pm.H), ("A", pm.A), ("D", pm.D), ("X", pm.X))}
    starts = np.flatnonzero(np.diff(np.concatenate(([pm.D], codes))) != 0)
    spells = int(sum(1 for i in starts if codes[i] in (pm.H, pm.A)))
    print("{}\n  sha256 {}\n  rows {}  H {H}  A {A}  D {D}  X {X}  possessions {}".format(
        path, digest, len(codes), spells, **counts))
    return digest, codes


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python diagnostics/compare_submissions.py first.csv second.csv")
    d1, c1 = describe(sys.argv[1])
    d2, c2 = describe(sys.argv[2])
    if d1 == d2:
        print("\nidentical files: the run is reproduced exactly")
        return
    if len(c1) != len(c2):
        print("\ndifferent number of rows: one of the files is not a full submission")
        return
    diff = c1 != c2
    edges = np.flatnonzero(np.diff(np.concatenate(([0], diff.astype(np.int8), [0]))) == 1)
    print("\nfiles differ on {} of {} frames ({:.4%}), in {} separate stretches".format(
        int(diff.sum()), len(c1), diff.mean(), len(edges)))


if __name__ == "__main__":
    main()
