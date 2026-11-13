"""Pack three trained sequence networks into model_v11_seq.npz (v11 averages their probabilities).

    python training/build_v11_seq.py previous_versions/model_v10_seq.npz cache/seq_s1.npz cache/seq_s2.npz

Each network's arrays are stored with a prefix (n0__, n1__, n2__), including its own input mean and
spread, so possession_v11_standalone.py can run them one by one and average.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    args = sys.argv[1:]
    dest = os.path.join(ROOT, "previous_versions", "model_v11_seq.npz")
    if "--out" in args:
        i = args.index("--out"); dest = os.path.abspath(args[i + 1]); del args[i:i + 2]
    paths = args
    if len(paths) < 2:
        sys.exit(__doc__)
    out = {}
    for i, p in enumerate(paths):
        z = np.load(p)
        for k in z.files:
            out["n{}__{}".format(i, k)] = z[k]
    np.savez(dest, **out)
    print("wrote {} with {} networks".format(os.path.relpath(dest, ROOT), len(paths)))


if __name__ == "__main__":
    main()
