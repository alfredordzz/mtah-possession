"""Pack three trained sequence networks into model_v11_seq.npz (v11 averages their probabilities).

    python training/build_v11_seq.py model_v10_seq.npz cache/seq_s1.npz cache/seq_s2.npz

Each network's arrays are stored with a prefix (n0__, n1__, n2__), including its own input mean and
spread, so possession_v11_standalone.py can run them one by one and average.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        sys.exit(__doc__)
    out = {}
    for i, p in enumerate(paths):
        z = np.load(p)
        for k in z.files:
            out["n{}__{}".format(i, k)] = z[k]
    dest = os.path.join(ROOT, "model_v11_seq.npz")
    np.savez(dest, **out)
    print("wrote {} with {} networks".format(os.path.relpath(dest, ROOT), len(paths)))


if __name__ == "__main__":
    main()
