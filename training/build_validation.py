"""Build every validation variant used for training, from the two external sources.

    python training/build_validation.py METRICA_SAMPLE_DATA IDSSE_NPZ OUT_ROOT

METRICA_SAMPLE_DATA  the data/ folder of https://github.com/metrica-sports/sample-data
IDSSE_NPZ            the output folder of training/idsse_convert.py
OUT_ROOT             where to write; training/train_final.py reads it

Writes OUT_ROOT/metrica/<variant>/val_game{1,2,3}.npz and OUT_ROOT/idsse/<variant>/idsse_<ID>.npz
for the four variants noisy, noisy_hard, smooth, smooth_hard (tracker profile x extra in-play ball
gaps). Every random draw is seeded, so the output is the same on every run.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idsse_val  # noqa: E402
import val_data  # noqa: E402

VARIANTS = {  # name: (extra in-play ball miss share, mean gap seconds, tracker profile)
    "noisy": (0.0, 2.5, "noisy"),
    "noisy_hard": (0.08, 8.0, "noisy"),
    "smooth": (0.0, 2.5, "smooth"),
    "smooth_hard": (0.08, 8.0, "smooth"),
}


def main(metrica_src, idsse_src, out_root):
    for name, (miss, gap, profile) in VARIANTS.items():
        print("metrica / {}".format(name), flush=True)
        val_data.main(metrica_src, os.path.join(out_root, "metrica", name), miss, gap, profile=profile)
    print("idsse (all variants)", flush=True)
    idsse_val.main(idsse_src, os.path.join(out_root, "idsse"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
