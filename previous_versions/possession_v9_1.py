"""The submission: v9.1 for the competition match.

    python possession_v9_1.py

v9.1 = v9's features, Smart Tagging and decoding (switch cost 16, dead cost 35) with the team
regressions trained with extra weight on the 75 frames (3 s) either side of a change of possession
instead of 25. The weights are in model_v9_1_params.json, learned on 8 external matches only
(training/train_final.py --version v9 --name v9.1 --transition-window 75).

Runs the single-file pipeline (possession_v9_standalone.py) with those weights and writes
evaluation/results/v9_1.csv, then validates it. Prints counts and format checks only.
"""

import os

import possession_v9_standalone as S

PARAMS_PATH = os.path.join(S.ROOT, "model_v9_1_params.json")
OUT_PATH = os.path.join(S.ROOT, "evaluation", "results", "v9_1.csv")


def main():
    S.main(OUT_PATH, PARAMS_PATH)


if __name__ == "__main__":
    main()
