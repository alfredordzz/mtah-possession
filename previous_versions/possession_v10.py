"""The v10 candidate: v9.1 averaged with a sequence model.

    python possession_v10.py

v10 = v9.1's judges (model_v9_1_params.json) and, over the same per-frame features, a bidirectional
GRU (two layers, 64 units per direction, model_v10_seq.npz) that reads each half at 5 Hz forwards and
backwards. The two sets of log-probabilities for H, A and D are averaged frame by frame and decoded by
the same Viterbi editor (switch cost 16, dead cost 35), with v9.1's Smart Tagging nudge and kickoff
rule. Both were trained on 8 external matches only (training/train_final.py, training/train_seq.py).
Inference is plain numpy, like the rest of the pipeline. Writes evaluation/results/v10.csv.
"""

import os

import possession_v9_standalone as S

PARAMS_PATH = os.path.join(S.ROOT, "model_v9_1_params.json")
SEQ_PATH = os.path.join(S.HERE, "model_v10_seq.npz")
OUT_PATH = os.path.join(S.ROOT, "evaluation", "results", "v10.csv")


def main():
    S.main(OUT_PATH, PARAMS_PATH, SEQ_PATH)


if __name__ == "__main__":
    main()
