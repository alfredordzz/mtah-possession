# History of the earlier versions (v4 to v12)

Everything here still runs from the repository root. The submission is v13 (see the main README).

### Single-file version of the submission

`possession_v10_standalone.py` is the whole v10 in one file, from the raw tracking to the CSV. It
does not import any of our other scripts: the tracking reader of `cache_atd.py` and every function
of `possession_v9_standalone.py` that v10 uses are copied into it. It needs only numpy, the weights
in `model_v9_1_params.json` and `model_v10_seq.npz`, and the organisers' loaders (`games.py`,
`starter/`, `evaluation/`). On the first run it reads the raw tracking into `cache/` by itself
(`--rebuild-cache` forces it). It writes `evaluation/results/v10_standalone.csv` (or
`--out path.csv`), byte-identical to `v10.csv`.

```powershell
python previous_versions/possession_v10_standalone.py
Get-FileHash evaluation\results\v10_standalone.csv   # 842EA46B1035BDA05A6B708D40E78BBF8138E6ECBF87C824E06AD7CA4A646A37
```

`possession_v9_standalone.py` is the same v9 in one file: every function it needs from
`model_v2.py` ... `model_v9.py` is copied into it, so it runs with only numpy, the weights in
`model_v9_params.json` and the organisers' loaders (`games.py`, `starter/`, `evaluation/`).
It writes `evaluation/results/v9_standalone.csv` (or `--out path.csv`), which is byte-identical to
`v9.csv`: both scripts gave the same SHA-256 on external games and on the competition match
(`A5F2DDEB…`).

```powershell
python previous_versions/possession_v9_standalone.py
Get-FileHash evaluation\results\v9_standalone.csv   # same hash as v9.csv
```

### v9.1 (previous submission, same features as v9)

v9.1 changes one training setting: the team regressions weigh frames within 75 frames (3 s) of a
change of possession, instead of 25 (1 s). Features, Smart Tagging and decoding costs (16 / 35) are
the same as v9; only the weights in `model_v9_1_params.json` differ.

On the 8 external games (leave one game out) it beats v9 by +0.0066 (22 of 32 game x variant pairs);
with the validation variants rebuilt from two other seeds, +0.0075 (27/32) and +0.0078 (25/32).
Held-out spellF1: noisy 0.704, noisy with gaps 0.685, smooth 0.662, smooth with gaps 0.641.

```powershell
python previous_versions/possession_v9_standalone.py --params model_v9_1_params.json --out evaluation\results\v9_1.csv
# retrain: python training/train_final.py VALIDATION_ROOT --version v9 --name v9.1 --transition-window 75
```

### v10 (fallback): v9.1 averaged with a sequence model

v10 keeps v9.1 and adds a bidirectional GRU (two layers, 64 units per direction, 135k parameters,
`model_v10_seq.npz`) that reads the same per-frame features at 5 Hz, forwards and backwards over each
half. Its log-probabilities for H, A and D are averaged with v9.1's, and the same Viterbi editor
(16 / 35) decodes the result. Inputs are standardised with the training statistics, columns that never
varied in training are ignored and values are clipped to +/-6 spreads, so unusual values in a new
match cannot saturate the network. Trained on the 8 external games only (`training/train_seq.py`);
inference is plain numpy inside `possession_v9_standalone.py`.

| Check (spellF1, mean) | v9.1 | v10 | v10 better on |
|---|---|---|---|
| 8 external games x 4 variants, leave one game out | 0.673 | 0.705 (+0.032) | 31 / 32 |
| same, variants rebuilt with seed +1000 | 0.671 | 0.694 (+0.023) | 26 / 32 |
| same, variants rebuilt with seed +2000 | 0.674 | 0.706 (+0.032) | 32 / 32 |
| 20 SkillCorner games (other provider, other labels), trained on the 8 | 0.448 | 0.456 (+0.008) | 13 / 20 |

```powershell
python previous_versions/possession_v10.py            # writes evaluation/results/v10.csv
python diagnostics/feature_shift.py # how far the match's features fall outside the training range
```


### v12 (fallback): v11 plus two networks trained at several paces

The competition tracking moves about 20% slower than the training games (players' speed towards the
attacked goal). v12 keeps v11's three networks and adds two more, trained in the same way on the 8
external games played at normal pace, 20% slower and 25% faster (tempo copies resampled in time). The
five networks' probabilities are averaged; judges, average with them and editor (16 / 35) unchanged.
Weights in `model_v12_seq.npz` (prefixes `n0__` to `n4__`); `previous_versions/possession_v12_standalone.py` is the
whole pipeline in one file.

| Check (spellF1, mean) | v11 | v12 | v12 better on |
|---|---|---|---|
| 8 external games x 4 variants, leave one game out | 0.7178 | 0.7206 (+0.0028) | 18 / 32 |
| same, variants rebuilt with seed +1000 | 0.7178 | 0.7196 (+0.0018) | 18 / 32 |
| same, variants rebuilt with seed +2000 | 0.7180 | 0.7199 (+0.0020) | 20 / 32 |
| same 32 cases played 20% slower (like the competition match) | 0.7054 | 0.7131 (+0.0077) | 28 / 32 |
| 20 SkillCorner games (other provider, other labels) | 0.4662 | 0.4681 (+0.0019) | 12 / 20 |

v12 is never below v11; its gain at normal pace is under the usual +0.005 rule, and it was chosen for
the slower pace measured in the competition tracking. Two identical extra networks (no tempo copies)
gave +0.0014 at normal pace and -0.0005 when slower, so the gain comes from the tempo copies.

```powershell
python previous_versions/possession_v12_standalone.py
Get-FileHash evaluation\results\v12_standalone.csv   # 5D36670AA83E509255C64603F8F2E5160FAD789EA796F59ACA0EC541B28C7FEE
```

Training the two extra networks (byte-identical to the shipped ones, checked for seed 10):

```bash
python training/train_seq.py validation --seed 10 --epochs 8 --tempos 0.8,1.25 --out cache/seq_a10.npz
python training/train_seq.py validation --seed 11 --epochs 8 --tempos 0.8,1.25 --out cache/seq_a11.npz
python training/build_v11_seq.py previous_versions/model_v10_seq.npz cache/seq_s1.npz cache/seq_s2.npz cache/seq_a10.npz cache/seq_a11.npz --out model_v12_seq.npz
```

The per-possession quality (`--quality`) and the gate's bands were calibrated with v11; they apply to
v12 unchanged (same judges, same editor, two more networks in the average).

### v11 (fallback): v10 with the average of three networks

v11 is v10 with one change: instead of one sequence network it averages the probabilities of three,
trained in exactly the same way on the same external data with different random seeds (seed 0 is the
v10 network, 12 epochs; seeds 1 and 2 were trained with 8 epochs, as they were validated). The three
are packed in `model_v11_seq.npz` (prefixes `n0__`, `n1__`, `n2__`, each with its own input mean and
spread). Everything else (v9.1 judges, the average with them, the Viterbi editor 16 / 35) is unchanged.
`possession_v11_standalone.py` is the whole pipeline in one file, like the v10 one; with only the v10
network in the weights file it gives the same CSV as v10.

| Check (spellF1, mean) | v10 | v11 | v11 better on |
|---|---|---|---|
| 8 external games x 4 variants, leave one game out | 0.7052 | 0.7178 (+0.0126) | 26 / 32 |
| same, variants rebuilt with seed +1000 | | +0.0132 | 28 / 32 |
| same, variants rebuilt with seed +2000 | | +0.0132 | 24 / 32 |
| 20 SkillCorner games (other provider, other labels), trained on the 8 | 0.4559 | 0.4662 (+0.0103) | 16 / 20 |

A single extra network (8 epochs) alone adds only +0.0007 and +0.0043: the gain comes from averaging.

```powershell
python previous_versions/possession_v11_standalone.py
Get-FileHash evaluation\results\v11_standalone.csv   # 47D4A7D4A9E7424E12DB7C0E12845C0E02AE574FB163CE46439710FCE902CB1A
```

Before submitting, a quality gate checks that the file looks like a plausible match (no ground
truth involved). `--quality` also writes the Phred-like quality of each possession to a separate
file (`evaluation/results/v11_quality.csv`, git-ignored like every labels file); the submission CSV
and its hash do not change.

```powershell
python previous_versions/possession_v11_standalone.py --quality
python diagnostics/possession_dna.py     # ends with PASS or FAIL
```

The gate fails when 2 or more of 14 "DNA" measures of the label sequence (letters per 90 minutes,
motifs, transitions, dead-ball share, median possession) fall outside what v11 writes on the
external games, or when the share of possessions with quality Q < 5 is above its external maximum.
Calibrated on 25 kinds of deliberate damage to the external tracking: no false alarm on 74
harmless runs, 22 of 63 harmful runs caught (every run without the ball). It is a check that the
pipeline did not break, not a guarantee of quality: moderate damage still produces a
plausible-looking match.

Training the networks (JAX, CPU is enough; each takes about 10 to 15 minutes). Training is deterministic:
retraining seed 1 with the command below gave a network byte-identical to the one in `model_v11_seq.npz`.

```bash
pip install "jax[cpu]" optax
python training/train_seq.py validation                                   # seed 0, 12 epochs -> previous_versions/model_v10_seq.npz
python training/train_seq.py validation --seed 1 --epochs 8 --out cache/seq_s1.npz
python training/train_seq.py validation --seed 2 --epochs 8 --out cache/seq_s2.npz
python training/build_v11_seq.py previous_versions/model_v10_seq.npz cache/seq_s1.npz cache/seq_s2.npz   # -> previous_versions/model_v11_seq.npz
```
