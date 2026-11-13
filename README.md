# Frame-level ball possession: More than a Hack 2027, Phase 1

Submission repository for the Metrica Sports / FC Barcelona possession challenge. For every frame of
the tactical video of FC Barcelona vs Athletic Club it outputs `H`, `A` or `D` (`X` outside the two
halves). The original challenge README is kept as [README_challenge.md](README_challenge.md).

## Approach in one paragraph

A hybrid hidden Markov model. Two logistic regressions estimate, frame by frame, the probability that
the ball is out of play and, if it is alive, the probability that each team has it. Their evidence
comes from the automatic tracking: where the ball and the players are, how fast the players move,
how long the ball has been missing and where it was last seen, detected touches of the ball and how
long a team has kept a player on it, the ball's flights between touches (passes, clearances and
interceptions, and which team is about to receive the ball), the shape of each team (depth, width,
compactness, players past the ball, how fast they close on it), how both teams turn round when
the ball changes hands, and the Smart Tagging phases and kick-offs. The team regressions are fitted
with extra weight on the 3 s around each change of possession, where the errors are. A small sequence
model (a bidirectional GRU) reads the same features over each whole half, forwards and backwards, and its
probabilities are averaged with the regressions'. A Viterbi decoder,
run separately on each half, turns these probabilities into the most likely sequence of possessions
and stoppages, charging a cost for every change so that a touch without control does not flip
possession. **All weights and costs were learned on 8 external matches**; the competition match and
its ground truth were never used for training, tuning or model selection.

## Reproducing the submission

Requirements: Python 3.12, `pip install -r requirements-lock.txt` (numpy and pandas only).
The match data is not in this repository; unpack the bundle into `data/` as described in
[DATA.md](DATA.md), or point `MTAH_DATA_DIR` at it.

```bash
python cache_atd.py        # reads the tactical ATD once into cache/atd_tactical.npz (+ sanity checks)
python possession_v12_standalone.py   # writes evaluation/results/v12_standalone.csv and validates it (the submission)
```

The submission is v12 (below), SHA-256
`5D36670AA83E509255C64603F8F2E5160FAD789EA796F59ACA0EC541B28C7FEE`; it passes the quality gate. v11 stays as the fallback, SHA-256
`47D4A7D4A9E7424E12DB7C0E12845C0E02AE574FB163CE46439710FCE902CB1A`.

v10: `possession_v10.py` runs the single-file pipeline
(`possession_v9_standalone.py`) with the regression weights in `model_v9_1_params.json` and the
sequence model in `model_v10_seq.npz`, all in numpy. The output is deterministic: the same data and
code produce the same file, SHA-256
`842EA46B1035BDA05A6B708D40E78BBF8138E6ECBF87C824E06AD7CA4A646A37`. Earlier versions can still be
reproduced with `python possession_v9_1.py` (SHA-256 `FE39AA79…B6F5B`), `python possession_v9.py`
(SHA-256 `A5F2DDEB…F7529CC`), `possession_v8.py`,
`possession_v6.py` and `possession_v4.py`.

### Checking the result on another machine

The submission must come out identical on any machine. On a second computer (Windows PowerShell
shown; the same steps work on macOS / Linux):

```powershell
git clone https://github.com/alfredordzz/mtah-possession.git
cd mtah-possession
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
# unpack the organisers' bundle into data\ (see DATA.md) or set $env:MTAH_DATA_DIR
python cache_atd.py
python possession_v10.py
Get-FileHash evaluation\results\v10.csv     # 842EA46B1035BDA05A6B708D40E78BBF8138E6ECBF87C824E06AD7CA4A646A37
```

If the hashes differ, `python diagnostics/compare_submissions.py first.csv second.csv` reports how
many frames differ (counts only).

### Single-file version of the submission

`possession_v10_standalone.py` is the whole v10 in one file, from the raw tracking to the CSV. It
does not import any of our other scripts: the tracking reader of `cache_atd.py` and every function
of `possession_v9_standalone.py` that v10 uses are copied into it. It needs only numpy, the weights
in `model_v9_1_params.json` and `model_v10_seq.npz`, and the organisers' loaders (`games.py`,
`starter/`, `evaluation/`). On the first run it reads the raw tracking into `cache/` by itself
(`--rebuild-cache` forces it). It writes `evaluation/results/v10_standalone.csv` (or
`--out path.csv`), byte-identical to `v10.csv`.

```powershell
python possession_v10_standalone.py
Get-FileHash evaluation\results\v10_standalone.csv   # 842EA46B1035BDA05A6B708D40E78BBF8138E6ECBF87C824E06AD7CA4A646A37
```

`possession_v9_standalone.py` is the same v9 in one file: every function it needs from
`model_v2.py` ... `model_v9.py` is copied into it, so it runs with only numpy, the weights in
`model_v9_params.json` and the organisers' loaders (`games.py`, `starter/`, `evaluation/`).
It writes `evaluation/results/v9_standalone.csv` (or `--out path.csv`), which is byte-identical to
`v9.csv`: both scripts gave the same SHA-256 on external games and on the competition match
(`A5F2DDEB…`).

```powershell
python possession_v9_standalone.py
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
python possession_v9_standalone.py --params model_v9_1_params.json --out evaluation\results\v9_1.csv
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
python possession_v10.py            # writes evaluation/results/v10.csv
python diagnostics/feature_shift.py # how far the match's features fall outside the training range
```

### v12 (the submission): v11 plus two networks trained at several paces

The competition tracking moves about 20% slower than the training games (players' speed towards the
attacked goal). v12 keeps v11's three networks and adds two more, trained in the same way on the 8
external games played at normal pace, 20% slower and 25% faster (tempo copies resampled in time). The
five networks' probabilities are averaged; judges, average with them and editor (16 / 35) unchanged.
Weights in `model_v12_seq.npz` (prefixes `n0__` to `n4__`); `possession_v12_standalone.py` is the
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
python possession_v12_standalone.py --quality
Get-FileHash evaluation\results\v12_standalone.csv   # 5D36670AA83E509255C64603F8F2E5160FAD789EA796F59ACA0EC541B28C7FEE
python diagnostics/possession_dna.py evaluation/results/v12_standalone.csv --quality evaluation/results/v12_quality.csv
```

Training the two extra networks (byte-identical to the shipped ones, checked for seed 10):

```bash
python training/train_seq.py validation --seed 10 --epochs 8 --tempos 0.8,1.25 --out cache/seq_a10.npz
python training/train_seq.py validation --seed 11 --epochs 8 --tempos 0.8,1.25 --out cache/seq_a11.npz
python training/build_v11_seq.py model_v10_seq.npz cache/seq_s1.npz cache/seq_s2.npz cache/seq_a10.npz cache/seq_a11.npz --out model_v12_seq.npz
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
python possession_v11_standalone.py
Get-FileHash evaluation\results\v11_standalone.csv   # 47D4A7D4A9E7424E12DB7C0E12845C0E02AE574FB163CE46439710FCE902CB1A
```

Before submitting, a quality gate checks that the file looks like a plausible match (no ground
truth involved). `--quality` also writes the Phred-like quality of each possession to a separate
file (`evaluation/results/v11_quality.csv`, git-ignored like every labels file); the submission CSV
and its hash do not change.

```powershell
python possession_v11_standalone.py --quality
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
python training/train_seq.py validation                                   # seed 0, 12 epochs -> model_v10_seq.npz
python training/train_seq.py validation --seed 1 --epochs 8 --out cache/seq_s1.npz
python training/train_seq.py validation --seed 2 --epochs 8 --out cache/seq_s2.npz
python training/build_v11_seq.py model_v10_seq.npz cache/seq_s1.npz cache/seq_s2.npz   # -> model_v11_seq.npz
```

## Reproducing the training (external data only)

Requirements: `pip install -r training/requirements-training.txt`.

1. **Metrica Sports open sample data** (3 matches, tracking and events):
   `git clone https://github.com/metrica-sports/sample-data`
2. **IDSSE** (5 Bundesliga matches, DFL, CC BY 4.0), downloaded for example with
   `python -c "from huggingface_hub import snapshot_download; snapshot_download('pysport/idsse-data', repo_type='dataset', local_dir='idsse')"`
   and converted to compact arrays:
   `python training/idsse_convert.py idsse idsse_npz`
3. Build the validation variants and train:

```bash
python training/build_validation.py sample-data/data idsse_npz validation
python training/train_final.py validation                 # rewrites model_v9_params.json
python training/train_final.py validation --version v8    # rewrites model_v8_params.json
python training/train_final.py validation --version v6    # rewrites model_v6_params.json
python training/train_final.py validation --version v4    # rewrites model_v4_params.json
```

`train_final.py` first scores every match with a model fitted on the other seven (leave one game
out) to choose the Viterbi costs, then fits the final model on all eight.

### How the external data was made to look like the competition tracking

The external tracking is cleaner than the automatic tracking of the competition match, so it is
degraded before training. Every choice below comes from aggregate statistics of the provided input
(scripts in `diagnostics/`), never from labels:

* **Ball gaps**: the competition tracking loses the ball in few, long gaps (median about 11 s),
  mostly during stoppages (missing 70% of the time when players walk, 11% when they run). Gaps are
  simulated the same way, plus a "hard" variant with extra long in-play gaps.
* **Tracker profile**: the competition ball track is smooth and players appear about 1.9 m from the
  ball. A "smooth" profile (smoothed, slightly lagging ball; slowly drifting player offsets)
  reproduces those statistics; a "noisy" profile covers the opposite case. Training mixes both.
* **Controlled possession (IDSSE)**: the DFL possession flag changes at nearly every contact. Following
  the challenge definition (a touch without control is not a change of possession), a spell shorter
  than 1.5 s that interrupts the previous team's possession is returned to that team.

## Validation results

spellF1 (the ranking metric, `evaluation/possession_metrics.py`), mean over the 8 external matches,
each scored by a model that never saw it:

| | noisy | noisy + gaps | smooth (competition-like) | smooth + gaps |
|---|---|---|---|---|
| Baseline (nearest player, +/-50 frames) | 0.335 | 0.316 | 0.282 | 0.252 |
| v4 | 0.637 | 0.613 | 0.590 | 0.570 |
| v6 | 0.665 | 0.642 | 0.618 | 0.601 |
| v8 | 0.675 | 0.656 | 0.635 | 0.620 |
| v9 | 0.698 | 0.678 | 0.655 | 0.635 |
| v9.1 | 0.704 | 0.685 | 0.662 | 0.641 |
| v10 | 0.727 | 0.721 | 0.698 | 0.674 |

v6 adds the ball-flight features to v4 (higher than v4 on 30 of the 32 match x variant pairs); v8
adds team shape and counter-press to v6 (higher on 28 of 32, and on all 8 matches of the
"smooth + gaps" variant); v9 adds the collective movement reversal and fits the team models with
three times the weight on frames within 1 s of a change of possession (higher than v8 on all 32); v9.1 widens that window to 3 s (+0.007 over v9 averaged over three rebuilds of the validation variants with different seeds, 74 of 96 pairs); v10 averages v9.1 with a bidirectional GRU over the same features (+0.032, higher on 31 of 32; +0.023 and +0.032 with the variants rebuilt from two other seeds; +0.008 on 20 SkillCorner games from another provider). A ceiling analysis guided the order: with a perfect dead-ball call v6
would gain 0.04, with a perfect team call 0.33, so the later versions only add team evidence.
Ideas that did not beat the version of their time and were left out: decision trees instead of
logistic regressions, set-piece features, temperature calibration, a semi-Markov decoder with
learned possession durations, touch-outcome / carry / pressure-count features (v7, +0.006),
moving each change to the nearest touch after decoding, a second-stage judge of short possessions,
and a switch cost that depends on the context of play. They re-arrange evidence the model already
has; possessions shorter than 3 s (a quarter of all) leave almost no trace in this tracking.
A learning curve (1 to 7 training matches) flattens from 5 matches on, so more external matches
would mostly sharpen the validation rather than the model.

For reference, the official baseline scores 0.2433 on the full competition match.

## Interactive explainer (docs/)

`docs/index.html` is a self-contained page that shows how the submission (v11) decides, step by step: two 90 s stretches
of real external matches (IDSSE, CC BY 4.0, scored with models trained without them), a synthetic play,
a lab with controlled experiments and 24 stress tests on the 8 external games, and a tab with
descriptive statistics of v11's own answers for the competition match (aggregates from
`diagnostics/describe_submission.py`; no ground truth and no other data about the match), and a tab on what changed from v10 to v11 and the ideas tested afterwards. A seventh tab reads each match as a string of letters (L home, V away, M dead ball) and aligns the model's string with the true one, like two DNA sequences, to type its errors and count recurring patterns; `python diagnostics/possession_dna.py` prints the same aggregates for a submission file. Open it in a browser, or publish it with GitHub Pages
(Settings > Pages > Deploy from a branch > `main` / `/docs`) once the repository is public.

## Repository layout

| Path | Purpose |
|---|---|
| `cache_atd.py` | Reads the tactical ATD into arrays in metres, with checks against the published numbers |
| `possession_v12_standalone.py` | The submission (v12) in one file, raw tracking to CSV |
| `model_v12_seq.npz` | The five networks of v12 (prefixes `n0__` to `n4__`) |
| `possession_v11_standalone.py` | v11 (fallback) in one file |
| `model_v11_seq.npz` | The three networks of v11 (prefixes `n0__`, `n1__`, `n2__`) |
| `possession_v10.py` | v10 (fallback) |
| `possession_v10_standalone.py` | The whole v10 in one file, raw tracking to CSV (byte-identical to `possession_v10.py`) |
| `possession_v9_1.py` | v9.1 (previous submission) |
| `possession_v9_standalone.py` | The whole v9 / v9.1 pipeline in one file (needs only numpy and the organisers' loaders) |
| `possession_v9.py` | v9 (previous submission) |
| `model_v2.py`, `model_v3.py`, `model_v4.py`, `model_v6.py`, `model_v8.py`, `model_v9.py` | The model (each version builds on the features of the previous ones) |
| `model_v9_1_params.json` | Regression weights and decoding costs (v9.1 and v10) |
| `model_v10_seq.npz` | Weights of v10's sequence model, with the training mean and spread of its inputs |
| `model_v9_params.json` | Weights of v9 |
| `training/` | Conversion of the external data, validation variants, training |
| `diagnostics/` | Automatic checks on the provided data; they print aggregates only |
| `possession_v1.py`, `possession_v3.py`, `possession_v4.py`, `possession_v6.py`, `possession_v8.py`, `model_v3_params.json`, `model_v4_params.json`, `model_v6_params.json`, `model_v8_params.json` | Earlier versions, kept for the record |

## Rules and data

* No hand labels and no human-produced data about the competition match are used anywhere.
* The public 5-minute ground truth was used only to check that files are well formed.
* The competition data is not in this repository (`data/` and `cache/` are git-ignored).
* Large language models were used as coding assistants; no competition data was uploaded to them.

## External data and citation

* Metrica Sports sample data, https://github.com/metrica-sports/sample-data
* Bassek, M. et al. (2025). *An integrated dataset of spatiotemporal and event data in elite
  soccer.* Scientific Data. Data: https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0).
