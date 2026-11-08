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
with extra weight on the frames around changes of possession, where the errors are. A Viterbi decoder,
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
python possession_v9.py    # writes evaluation/results/v9.csv and validates it (the submission)
```

`possession_v9.py` applies `model_v9.py` with the weights in `model_v9_params.json`. The output is
deterministic: the same data and code produce the same file. Earlier versions can still be
reproduced with `python possession_v8.py`, `possession_v6.py` and `possession_v4.py`.

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
| **v9 (this submission)** | **0.698** | **0.678** | **0.655** | **0.635** |

v6 adds the ball-flight features to v4 (higher than v4 on 30 of the 32 match x variant pairs); v8
adds team shape and counter-press to v6 (higher on 28 of 32, and on all 8 matches of the
"smooth + gaps" variant); v9 adds the collective movement reversal and fits the team models with
three times the weight on frames within 1 s of a change of possession (higher than v8 on all 32). A ceiling analysis guided the order: with a perfect dead-ball call v6
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

## Repository layout

| Path | Purpose |
|---|---|
| `cache_atd.py` | Reads the tactical ATD into arrays in metres, with checks against the published numbers |
| `possession_v9.py` | Produces the submission file |
| `model_v2.py`, `model_v3.py`, `model_v4.py`, `model_v6.py`, `model_v8.py`, `model_v9.py` | The model (each version builds on the features of the previous ones) |
| `model_v9_params.json` | Learned weights and decoding costs |
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
