# Frame-level ball possession: More than a Hack 2027, Phase 1

Submission repository for the Metrica Sports / FC Barcelona possession challenge. For every frame of
the tactical video of FC Barcelona vs Athletic Club it outputs `H`, `A` or `D` (`X` outside the two
halves). The original challenge README is kept as [README_challenge.md](README_challenge.md).

## Approach in one paragraph

A hybrid hidden Markov model. Two logistic regressions estimate, frame by frame, the probability that
the ball is out of play and, if it is alive, the probability that each team has it. Their evidence
comes from the automatic tracking: where the ball and the players are, how fast the players move,
how long the ball has been missing and where it was last seen, detected touches of the ball and how
long a team has kept a player on it, and the Smart Tagging phases and kick-offs. A Viterbi decoder,
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
python possession_v4.py    # writes evaluation/results/v4.csv and validates it
```

`possession_v4.py` applies `model_v4.py` with the weights in `model_v4_params.json`. The output is
deterministic: the same data and code produce the same file.

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
python training/train_final.py validation          # rewrites model_v4_params.json
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
| **v4 (this submission)** | **0.637** | **0.613** | **0.590** | **0.570** |

For reference, the official baseline scores 0.2433 on the full competition match.

## Repository layout

| Path | Purpose |
|---|---|
| `cache_atd.py` | Reads the tactical ATD into arrays in metres, with checks against the published numbers |
| `possession_v4.py` | Produces the submission file |
| `model_v2.py`, `model_v3.py`, `model_v4.py` | The model (v4 builds on the features of v2 and v3) |
| `model_v4_params.json` | Learned weights and decoding costs |
| `training/` | Conversion of the external data, validation variants, training |
| `diagnostics/` | Automatic checks on the provided data; they print aggregates only |
| `possession_v1.py`, `possession_v3.py`, `model_v3_params.json` | Earlier versions, kept for the record |

## Rules and data

* No hand labels and no human-produced data about the competition match are used anywhere.
* The public 5-minute ground truth was used only to check that files are well formed.
* The competition data is not in this repository (`data/` and `cache/` are git-ignored).
* Large language models were used as coding assistants; no competition data was uploaded to them.

## External data and citation

* Metrica Sports sample data, https://github.com/metrica-sports/sample-data
* Bassek, M. et al. (2025). *An integrated dataset of spatiotemporal and event data in elite
  soccer.* Scientific Data. Data: https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0).
