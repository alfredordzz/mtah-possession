# Frame-level ball possession: More than a Hack 2027, Phase 1

**Team: Sultanes del Norte.** Technical document: [Sultanes_del_Norte.pdf](Sultanes_del_Norte.pdf). Submission: v13.

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
python possession_v13_standalone.py   # reads the tracking once into cache/, writes evaluation/results/v13_standalone.csv and validates it (the submission)
```

The submission is v13 (below), SHA-256
`F2386073426F2BFEAB551DED835140FE26741A59C794B20790ECB384A042363B` (first run 13 nov 2026, reproduced identically by a second run).
v12 stays as the fallback (`python previous_versions/possession_v12_standalone.py`), SHA-256
`5D36670AA83E509255C64603F8F2E5160FAD789EA796F59ACA0EC541B28C7FEE`; v11 before it, SHA-256
`47D4A7D4A9E7424E12DB7C0E12845C0E02AE574FB163CE46439710FCE902CB1A`.

Everything the submission needs is in the root: `possession_v13_standalone.py`, `model_v12_seq.npz` (the five networks, shared with v12),
`model_v9_1_params.json` (the judges), `model_v11_phred.json` (quality, only with `--quality`) and the
organisers' `games.py`, `starter/` and `evaluation/`. Every earlier version (v1 to v12, with its
weights and `cache_atd.py`) is in `previous_versions/` and still runs from the repository root,
e.g. `python previous_versions/possession_v11_standalone.py`.

Older versions are listed in [previous_versions/HISTORY.md](previous_versions/HISTORY.md).

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
python possession_v13_standalone.py
Get-FileHash evaluation\results\v13_standalone.csv     # F2386073426F2BFEAB551DED835140FE26741A59C794B20790ECB384A042363B
```

If the hashes differ, `python diagnostics/compare_submissions.py first.csv second.csv` reports how
many frames differ (counts only).


## The submission: v13 (v12 also reading the match sped up 1.25x)

The competition tracking moves about 20% slower than the training games. v12 handled that with two
networks trained on slowed and sped-up copies; v13 adds the same idea at prediction time, with no
retraining: the five v12 networks read the match as is and once more resampled 1.25x faster (the pitch
features recomputed on the faster copy), the second reading's probabilities are interpolated back to the
original frames, and the two readings are averaged. Judges, average with them and editor (16 / 35)
unchanged; same model files as v12. Nothing is tuned on the match.

| Check (spellF1, mean) | v12 | v13 | v13 better on |
|---|---|---|---|
| 8 external games x 4 variants played 20% slower, leave one game out | 0.7131 | 0.7195 (+0.0064) | 24 / 32 |
| same, variants rebuilt with seed +1000 (not used to choose v13) | 0.7112 | 0.7184 (+0.0072) | 28 / 32 |
| 8 external games x 4 variants, normal pace | 0.7206 | 0.7223 (+0.0017) | 21 / 32 |
| same, variants rebuilt with seed +1000 | 0.7190 | 0.7189 (-0.0002) | 17 / 32 |
| 20 SkillCorner games (other provider, final models) | 0.4681 | 0.4696 (+0.0014) | 11 / 20 |

How it was chosen: the planned variant (reading at 0.8x, 1x and 1.25x) was flat (+0.0006 normal,
-0.0010 slow); reading slower hurt (-0.0107 slow, 3 / 32); reading faster was the only one that helped.
Because it was picked after seeing those 32 cases, it was confirmed on the seed +1000 games, which were
not used to choose it. Same criterion as v12: passes the rule on the slow cases, ties at normal pace.
A dilated convolutional network (TCN) as a sixth network was also tried and dropped (-0.0002 normal,
+0.0012 slow).

```powershell
python possession_v13_standalone.py --quality
Get-FileHash evaluation\results\v13_standalone.csv
python diagnostics/possession_dna.py evaluation/results/v13_standalone.csv --quality evaluation/results/v13_quality.csv
```

The per-possession quality (`--quality`) and the gate's bands were calibrated with v11; they apply to
v13 unchanged.


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
python training/train_final.py validation                 # rewrites previous_versions/model_v9_params.json
python training/train_final.py validation --version v8    # rewrites previous_versions/model_v8_params.json
python training/train_final.py validation --version v6    # rewrites previous_versions/model_v6_params.json
python training/train_final.py validation --version v4    # rewrites previous_versions/model_v4_params.json
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

`docs/index.html` is a self-contained page that shows how the submission (v13) decides, step by step: two 90 s stretches
of real external matches (IDSSE, CC BY 4.0, scored with models trained without them), a synthetic play,
a lab with controlled experiments and 24 stress tests on the 8 external games, and a tab with
descriptive statistics of v13's own answers for the competition match (aggregates from
`diagnostics/describe_submission.py`; no ground truth and no other data about the match), and a tab on what changed from v10 to v11 and the ideas tested afterwards. A seventh tab reads each match as a string of letters (L home, V away, M dead ball) and aligns the model's string with the true one, like two DNA sequences, to type its errors and count recurring patterns; `python diagnostics/possession_dna.py` prints the same aggregates for a submission file. Open it in a browser, or publish it with GitHub Pages
(Settings > Pages > Deploy from a branch > `main` / `/docs`) once the repository is public.

## Repository layout

| Path | Purpose |
|---|---|
| `possession_v13_standalone.py` | The submission (v13) in one file, raw tracking to CSV |
| `model_v12_seq.npz` | The five networks of v12, also used by v13 (prefixes `n0__` to `n4__`) |
| `model_v9_1_params.json` | The two judges (regressions), touch/flight thresholds and decoding costs (v9.1 to v13) |
| `model_v11_phred.json` | Weights of the per-possession quality score (`--quality`, v11 to v13) |
| `games.py`, `starter/`, `evaluation/` | The organisers' loaders, baseline and metric |
| `training/` | Conversion of the external data, validation variants, training |
| `diagnostics/` | Automatic checks on the provided data and on a submission; they print aggregates only |
| `docs/index.html` | «La v13 por dentro», the explanatory page |
| `previous_versions/` | v1 to v12 with their weights (`possession_v12_standalone.py` is the fallback; it uses `model_v12_seq.npz` from the root), the older model modules `model_v2.py` ... `model_v9.py` and `cache_atd.py`; run them from the repository root |

## Rules and data

* No hand labels and no human-produced data about the competition match are used anywhere.
* The public 5-minute ground truth was used only to check that files are well formed.
* The competition data is not in this repository (`data/` and `cache/` are git-ignored).
* Large language models were used as coding assistants; no competition data was uploaded to them.

## External data and citation

* Metrica Sports sample data, https://github.com/metrica-sports/sample-data
* Bassek, M. et al. (2025). *An integrated dataset of spatiotemporal and event data in elite
  soccer.* Scientific Data. Data: https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0).


The full history (v4 to v12, with their hashes and validation tables) is in [previous_versions/HISTORY.md](previous_versions/HISTORY.md).
