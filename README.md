# time-series-forecasting
This is a repo for assignment "Time Series Forecasting".

## Conventions

**Data columns**
- `datetime` (timezone-naive pandas datetime, hourly cadence, sorted ascending)
- `pm2_5` (float; target)
- Other meteorological columns kept as in source; all numeric cast to float32 where possible.

**File naming**
- Notebooks: `NN_topic.ipynb` (e.g., `10_model_lstm_baseline.ipynb`)
- Outputs: `outputs/models/<model>_<run_id>.keras`, `outputs/submissions/submission_YYYYMMDD_HHMM.csv`

**Randomness & reproducibility**
- Global seed: **42**
- We seed Python, NumPy, and TensorFlow in one place (`src/utils.py::seed_all`).
- CV splits are time-blocked, defined explicitly in notebooks and report.

**Code style**
- Python ≥3.11, type hints required
- Black formatting (line length 100), Ruff linting, isort imports (Black profile)
- Docstrings: NumPy style

**Logging & errors**
- Minimal structured logging via `utils.get_logger()`
- Raise explicit `ValueError`/`RuntimeError` on misuse (e.g., leakage checks)

**Plotting**
- Matplotlib defaults; every figure saved to `outputs/figures/` and captioned with “why it matters”


Awesome—let’s turn this into a clean, rubric-aligned plan you can drop straight into a GitHub Project. No coding yet; just structure, tasks, and acceptance criteria.

# 1) Repo & folder structure (professional, modular)

```
air-quality-forecasting/
├─ README.md
├─ REPORT.md                  # final report (aligns to rubric sections)
├─ environment.yml            # or requirements.txt (pin TF, pandas, etc.)
├─ .gitignore
├─ data/
│  ├─ raw/                    # train.csv, test.csv, sample_submission.csv
│  ├─ interim/                # cleaned/filled versions
│  └─ processed/              # windowed tensors, scalers, metadata
├─ notebooks/
│  ├─ 00_eda_preprocessing.ipynb
│  ├─ 10_model_lstm_baseline.ipynb
│  ├─ 11_model_gru_baseline.ipynb
│  ├─ 12_model_lstm_stacked.ipynb
│  ├─ 19_submit_inference.ipynb      # loads best weights → kaggle CSV
│  └─ 90_appendix_visuals.ipynb      # optional visuals for report
├─ src/
│  ├─ __init__.py
│  ├─ config.py                # paths, constants, seeds, feature flags
│  ├─ data_loading.py          # read, type-cast, datetime parsing
│  ├─ preprocessing.py         # missing-values, scaling, time features
│  ├─ windowing.py             # create (X,y) sequences safely (shift/roll)
│  ├─ models/
│  │   ├─ lstm.py              # build_lstm(config)
│  │   └─ gru.py               # build_gru(config)
│  ├─ train.py                 # train_loop(), callbacks, early stopping
│  ├─ evaluate.py              # RMSE/MAE/SMAPE, plots
│  ├─ inference.py             # recursive 1-step forecasting over test
│  ├─ metrics.py               # rmse(), smape(), etc.
│  ├─ plots.py                 # reusable plotting funcs
│  ├─ utils.py                 # seeds, logging, IO, timers
│  └─ experiment_log.py        # append results to experiments.csv
├─ experiments/
│  ├─ experiments.csv          # experiment table (15+ rows per rubric)
│  └─ configs/                 # JSON/YAML configs per run (optional)
├─ outputs/
│  ├─ figures/                 # EDA & result plots for report
│  ├─ models/                  # saved weights (best.ckpt / .keras)
│  └─ submissions/             # kaggle-ready csv files
└─ LICENSE (optional)
```

# 2) GitHub Project: Epics → Issues (with acceptance criteria)

## EPIC A — Repository scaffold & environment

* **A1: Initialize repo & structure**

  * *Tasks:* create folders above; add `.gitignore` (Python, notebooks, data).
  * *AC (acceptance criteria):* clean tree compiles; no tracked large data; paths centralized in `config.py`.
* **A2: Environment & reproducibility**

  * *Tasks:* `environment.yml` (Python 3.11; tensorflow 2.x; pandas; numpy; scikit-learn; matplotlib; seaborn (optional); jupyter; pyarrow; pydantic (optional)); seed utility.
  * *AC:* `pip/conda` install works; `python -c "import tensorflow as tf"` succeeds; seed function sets NumPy & TF seeds.

## EPIC B — EDA & preprocessing (rubric: Data Exploration 15 pts)

* **B1: Data audit**

  * *Tasks:* in `00_eda_preprocessing.ipynb`, load `train.csv/test.csv`; parse datetime; check continuity; summary stats; missingness map; target distribution; seasonality (hour/day/month) plots; correlations; ACF.
  * *AC:* notebook outputs saved to `outputs/figures/`; each plot captioned with “why it matters” per rubric.
* **B2: Preprocessing decisions**

  * *Tasks:* document handling of missing `pm2.5` (drop for training); feature scaling strategy (StandardScaler/MinMax) fitted on train only; `log1p(pm2.5)` decision & rationale.
  * *AC:* a short “Preprocessing Rationale” section in notebook + bullets copied to REPORT.md.
* **B3: Time features & exogenous prep**

  * *Tasks:* cyclic time features (sin/cos hour, day-of-year); preserve meteorological features; optional rolling means on exogenous (kept lightweight for RNN).
  * *AC:* `preprocessing.py` exposes `prepare_features(df, config)` with leakage-safe shifts.

## EPIC C — Sequence dataset & windowing (core to RNN/LSTM)

* **C1: Window function**

  * *Tasks:* in `windowing.py`, implement `make_windows(df, lookback=L, horizon=1, step=1)`; ensure strict past→future (`shift(1)` for target).
  * *AC:* unit-style sanity in notebook: last window aligns; no future leakage.
* **C2: Train/val time splits**

  * *Tasks:* blocked/rolling CV splits (e.g., 3 folds by month blocks); scaler fit per train fold and applied to val.
  * *AC:* fold summary table printed; no overlap between train and val in each fold.

## EPIC D — Baselines (sanity, not main)

* **D1: Naïve & seasonal-naïve**

  * *Tasks:* add tiny section in EDA notebook to compute RMSE for `y[t-1]` and `y[t-24]` on validation periods.
  * *AC:* numbers included in report; models intended as yardsticks only.

## EPIC E — Model notebooks (separate per your request; rubric: Model Design 15 pts)

* **E1: LSTM baseline (`10_model_lstm_baseline.ipynb`)**

  * *Design:* 1–2 LSTM layers (e.g., 64 units), dropout & recurrent\_dropout, Dense(1) head, MSE loss, Adam optimizer, gradient clipping, EarlyStopping + ReduceLROnPlateau.
  * *AC:* clear architecture diagram (Keras `plot_model` or drawn block), justification bullets, first RMSE reported.
* **E2: GRU baseline (`11_model_gru_baseline.ipynb`)**

  * *Design:* mirror LSTM config with GRU cells; compare params/perf.
  * *AC:* comparative table vs LSTM with narrative.
* **E3: Stacked LSTM (`12_model_lstm_stacked.ipynb`)**

  * *Design:* 2–3 LSTM layers (e.g., 128→64), LayerNorm (optional), tuned dropout.
  * *AC:* demonstrates whether depth helps; discuss over/underfitting signals.

> Each model notebook should:
>
> * Load prepared windows from `data/processed/` (not re-compute heavy steps),
> * Log hyperparams + fold RMSE to `experiments/experiments.csv`,
> * Save best weights to `outputs/models/…`.

## EPIC F — Experiments (rubric: ≥15 experiments table, 10 pts)

* **F1: Experiment plan**

  * *Tasks:* define grid of variations: lookback (72/168/336), batch size (32/64/128), units (32/64/128), layers (1/2/3), dropout (0/0.2/0.4), optimizer (Adam/AdamW), learning rate (1e-2/1e-3/5e-4), recurrent\_dropout (0/0.2), gradient\_clipnorm (1.0/5.0), normalization (Std vs MinMax), `log1p` on target (on/off).
  * *AC:* a markdown “Experiment Matrix” in REPORT.md + checkboxes.
* **F2: Experiment logging**

  * *Tasks:* `experiment_log.py` appends rows with columns: `run_id, date, model, lookback, features, layers, units, dropout, rec_dropout, optimizer, lr, batch, epochs, clipnorm, scaler, target_transform, cv_rmse_mean, cv_rmse_std, notes`.
  * *AC:* `experiments.csv` contains ≥15 distinct rows with varied params.

## EPIC G — Evaluation & analysis (rubric: Results & Discussion 10 pts)

* **G1: Metrics & plots**

  * *Tasks:* implement `rmse()` and **define RMSE with formula** in report; add predictions vs actual plots on val folds; residual distributions; error by hour-of-day and season.
  * *AC:* figures saved; discussion includes overfitting/underfitting cues; mention vanishing/exploding gradients & mitigations (LSTM gates, gradient clipping, careful LR).
* **G2: Final model selection**

  * *Tasks:* choose best config by mean CV RMSE; justify choice vs second-best; list trade-offs.
  * *AC:* 1–2 clear paragraphs in REPORT.md.

## EPIC H — Inference & Kaggle submission

* **H1: Deterministic inference**

  * *Tasks:* `inference.py` runs recursive 1-step over full test horizon, updating the window with each prediction; invert `log1p`; unscale; clip to non-negative reasonable max.
  * *AC:* `19_submit_inference.ipynb` writes `outputs/submissions/submission_YYYYMMDD_HHMM.csv` matching `sample_submission.csv` exactly (IDs, order).
* **H2: Submission QA**

  * *Tasks:* assert row count matches; no NaNs; spot-check first/last 5 timestamps; histogram sanity.
  * *AC:* pass a “Submission Checklist” cell before export.

## EPIC I — Report & citations (rubric: Approach, Results, Conclusion, Citations)

* **I1: REPORT.md scaffold (IEEE citations)**

  * *Sections:* Introduction; Data Exploration & Preprocessing; Model Design (with diagram); Experiments Table (≥15); Results & Discussion (RMSE definition + formula); Conclusion & Next steps; **References (IEEE style)**; **GitHub link**.
  * *AC:* each figure/table referenced in text; citations added (e.g., Hochreiter & Schmidhuber 1997; Bengio et al. on vanishing gradients; Keras/TensorFlow docs).
* **I2: Originality statement**

  * *Tasks:* include short statement on originality (no AI-generated report text; tools only for brainstorming/code; all sources cited).
  * *AC:* present in report; aligns with course note.

## EPIC J — Code quality & hygiene (rubric: Code Quality 10 pts)

* **J1: Style & docs**

  * *Tasks:* `black`/`ruff` formatting; module-level docstrings; type hints; small functions; no notebook-only logic (reusable funcs live in `src/`).
  * *AC:* `ruff` passes; README has “How to run” and “Reproduce results” sections.
* **J2: Lightweight logs**

  * *Tasks:* simple console logging with timestamps; `utils.py` logger helper.
  * *AC:* training prints epochs, val RMSE, early stopping reasons.

---

# 3) Notebook purposes (clear scope)

* **00\_eda\_preprocessing.ipynb**
  Full audit; decisions; write cleaned data & scalers to `data/interim/`/`processed/`. All plots captioned with “so what?”.
* **10\_model\_lstm\_baseline.ipynb**
  First working LSTM; early stopping; logs to `experiments.csv`.
* **11\_model\_gru\_baseline.ipynb**
  GRU comparison; note parameter efficiency and results.
* **12\_model\_lstm\_stacked.ipynb**
  Depth/units/dropout ablations; discuss vanishing gradient mitigation (clipnorm).
* **19\_submit\_inference.ipynb**
  Loads **best** config weights + scalers; produces Kaggle CSV; runs submission QA checklist.

> If needed, you can add **13\_bidirectional\_lstm.ipynb** (ok since the window is past-only; no leakage beyond the window) for another comparison.

---

# 4) Coding practices you’ll follow

* **Modularization:** all heavy lifting (prep, windowing, model builders, train loop, metrics) lives in `src/` and is imported into notebooks.
* **Config-driven runs:** a `config.py` (or JSON/YAML per run) defines lookback, batch size, units, layers, dropout, learning rate, optimizer, scaler type, target transform, and CV split dates.
* **Reproducibility:** global seeds; fixed CV splits; pinned deps.
* **Leakage guards:** shifts and windowing exclude future; scalers fit on train fold only; validation never touches fitting steps.
* **Mitigations:** gradient clipping; ReduceLROnPlateau; early stopping; avoid too-long lookbacks that cause vanishing gradients.

---

# 5) Experiment table schema (meets rubric: ≥15 rows)

Columns (in `experiments/experiments.csv`):

```
run_id,date,model,lookback,features,layers,units,dropout,recurrent_dropout,
optimizer,lr,batch,epochs,clipnorm,scaler,target_transform,cv_rmse_mean,cv_rmse_std,notes
```

You’ll paste this table (rendered) into REPORT.md and discuss trends.

---

# 6) Kaggle submission checklist

* [ ] Load **exact** `test.csv` and `sample_submission.csv` ID order.
* [ ] Predict recursively 1-step ahead across horizon (no peeking).
* [ ] Invert transforms: `expm1` then unscale; clip to `>= 0`.
* [ ] Shape: 13,148 rows; columns exactly `row ID, pm2.5`.
* [ ] Spot-check first/last 5 rows match expected timestamps.
* [ ] Save to `outputs/submissions/submission_YYYYMMDD_HHMM.csv`.

---

# 7) Report outline (matches rubric sections)

1. **Introduction** — problem, why PM2.5, sequence approach (justify RNN/LSTM).
2. **Data Exploration & Preprocessing** — key findings + why each step helps forecasting (with plots + captions).
3. **Model Design & Architecture** — chosen architecture diagram + hyperparams + rationale.
4. **Experiment Table** — ≥15 runs with commentary on trends.
5. **Results & Discussion** — define **RMSE** with formula; compare models; show prediction vs actual; discuss over/underfitting; mention vanishing/exploding gradients and mitigations used.
6. **Conclusion & Next steps** — what worked, what didn’t, concrete improvements.
7. **Citations (IEEE)** — primary papers & docs.
8. **GitHub link** — reproducibility instructions.

---

# 8) Quick starter backlog (copy into GitHub Issues)

**\[EPIC] A — Scaffold & Env**

* [ ] A1 Repo structure & .gitignore
* [ ] A2 Env file & seed utility

**\[EPIC] B — EDA & Preprocessing**

* [ ] B1 Data audit & visuals (saved)
* [ ] B2 Preprocessing rationale written
* [ ] B3 Time features & scaler fit (train-only)

**\[EPIC] C — Windowing & Splits**

* [ ] C1 Leak-safe windowing util
* [ ] C2 Rolling/blocked CV splits

**\[EPIC] D — Baselines (sanity)**

* [ ] D1 Naïve & seasonal-naïve RMSE

**\[EPIC] E — Models**

* [ ] E1 LSTM baseline notebook
* [ ] E2 GRU baseline notebook
* [ ] E3 Stacked LSTM notebook

**\[EPIC] F — Experiments**

* [ ] F1 Define experiment grid (≥15)
* [ ] F2 Log to experiments.csv

**\[EPIC] G — Evaluation**

* [ ] G1 Metrics + plots + discussion
* [ ] G2 Select best model (justify)

**\[EPIC] H — Submission**

* [ ] H1 Inference & CSV export
* [ ] H2 Submission QA checklist

**\[EPIC] I — Report & Citations**

* [ ] I1 REPORT.md draft to final
* [ ] I2 IEEE references + originality note

**\[EPIC] J — Code Quality**

* [ ] J1 Formatting, docstrings, types
* [ ] J2 README “How to run + reproduce”

---

If you like, I can turn this into a ready-to-commit README.md + REPORT.md skeleton and an issue list you can paste into GitHub Projects.
