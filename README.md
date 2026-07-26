# scoregate

A reference implementation of model governance for credit scoring. It takes the Home
Credit Default Risk data from raw CSVs to two registered, calibrated, fairness-checked
models, and every step is reproducible from one command.

Most public ML repos show training. Almost none show governance: how a model gets
promoted, how a decision gets explained, how fairness gets measured, how the whole run
stays reproducible. That's the gap this fills. It's educational, built on public data,
and it's not financial advice.

## What it demonstrates

- **A gated pipeline, not a notebook.** Data flows through a Pandera contract that
  fails the run on any violation, a seeded split that's persisted and hashed, and WOE/IV
  feature selection with an audited decision record.
- **Champion and challenger.** A logistic scorecard (points, PDO 20) and a LightGBM
  challenger are trained, evaluated on a frozen holdout, and registered in MLflow under
  one model with `@champion` / `@challenger` aliases. The champion is chosen from the
  committed evaluation record, not hardcoded.
- **Calibrated probabilities.** Both models use balanced class weights, which distorts
  raw probabilities, so both are calibrated with isotonic regression. The Brier score
  drops from around 0.20 to around 0.07 while ranking is unchanged.
- **Explanations and fairness.** Adverse-action reason codes (SHAP for the challenger,
  points for the scorecard) and a holdout fairness snapshot across gender and age bands.
- **Reproducibility.** `make train` produces byte-identical `evaluation.json`,
  `calibration.json`, and `fairness.json` on every run. Experiment tracking is optional
  and offline-safe: the pipeline runs start to finish with no W&B account, no key, and no
  network.

## Architecture (Phase 1)

```
Kaggle Home Credit  --download_data.py-->  data/raw
        |
        v
DuckDB  --SQL feature engineering (sql/01..07)-->  features + fairness_metadata
        |  Pandera contract (contracts.py): fails the run on violation
        v
stratified 80/20 split (split.py, seed 42, persisted table + manifest)
        |
        v
WOE/IV binning on train only (binning.py, optbinning), IV selection audited
        |
        +--> logistic scorecard, points scaled (scorecard.py) ---+
        +--> LightGBM challenger, raw features (challenger.py) ---+--> evaluate on the
        |                                                         |    frozen holdout
        v                                                         v    (evaluate.py)
   isotonic calibration (calibration.py)              fairness snapshot (fairness.py)
        |
        v
MLflow registry (registry.py): two versions, model cards, @champion / @challenger
```

Metrics are hand-rolled in `metrics.py` (KS, Gini via the rank identity, PSI for the
Phase 3 drift monitor). W&B run tracking lives behind a thin sink in `tracking.py` that
the pipeline never depends on.

## Results (frozen holdout)

The holdout is touched once, for reporting. Nothing is fit, tuned, selected, or
calibrated on it.

| model | AUC | KS | Gini | overfit gap | Brier before -> after |
|---|---|---|---|---|---|
| scorecard | 0.750 | 0.375 | 0.501 | 0.003 | 0.201 -> 0.068 |
| challenger | 0.766 | 0.404 | 0.531 | 0.099 | 0.185 -> 0.067 |

The challenger wins the holdout and becomes `@champion`. The overfit gap (train Gini
minus holdout Gini) is tiny for the scorecard and larger for the challenger, which is the
honest generalization check and the number a future sweep is watched against.

The fairness snapshot at an 80% approval cutoff shows a small gender approval-rate gap
(around 5 to 6 points) and a large age-band gap (34 to 42 points). CODE_GENDER is never a
model feature; age is, and age-in-scoring is jurisdiction-dependent, so the gap is
reported rather than hidden.

## Quickstart

You need Python 3.11 and Kaggle API credentials (the competition data can't be
redistributed, so you download it yourself).

```
make install                 # venv + editable install with dev tools
make download                # pull the Home Credit data via the Kaggle API
make train                   # the whole pipeline, end to end
```

`make train` runs offline by default. To log runs to W&B, `wandb login` first; to force
it off, `make train` already stays offline without a key, or run the module with
`--no-wandb`.

To tune the challenger:

```
make sweep                   # register the Bayesian sweep, prints the agent command
wandb agent <sweep-id> --count N
```

## Pipeline stages

Each stage is a `make` target and a module, run in order by `make train`:

`download` -> `ingest` -> `features` -> `contract` -> `split` -> `binning` ->
`scorecard` / `challenger` -> `evaluate` -> `calibrate` -> `fairness` -> `register`.

The committed records are the audit trail: `split_manifest.json` (split hashes),
`feature_selection.json` (IV dispositions), `evaluation.json` (the champion decision),
`calibration.json` (before/after reliability), and `fairness.json` (the parity snapshot).

## Limitations

- Educational, on public data. Not financial advice.
- Reject inference: outcomes are only observed for approved applicants, a known
  limitation of any scorecard built on historical decisions.
- CODE_GENDER is excluded as a feature, but proxy effects can remain, which is why
  outcome parity is measured. Age is a feature and age-in-scoring is
  jurisdiction-dependent.

## Roadmap

Phase 1 (this) is data to registered models. Later phases add a FastAPI decision service
with a hash-chained audit trail, a guarded LLM memo layer, and a drift monitor that opens
retraining tickets when PSI breaches threshold.

## License

MIT.
