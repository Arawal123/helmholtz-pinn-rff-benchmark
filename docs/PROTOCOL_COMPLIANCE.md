# Approved protocol compliance map

| Requirement | Implementation / evidence |
|---|---|
| Phase separation; no protected construction runs | `scripts/run_frozen.py` is the only protected entrypoint; `scripts/run_smoke.py` uses `configs/smoke.yaml`, CPU, and ignored `smoke_outputs/`. Official `results/raw/` is empty until Colab execution. |
| Source precedence, no post-hoc changes | `configs/frozen.yaml` specifies approved values; `scripts/validate_protocol.py` asserts them exactly before the runner starts. `EXPERIMENT_LOG.md` records chronology and deviations. |
| PDE, domain, k, exact solution, forcing | `src/pde.py`; checks in `tests/test_protocol.py`. |
| n=6 primary; n=12 preregistered stress | `configs/frozen.yaml`, fixed order and labels in `scripts/run_frozen.py`, all tables and report. |
| 2,048 interior scrambled Sobol points; seeds 0,1,2; shared realization | `data/frozen/sobol_seed_*.npy`, `data/frozen/generation.json`, hashes in config; `src/data.py` loads seed asset independent of method/n. `run_manifest.csv` records the same seed hash across pair and difficulty. |
| 512 fixed boundary points, 128 per side, no duplicate corners | `src/data.py:boundary_points`. Each oriented side uses half-open equally spaced coordinates, starting at one corner and ending before the next; 512 unique points. |
| 257×257 held-out grid, delayed evaluation, no selection | `src/data.py:heldout_grid`, six-training-run barrier in `scripts/run_frozen.py`, `src/evaluate.py`. Training never imports or calls held-out evaluation. |
| Normalized residual and equal-weight boundary MSE | `src/pde.py`, checked in `tests/test_protocol.py`. |
| Vanilla architecture, tanh, Xavier, zero bias, 12,737 parameters | `src/models.py`, validator and tests. |
| RFF mapping and immutable B, seed 2026, Gaussian σ=3, 12,871 parameters, <2% gap | `artifacts/rff/B_seed_2026.npy` and `.csv`, hash and metadata; `src/models.py` buffer; validator and tests. |
| float32 full-batch Adam, initial/final LR, cosine over exactly 8,000 updates | `src/train.py:learning_rate` and `train_run`, validator and tests. No L-BFGS, adaptive weights, resampling, curriculum, early stopping, or best-checkpoint path. |
| Numerical failure preserved, no scientific retry | `src/train.py` checks loss, gradients, parameters; writes failure step/reason and partial log. `scripts/run_frozen.py` treats `failed` as terminal. |
| Infrastructure interruption distinct and recoverable | `scripts/run_frozen.py` writes `infrastructure_interrupted`, preserves attempt directory, requires explicit `--recover-infrastructure`, and refuses prior SHA/config mismatch. `docs/COLAB_EXECUTION.md` describes persistence and lost-storage limit. |
| Overwrite safety | Exclusive `attempt_NNN` creation; completed/failed slots skipped and never overwritten. |
| Every seed and per-seed raw evidence | Twelve deterministic run roots under `results/raw/`; each attempt has resolved config, `run.json` provenance/status, log, final state if successful, predictions and metrics after gate. |
| Relative L2 primary; L∞, residual RMS, boundary RMSE secondary | `src/evaluate.py`, raw `metrics.json`, `results/processed/per_seed_results.csv`. |
| Paired effects; percentage improvement; means/SD; failures visible | `scripts/aggregate_results.py`, `paired_effects.csv`, `summary.csv`, `per_seed_results.csv`, `run_manifest.csv`. No significance test. |
| Training curves, error fields, seed comparison | `scripts/aggregate_results.py` creates figures only after condition-gated metrics are available. |
| Compute, timing, parameters, hardware | `src/provenance.py`, per-attempt `run.json`, `results/processed/compute_record.csv`. |
| Git SHA, clean status, Python/PyTorch/CUDA/GPU and hashes | `src/provenance.py`, `scripts/run_frozen.py`; protected run refuses dirty/uncommitted source, no SHA, no CUDA, or invalid hashes/config. |
| Methods/results note; pending status; limitations | `scripts/aggregate_results.py` generates `report/methods_results.md`; absent evidence is labelled `PENDING PROTECTED RUN`. Per-seed and effect tables, seed variability, frozen hyperparameters, no independent sweep, no post-hoc tuning, negative results and restrained interpretation are explicit. |
| Experiment log and citations | `EXPERIMENT_LOG.md`, `CITATIONS.md`; no reused code/data. |
| Colab reproducibility and single command | `docs/COLAB_EXECUTION.md`; `python scripts/run_frozen.py --config configs/frozen.yaml`. No notebook scientific logic or machine-specific path. |
| Local required tests | `tests/test_protocol.py`, `scripts/validate_protocol.py`, `scripts/run_smoke.py`; test log/status recorded in final construction response. |

The protected phase is intentionally pending. Generated tables, plots, and report cannot contain official values until the reviewed commit is run on Colab.
