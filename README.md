# Ryan PINN failure-and-repair benchmark

This repository implements the approved The Bu1LD / FinanceMeta Scientific ML trial. Its frozen hypothesis is that a fixed random Fourier feature (RFF) input map can reduce the spectral bias of a vanilla tanh PINN on a manufactured Helmholtz problem. **n=6 is primary. n=12 is the preregistered stress/failure condition regardless of outcome.** This repository contains implementation and validation only; the 12 protected runs have not been executed here.

## Frozen experiment

On `[-1,1]²`, solve `Δu + 20²u = [20² − 2(nπ)²] sin(nπx)sin(nπy)` with zero Dirichlet boundary. Per seed `{0,1,2}`, one committed 2,048-point scrambled Sobol realization is reused across both methods and both n values. Boundary points are 128 per side, 512 unique total. The normalized PDE residual and boundary MSE have equal weight. Vanilla is `2→64→64→64→64→1` tanh; RFF uses committed, nontrainable `B∈R^(32×2)` from independent seed 2026 and `64→55→55→55→55→1` tanh. Both use full-batch float32 Adam, 8,000 updates, and a predeclared cosine LR from `1e-3` at update 1 to `1e-4` at update 8,000. Final weights are those after update 8,000.

No resampling, adaptive weighting, curriculum, early stopping, best checkpoint, scientific retry, or post-hoc retuning is permitted. The held-out `257×257` grid is accessed for each n only after its six training runs terminate. Failures remain visible; infrastructure interruptions use explicit attempt records and recovery.

## Review and reproduce

1. Inspect [the compliance map](docs/PROTOCOL_COMPLIANCE.md), [Colab steps](docs/COLAB_EXECUTION.md), [experiment log](EXPERIMENT_LOG.md), and [citations](CITATIONS.md).
2. From the repository root run `python -m pip install -r requirements.txt`, `python scripts/validate_protocol.py`, `python -m pytest -q`, and optionally `python scripts/run_smoke.py`. Smoke output is explicitly non-evidence and outside `results/raw/`.
3. Push a reviewed clean commit and freeze its exact SHA. Use an ordinary Colab GPU runtime and follow [docs/COLAB_EXECUTION.md](docs/COLAB_EXECUTION.md). The single official command is `python scripts/run_frozen.py --config configs/frozen.yaml`.

For one model per cell, open [colab/run_protected.ipynb](colab/run_protected.ipynb) and run its sequential stage cells. The corrected launcher defaults to the reviewed experiment SHA `0677b40869fabf50035075e240aea3b98a0c7e05`, preserves verified pretraining setup failures in an external archive, and refuses revision changes for actual or ambiguous training evidence. A newer launcher still checks out that same frozen experiment revision. The same runner exposes `--action train`, `evaluate`, `finalize`, and `status`. Training prints update counts, loss, elapsed time, and ETA every 100 updates. Every model retains its frozen budget; held-out evaluation still requires six terminal training runs for its condition. Completed and scientifically failed slots are skipped rather than retrained.

If existing results are recorded at original SHA `3258b1d47baa734fea102d7f8ea38fdd626d8174`, use [the original experiment recovery notebook](colab/resume_original.ipynb). It retains that revision and adds an external training-log monitor to its original full pipeline command. Confirm the previous process has stopped before enabling explicit infrastructure recovery.

`scripts/prepare_frozen_assets.py` documents one-time asset creation but refuses to replace the committed files. The official runner loads and verifies their SHA-256 hashes; it never regenerates them. `configs/frozen.yaml` and `scripts/validate_protocol.py` define and enforce the approved values. The runner requires a clean Git revision and CUDA, and captures SHA, config/asset hashes, software versions, and actual GPU model.

## Evidence layout

- `results/raw/n06/{vanilla,rff}/seed_{0,1,2}/attempt_000/`: resolved config, provenance/status, training log, final weights, then held-out predictions and metrics for successful runs. The same pattern applies to `n12`.
- `results/processed/`: per-seed results, paired effects, summary, compute record, and complete attempt manifest.
- `figures/`: training curves, per-seed exact/predicted/error fields, and comparison charts.
- `report/methods_results.md`: generated methods/results note, with `PENDING PROTECTED RUN` if run evidence is absent.

The comparison is descriptive for three seeds. An independent hyperparameter sweep was outside the approved protected protocol. GPU arithmetic can differ across hardware and library versions even with fixed seeds and deterministic settings; the exact committed data and provenance make such variation auditable.

`requirements.txt` pins one set for Python below 3.13 and a second set for Python 3.13+ environments. The committed Sobol/RFF arrays, not regeneration under either dependency set, govern protected execution.
