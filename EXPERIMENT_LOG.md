# Experiment log — append-only record

## Frozen hypothesis and approval

RFF input features should mitigate vanilla tanh PINN spectral bias for the primary `n=6` Helmholtz manufactured solution. `n=12` is a preregistered stress/failure check. Ryan's later approval freezes bandwidth, widths, optimizer, points, scaling, seeds, and step budget. Negative or inconclusive outcomes remain evidence.

## Construction decisions (before protected execution)

- Boundary points: 128 half-open, equally spaced samples around each oriented side; each corner occurs once, giving 512 unique points.
- Weight initialization: Xavier uniform; all biases exactly zero. The RFF matrix is a nontrainable model buffer.
- Sobol: PyTorch scrambled 2D Sobol, seed 0/1/2, 2,048 draws, mapped to `(-1,1)²`, saved as float32 `.npy`. RFF: independent CPU `torch.Generator` seed 2026, one float32 normal draw of shape `(32,2)` scaled by 3. Generation metadata and SHA-256 hashes are committed.
- Cosine LR uses `update-1` over `7999`: update 1 is exactly `1e-3`, update 8,000 exactly `1e-4`; no restarts.
- End-of-training weights are retained; there is no checkpoint selection. Evaluation is barred until all six training slots for that n are terminal.
- Infrastructure recovery starts a new attempt with the identical frozen specification and preserves prior attempts; scientific failures remain terminal.

## Pre-run verification

Construction verification, 2026-10-01 (local Windows CPU): Python 3.14.3, PyTorch 2.12.0+cpu. `python scripts/validate_protocol.py` passed frozen values and hashes; `python -m pytest -q` passed 9 tests; `python scripts/run_smoke.py` passed tiny CPU training and evaluation for both methods at non-official `n=2`, 24 interior points, and 3 updates; `python -m compileall -q src scripts tests` passed; `python -m pip install --dry-run -r requirements.txt` resolved the local pinned requirements. Smoke files are ignored under `smoke_outputs/`; `results/raw/` contains no protected attempts. A reviewed commit SHA and Colab GPU verification remain for the protected phase.

Before protected execution, append the reviewed Git SHA, reviewer, validation result, assigned Colab accelerator, and start timestamp here. Local construction does not execute protected runs.

## Protected execution / interruptions / failures / observations / decisions / deviations

Append dated entries with exact Git SHA, GPU, attempt path, event, evidence path, and decision. Preserve chronology. Never rewrite a failure into a successful story or alter frozen settings after seeing held-out metrics.
