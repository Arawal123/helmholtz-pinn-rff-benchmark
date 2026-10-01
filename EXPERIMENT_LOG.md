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

## Execution interface update, 2026-10-01 (before protected execution)

The user confirmed that no protected model had started and requested visible progress and separate runs. Added training-only console progress and runner actions for one approved slot per invocation, condition evaluation, finalization, and metadata-only status. Added a thin Colab launcher with one training cell per model. Frozen scientific configuration, assets, architectures, objective, optimizer, schedule, and step budget are unchanged. New attempts must all use the same reviewed updated Git SHA; existing attempts from another revision remain protected from reuse or overwrite. Local verification uses synthetic orchestration fixtures and tiny non-protected CPU smoke runs only.

Verification of this update: 16 unit tests passed, frozen protocol/asset validation passed, Python static compilation passed, and the tiny CPU smoke pipeline passed with visible progress. Tests cover all 12 staged slots, evaluation/finalization barriers, failed/completed skip behavior, revision mismatch, explicit infrastructure recovery, metadata-only status, and unexecuted notebook cell syntax/order. No protected attempts exist in the local raw output tree.

## Colab launcher repair, 2026-10-01

The user reported a placeholder-SHA assertion and a revision guard triggered by existing attempt folders. The corrected launcher defaults to experiment SHA `0677b40869fabf50035075e240aea3b98a0c7e05`; this launcher revision does not change the experiment checkout or scientific code. For the two known historical revisions, empty folders or narrowly verified infrastructure records written before training status may be archived outside the checkout, with all bytes retained, original SHA, and a journal entry. Any training/ambiguous evidence blocks migration. Added an unrelated CUDA second-derivative preflight, explicit notebook recovery flag, per-invocation SHA/clean-checkout guards, and timestamped environment records. No protected models were run locally. Local synthetic tests cover migration without data loss and refusal to move actual training records; 28 tests passed. Fresh Colab GPU execution remains unverified locally.

## Original experiment continuation, 2026-10-01

The user supplied Colab status records showing six primary models completed at 8,000 updates and one stress model recorded as running at original SHA `3258b1d47baa734fea102d7f8ea38fdd626d8174`. These are user-reported remote records; no local protected run or scientific result inspection occurred. Added a separate original-revision recovery notebook using only that original runner command and a read-only training CSV monitor. The monitor resolves stale `updates_completed=0` records without reading held-out metrics. It guards against visible concurrent runner processes and defaults explicit recovery to disabled. User must verify that other runtimes have stopped before recovery. The original revision and its evidence remain intact; the newer staged launcher cannot be used on these attempts. Synthetic tests verify stale status handling, monitoring the newest attempt while preserving old bytes, and the exact original entrypoint invocation.

## User-requested fresh execution, 2026-10-01

The user explicitly requested starting from the beginning. Added `colab/start_from_beginning.ipynb`, pinned to the reviewed staged experiment SHA `0677b40869fabf50035075e240aea3b98a0c7e05`, using a separate persistent Drive folder `ryan_pinn_trial_from_start_0677b40`. It preserves the earlier experiment folder, logs the restart decision in its external journal, and runs all 12 fixed slots with condition gates and visible progress. No earlier results are selected or combined, and no scientific parameters were changed. Tests check the complete stage sequence, fixed revision, separate folder, empty notebook outputs, syntax, and absence of moves/deletions in fresh setup. No protected training was executed locally.
