# Protected Google Colab execution

The approved protected phase is performed manually after review and GitHub push. Use a GPU runtime (`Runtime → Change runtime type → GPU`); ordinary Colab GPU access is sufficient when a compatible accelerator is assigned. The runner detects and logs the actual model. Keep the tab/runtime available for the full workload; 12 second-derivative PINN fits can take substantial time.

Replace `<COMMIT_SHA>` with the exact reviewed commit SHA. Do not edit source/config/assets after checkout.

```bash
git clone https://github.com/Arawal123/helmholtz-pinn-rff-benchmark.git ryan-pinn-trial
cd ryan-pinn-trial
git checkout <COMMIT_SHA>
git rev-parse HEAD
python -m pip install -r requirements.txt
python -c "import torch,platform; print(platform.python_version(), torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python scripts/validate_protocol.py
python -m pytest -q
python scripts/run_frozen.py --config configs/frozen.yaml
```

The runner requires a clean committed Git SHA, verified asset hashes, exact frozen configuration, and CUDA. Its CLI cannot override scientific settings. It creates each run attempt exclusively and never overwrites completed or scientifically failed attempts. For each n, all three vanilla and all three RFF training runs must finish or fail scientifically before any held-out grid evaluation for that n. It then writes processed tables, figures, and the note.

## One model per notebook cell

Open the latest `colab/run_protected.ipynb` in Colab and enable GPU. Its default `COMMIT_SHA` is already set to the reviewed staged experiment revision `0677b40869fabf50035075e240aea3b98a0c7e05`; keep that value for this experiment. The launcher itself may be newer: it checks out the original experiment revision, so notebook fixes do not change the scientific code SHA. The notebook clones into mounted Drive, checks that CUDA can execute a tiny unrelated second derivative, validates the protocol/tests, and logs each repository CLI invocation outside the source checkout. It contains 12 separate training cells, two condition evaluation cells, and a final deliverable-generation cell. It implements no scientific logic. Run cells sequentially in one runtime; pause between completed models as desired.

When switching from either known historical experiment revision, setup distinguishes empty folders or infrastructure errors created before training status from actual or ambiguous training records. Only verified pretraining attempts are moved, byte-preserved, to `BASE/setup_attempt_archive/<timestamp>/` with an original-SHA manifest and external journal entry. The submission ZIP includes this archive. A training log (even header-only), resolved configuration, running/failed record, weights, unknown files, malformed record, or provenance prevents that migration. Actual training attempts must retain their original SHA; an older revision without staged actions requires its original runner command. Setup prints the existing attempt statuses and recorded SHAs when it refuses migration.

Following a documented interruption, stop the old process and set `RECOVER_INFRASTRUCTURE = True` in the first cell, repeat setup at the same experiment SHA, then rerun the affected cell. The helper adds the explicit recovery flag. Set the value back to False after recovery. Validation saves a timestamped environment record each time; training invocations verify the SHA and clean checkout before starting.

Each training invocation prints the update count, percentage, training loss, LR, elapsed time, and estimated remaining time at update 1, every 100 updates, and update 8,000. ETA is an elapsed-time estimate, not a stopping rule. The training CSV and final weights follow the same frozen protocol.

The equivalent repository commands, each run in its own cell, are:

```bash
python scripts/run_frozen.py --action train --n 6 --method vanilla --seed 0
python scripts/run_frozen.py --action train --n 6 --method vanilla --seed 1
python scripts/run_frozen.py --action train --n 6 --method vanilla --seed 2
python scripts/run_frozen.py --action train --n 6 --method rff --seed 0
python scripts/run_frozen.py --action train --n 6 --method rff --seed 1
python scripts/run_frozen.py --action train --n 6 --method rff --seed 2
python scripts/run_frozen.py --action evaluate --n 6
python scripts/run_frozen.py --action train --n 12 --method vanilla --seed 0
python scripts/run_frozen.py --action train --n 12 --method vanilla --seed 1
python scripts/run_frozen.py --action train --n 12 --method vanilla --seed 2
python scripts/run_frozen.py --action train --n 12 --method rff --seed 0
python scripts/run_frozen.py --action train --n 12 --method rff --seed 1
python scripts/run_frozen.py --action train --n 12 --method rff --seed 2
python scripts/run_frozen.py --action evaluate --n 12
python scripts/run_frozen.py --action finalize
```

The selectors choose among the approved 12 slots; they cannot change scientific settings. Evaluation refuses an incomplete six-run condition. Staged n=12 execution requires completed n=6 evaluation. Finalization refuses incomplete training/evaluation and generates all five tables, figures, and the report. Repeating a terminal training cell safely skips it. The original single command still orchestrates all stages and now also displays live progress.

Between cells, `python scripts/run_frozen.py --action status` shows all 12 training statuses and their last logged update. It works without GPU and does not load or display held-out metrics. Use one training/evaluation process at a time. Infrastructure recovery uses the same selected action plus `--recover-infrastructure`; it creates a new attempt only for an infrastructure-interrupted training slot.

Freeze one SHA for the entire experiment. Existing protected attempts at another SHA cannot be combined with this revision. If training has already begun, continue at its original revision; preserve every attempt.

Check `results/processed/run_manifest.csv` for the 12 selected run slots and any older infrastructure attempts, and `results/processed/per_seed_results.csv` for every seed. Scientific failures are terminal and must not be rerun. If the runtime disconnects or crashes, retain the existing output directory. Restart from the **same clean commit**, restore the complete `results/raw/` tree (including interrupted attempts), validate, then explicitly invoke:

```bash
python scripts/run_frozen.py --config configs/frozen.yaml --recover-infrastructure
```

This creates a new attempt only for infrastructure-interrupted slots under the same scientific specification. It never re-executes a completed or scientifically failed slot. A bare `running` record left by a hard crash is marked infrastructure-interrupted. Do not silently delete an attempt. If Colab ephemeral storage is lost before it can be persisted, document the loss in `EXPERIMENT_LOG.md`; no raw result can be reconstructed from a missing artifact.

Persist evidence without changing code/config. For example, in Colab run:

```bash
zip -r ryan_pinn_protected_results.zip results/raw results/processed figures report/methods_results.md
```

Then use Colab's Files panel to download the ZIP or copy the four output paths to mounted Drive. Store the exact Git SHA with it. Do not commit generated outputs into the clean source checkout mid-run; their paths are ignored to keep the runner's clean-Git gate meaningful. There are no local-machine paths, secrets, notebooks, or paid services in the scientific pipeline.
