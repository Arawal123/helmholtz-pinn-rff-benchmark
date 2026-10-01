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
