# Helmholtz PINN: fixed Fourier features and solution recovery

A completed, frozen comparison of vanilla and random-Fourier-feature PINNs on a manufactured Helmholtz problem. The submission contains all twelve models, raw evidence, processed statistics, figures and a methods/results report.

## Results

At primary n=6, RFF reduces relative L2 error by a mean paired 5.04% across three seeds, while increasing held-out PDE residual and maximum solution error. At stress n=12, RFF improves relative L2 in one of three pairs. Very low RFF training loss coexists with poor solution recovery and a large residual generalization gap. The comparison is descriptive, and all seeds and negative outcomes are retained.

## Read the submission

- [Methods and results](report/methods_results.md) and [PDF report](report/methods_results.pdf).
- [Experiment log](EXPERIMENT_LOG.md), including completion facts and the execution-history deviation.
- [Per-seed results](results/processed/per_seed_results.csv), [paired effects](results/processed/paired_effects.csv), [summary](results/processed/summary.csv), [compute](results/processed/compute_record.csv) and [manifest](results/processed/run_manifest.csv).
- [Figures](figures/), [frozen protocol](configs/frozen.yaml), [requirements coverage](docs/PROTOCOL_COMPLIANCE.md) and [attribution](CITATIONS.md).
- [Submission integrity](submission/INTEGRITY.md) and [original-file mapping](submission/provenance/original_file_map.json).

## Scientific execution

The result-bearing scientific Git SHA is `0677b40869fabf50035075e240aea3b98a0c7e05`. The executed launcher SHA is `de869e2d240439cbe24a16d96478e7e824322625`. This submission revision adds documentation and evidence; it does not change the scientific settings or represent additional training.

The frozen problem is Δu+20²u=f on [-1,1]², with zero boundary values and u*=sin(nπx)sin(nπy). Primary n=6 and stress n=12 each use seeds 0,1,2, 2,048 fixed interior points, 512 fixed boundary points, float32 full-batch Adam for 8,000 updates and a cosine learning rate from 10^-3 to 10^-4. The committed point arrays and Fourier matrix are the source of truth.

## Verify existing evidence

```bash
python scripts/verify_submission.py
```

This verifies file checksums, original-file preservation, all twelve completed/evaluated slots, asset provenance and processed numerical consistency without training. The `--cpu-model-check` option additionally re-evaluates saved models on CPU after installing requirements.

## Reproduction

From a separate clean CUDA checkout of the scientific SHA:

```bash
git clone https://github.com/Arawal123/helmholtz-pinn-rff-benchmark.git helmholtz-reproduction
cd helmholtz-reproduction
git checkout --detach 0677b40869fabf50035075e240aea3b98a0c7e05
python -m pip install -r requirements.txt
python scripts/validate_protocol.py
python scripts/run_frozen.py --config configs/frozen.yaml
```

[Colab execution](docs/COLAB_EXECUTION.md) describes GPU setup and persistence. The submission already contains completed results; reproduction is not required to inspect them.

## Execution-history disclosure

The submitted cohort followed an earlier protected execution. The experimenter confirmed that the additional execution had no separate approval from Ryan. The history, available earlier status record and absence of the earlier raw archive from this package are disclosed in the experiment log. Acceptance of that workload deviation remains for reviewer disposition; no full-compliance claim is made.
