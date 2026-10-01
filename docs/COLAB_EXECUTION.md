# Colab reproduction

The completed submission can be inspected and verified without GPU training. Reproduction uses scientific SHA `0677b40869fabf50035075e240aea3b98a0c7e05`; the full CLI is `python scripts/run_frozen.py --config configs/frozen.yaml`.

1. Enable a Colab GPU runtime and clone a separate checkout under `/content`.
2. Check out the exact scientific SHA, verify clean Git status and install `requirements.txt`.
3. Run `scripts/validate_protocol.py` and the scientific unit tests.
4. Launch the full pipeline or the fixed stage commands in `colab/run_protected.ipynb`. All six primary models terminate before primary evaluation; stress execution follows primary evaluation.
5. Back up completed evidence to Drive and retain the source SHA, software/hardware metadata and all attempt directories.

The final scientific source contains an older launcher; the actual executed launcher is archived at `submission/execution/executed_launcher.ipynb`. The active notebook in this submission is a concise reproduction interface pinned to the same scientific SHA. Neither launcher contains scientific model logic.

The protected runner requires CUDA, fixed asset hashes and a clean committed scientific source. Completed or scientifically failed slots are terminal. Infrastructure recovery requires the explicit `--recover-infrastructure` flag at the original SHA and preserves the prior attempt. Final weights are those at update 8,000. No scientific retry, checkpoint selection or post-hoc tuning is part of reproduction.

Actual submitted execution used a Drive-mounted checkout for persistence. That differs from the recommended local `/content` workflow but did not change the scientific settings. Its exact launcher and original console logs are retained.
