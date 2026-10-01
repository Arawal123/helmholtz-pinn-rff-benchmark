# Figures

The 16 original figures are preserved unchanged:

- `n06_training_curves.png` and `n12_training_curves.png`: all-seed loss convergence.
- `n06_comparison.png` and `n12_comparison.png`: original method/seed comparisons.
- `n{06,12}_{vanilla,rff}_seed_{0,1,2}_error.png`: original exact, prediction and absolute-error panels for every model.

Additional figures summarize the same evidence:

- `submission_summary.png`: paired relative L2 and final-model training-point versus held-out residual. The residual comparison is a post-run CPU diagnostic, not an additional experiment.
- `solution_fields_shared_scale.png`: fixed seed 0, with common solution scales within each condition row and no clipped values.
- `n{06,12}_{vanilla,rff}_seed_{0,1,2}_shared_scale.png`: all twelve models with solution and absolute-error scales shared across all six models in the corresponding condition. Larger negative outcomes remain visible.

The original numerical tables remain the authoritative result values. Supplementary visualizations do not replace or modify them.
