# Near-zero collocation loss does not imply Helmholtz solution recovery

A frozen comparison of vanilla and random-Fourier-feature PINNs | Ryan PINN trial | 1 October 2026

## Abstract

We compare a vanilla tanh physics-informed neural network (PINN) with a fixed random Fourier feature (RFF) input map on a manufactured two-dimensional Helmholtz problem. A frozen, approximately parameter-matched protocol uses three paired seeds, 2,048 shared interior points, 512 boundary points and 8,000 full-batch Adam updates. The primary condition is n=6; n=12 is a preregistered stress case. RFF reduces primary relative L2 error by a mean paired 5.04%, from 1.00034 to 0.94988, and reduces boundary error. It increases primary held-out normalized PDE residual RMS from 0.49801 to 0.78864 and maximum absolute error from 1.01612 to 1.22419. At n=12, RFF improves relative L2 in one of three pairs; all six predictions have relative L2 above the zero-prediction reference of one. RFF training losses near 10^-6 coexist with substantial held-out error and a training-point versus held-out residual gap of three to four orders of magnitude. The evidence supports a limited primary repair with pronounced failure to generalize PDE constraints. All results, including repair-losing outcomes, are retained. A separate earlier execution and the unapproved additional workload are disclosed in the experiment log.

## 1. Scientific question and mechanism

The question is whether frequency-aware coordinate features improve solution recovery under an equal training budget when a vanilla PINN struggles with an oscillatory target. The vanilla model represents the solution directly from coordinates. RFF exposes sinusoidal coordinate components before the same class of tanh MLP, motivated by spectral-bias analyses of Fourier-feature networks [2]. The treatment is not guaranteed to win: the n=12 condition tests deterioration, loss of advantage and failure. Primary/stress designations were fixed before this submitted execution. The trial measures the effect of this fixed representation treatment; it does not directly measure a spectral-bias mechanism or establish general superiority.

## 2. Problem and frozen protocol

Ω = [-1, 1]²;  ∇²u + 20²u = fₙ;  u = 0 on ∂Ω

u*ₙ(x, y) = sin(nπx) sin(nπy);  fₙ = [20² - 2(nπ)²]u*ₙ

rθ = (∇²uθ + 20²uθ - fₙ) / Sₙ;  Sₙ = max(1, |20² - 2(nπ)²|)

L = mean(rθ²) + mean(uθ² on the boundary)

For each network/Sobol seed 0, 1 and 2, one committed scrambled Sobol realization supplies exactly 2,048 points strictly inside (-1,1)². The same realization is reused across methods and conditions. A deterministic set of 512 boundary points uses 128 uniformly spaced points per oriented side with no duplicated corners. The held-out grid is a deterministic 257×257 tensor grid over the closed domain. Evaluation occurs only after all six training models for the corresponding condition are terminal. Held-out metrics are not used for training, tuning, checkpoint selection or changes to the approved protocol.

| Setting | Vanilla | RFF |
| --- | --- | --- |
| Architecture | 2→64→64→64→64→1 | 64→55→55→55→55→1 |
| Input representation | Coordinates (x,y) | [sin(2πBx), cos(2πBx)] |
| Trainable parameters | 12,737 | 12,871 (+1.05%) |
| Activation / initialization | tanh; Xavier uniform; zero biases | Same |
| Arithmetic / optimizer | float32; full-batch Adam | Same |
| Learning-rate schedule | Cosine, 10^-3 to 10^-4 | Same |
| Training budget | 8,000 updates | Same |
| Interior / boundary | 2,048 / 512 fixed points | Same points per paired seed |
| Network/Sobol seeds | 0, 1, 2 | 0, 1, 2 |
| RFF matrix | Not applicable | B: 32×2; Gaussian σ=3; seed 2026 |

B is sampled once on CPU, committed, hashed and loaded as a non-trainable buffer in every RFF run. Final model weights are those after update 8,000. There is no L-BFGS stage, adaptive weighting, resampling, curriculum, early stopping, best-checkpoint selection or post-hoc tuning. Nonfinite loss, gradients or parameters constitute terminal numerical failure; infrastructure interruption is recorded separately and cannot silently replace a scientific failure. No such failure or recovery attempt occurs within the submitted 12-model cohort.

## 3. Metrics and descriptive analysis

The primary metric is held-out relative L2 error, ||uθ-u*||₂/||u*||₂. Secondary metrics are maximum absolute solution error, held-out normalized PDE residual RMS and fixed-boundary RMSE. Training curves, parameter counts and wall times describe optimization and cost. Paired differences are RFF minus vanilla, so negative differences favor RFF for these lower-is-better metrics. Percentage improvement is 100×(vanilla-RFF)/vanilla. Means and sample standard deviations supplement all per-seed values. With three seeds, comparisons are descriptive; no statistical significance or population-level superiority is claimed. A prediction of zero everywhere has relative L2 equal to one, boundary RMSE zero and normalized residual RMS approximately 0.49805 on this grid.

## 4. Per-seed results

### n=6: primary condition

| Method | Seed | Rel. L2 | Max error | PDE RMS | BC RMSE | Train s |
| --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 0 | 1.000349 | 1.018269 | 0.498030 | 0.007633 | 148.05 |
| Vanilla | 1 | 1.000278 | 1.010777 | 0.497948 | 0.007847 | 146.58 |
| Vanilla | 2 | 1.000395 | 1.019319 | 0.498049 | 0.007569 | 145.95 |
| RFF | 0 | 0.954105 | 1.194229 | 0.798524 | 0.001329 | 184.28 |
| RFF | 1 | 0.947476 | 1.305380 | 0.791686 | 0.001209 | 184.22 |
| RFF | 2 | 0.948059 | 1.172950 | 0.775706 | 0.001151 | 181.63 |

### n=12: preregistered stress/failure condition

| Method | Seed | Rel. L2 | Max error | PDE RMS | BC RMSE | Train s |
| --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 0 | 1.001258 | 1.054199 | 0.498070 | 0.002468 | 147.52 |
| Vanilla | 1 | 1.947995 | 4.749222 | 0.519844 | 0.038291 | 144.94 |
| Vanilla | 2 | 1.002112 | 1.075556 | 0.498080 | 0.003552 | 145.91 |
| RFF | 0 | 1.031753 | 1.948181 | 0.607655 | 0.000787 | 182.80 |
| RFF | 1 | 1.038761 | 2.056277 | 0.593025 | 0.000437 | 177.68 |
| RFF | 2 | 1.062093 | 1.938749 | 0.571373 | 0.001406 | 180.29 |

All twelve models completed training and evaluation, with one attempt per slot and finite metrics. Completion describes numerical execution; the large solution errors indicate failure to obtain accurate solutions under the fixed budget.

## 5. Primary result: a small global-error gain with adverse residual generalization

| Metric | Vanilla mean ± SD | RFF mean ± SD | Paired improvement |
| --- | --- | --- | --- |
| Relative L2 | 1.00034 ± 5.86e-05 | 0.94988 ± 0.00367 | +5.04% |
| Maximum error | 1.01612 ± 0.00466 | 1.22419 ± 0.0711 | -20.50% |
| PDE residual RMS | 0.498009 ± 5.35e-05 | 0.788638 ± 0.0117 | -58.36% |
| Boundary RMSE | 0.00768292 ± 0.000145 | 0.00122939 ± 9.09e-05 | +84.00% |

RFF lowers relative L2 in all three primary pairs, with improvements of 4.62%, 5.28% and 5.23%. The mean gain is 5.04%, with sample SD 0.37 percentage points. Relative L2 nevertheless remains approximately 0.95, close to the zero-prediction reference of one. RFF reduces boundary error by a mean paired 84.00%, while maximum solution error worsens by 20.50% and held-out PDE residual RMS worsens by 58.36%. The fixed treatment therefore produces a modest global solution-error gain while degrading two secondary measures of solution quality.

Vanilla primary predictions have RMS 0.005–0.010 compared with exact-field RMS about 0.498, consistent with a near-zero field. RFF introduces some target-aligned structure, but low amplitude and spurious spatial detail remain. These outcomes support a limited repair; they do not establish successful recovery of the oscillatory solution.

## 6. Stress result: seed-dependent benefit and an outlier-driven average

| Seed | Vanilla rel. L2 | RFF rel. L2 | RFF − vanilla | Improvement |
| --- | --- | --- | --- | --- |
| 0 | 1.001258 | 1.031753 | +0.030495 | -3.05% |
| 1 | 1.947995 | 1.038761 | -0.909234 | +46.68% |
| 2 | 1.002112 | 1.062093 | +0.059981 | -5.99% |

At n=12, vanilla mean relative L2 is 1.31712 ± 0.54635 and RFF is 1.04420 ± 0.01589. RFF improves one pair and loses two. The mean paired improvement of +12.55% is driven by vanilla seed 1, whose relative L2 is 1.94799; the median paired improvement is -3.05%. All stress predictions exceed the zero-prediction L2 reference of one. RFF lowers boundary error in every stress pair but increases held-out PDE residual in every pair. Its favorable average maximum-error comparison is likewise driven by the poor vanilla seed 1 outcome. The stress evidence shows failure of a consistent repair advantage.

## 7. Collocation convergence and out-of-sample residual

The training curves show vanilla primary loss remaining near 0.25 while RFF decreases to approximately 1.4×10^-6–1.8×10^-6. In the stress condition, RFF final logged total loss is approximately 1.8×10^-7–1.1×10^-6. Low training loss alone is insufficient evidence of solution recovery. Post-run CPU evaluation of the saved final models separately measured residual RMS at the frozen training points and on the held-out grid. This analysis does not add training, select checkpoints or modify official metrics.

| n | Seed | Training RMS | Held-out RMS | Ratio | Interior RMS |
| --- | --- | --- | --- | --- | --- |
| 6 | 0 | 0.000121 | 0.7985 | 6586× | 0.8033 |
| 6 | 1 | 9.44e-05 | 0.7917 | 8383× | 0.7966 |
| 6 | 2 | 0.000237 | 0.7757 | 3273× | 0.7804 |
| 12 | 0 | 0.000291 | 0.6077 | 2090× | 0.6113 |
| 12 | 1 | 5.79e-05 | 0.5930 | 10242× | 0.5965 |
| 12 | 2 | 0.000406 | 0.5714 | 1407× | 0.5749 |

The RFF held-out/training residual ratios span approximately 1,400–10,200. Excluding boundary locations from the held-out grid leaves a similarly large interior residual. This supports substantial overfitting of the PDE constraints at sampled locations. It does not isolate whether frequency choice, finite collocation density, model capacity or optimizer dynamics is the principal cause. The observed near-interpolation of training constraints and poor solution recovery are retained as the central failure mode of this frozen treatment.

![Figure 1. Paired held-out solution errors and final-model PDE residual at training points versus the held-out grid. Hollow circles denote training points; filled squares denote the grid. Connecting lines compare the same seed. Residuals in the lower panels are post-run CPU diagnostics.](../figures/submission_summary.png)
Figure 1. Paired held-out solution errors and final-model PDE residual at training points versus the held-out grid. Hollow circles denote training points; filled squares denote the grid. Connecting lines compare the same seed. Residuals in the lower panels are post-run CPU diagnostics.

![Figure 2. Exact and predicted fields for fixed seed 0. Each row shares a symmetric color limit covering the exact field and both displayed predictions; values are not clipped. Original all-seed error maps and additional common-scale maps are retained in figures/.](../figures/solution_fields_shared_scale.png)
Figure 2. Exact and predicted fields for fixed seed 0. Each row shares a symmetric color limit covering the exact field and both displayed predictions; values are not clipped. Original all-seed error maps and additional common-scale maps are retained in figures/.

## 8. Compute, fairness and sensitivity

Execution used one Tesla T4 (15,637,086,208 reported bytes), Python 3.13.15, PyTorch 2.12.0+cu130 and CUDA 13.0. Measured training time totaled 32.83 minutes; evaluation totaled 11.59 seconds. The first training start to final evaluation spanned 34.79 minutes, including between-cell gaps. Mean model training time was 146.49 seconds for vanilla and 181.82 seconds for RFF, approximately 24.1% higher despite the 1.05% parameter-count difference. Seeds, data, loss and update budgets are paired and fixed.

Seed sensitivity is reported through every per-seed value, paired effect and sample SD. The network and Sobol realizations are jointly indexed by seed, so observed variability combines initialization and point-realization effects. The Fourier matrix has one fixed realization and is not part of this seed variation. All hyperparameters were fixed by the approved protocol; an independent bandwidth, width or optimizer sweep was outside scope and was not performed. These limits preclude broad claims of hyperparameter robustness.

## 9. Reproducibility and execution history

The scientific execution SHA is 0677b40869fabf50035075e240aea3b98a0c7e05. The executed notebook came from launcher revision de869e2d240439cbe24a16d96478e7e824322625. Scientific source files, asset hashes, per-run resolved configs, clean-Git status, software versions, timing and all weights/predictions are supplied. The complete reproduction command is python scripts/run_frozen.py --config configs/frozen.yaml, run from a clean checkout of the scientific SHA after installing requirements and enabling CUDA. The current submission revision contains editorial additions and static evidence; it is not a new training revision.

An audit checked the original source against its Git blobs, recomputed all table statistics, verified evaluation chronology and re-evaluated saved models on CPU without optimization. All 199 checks passed. Maximum CPU/CUDA prediction difference was approximately 3.1×10^-6; the largest metric difference was approximately 6.1×10^-7, consistent with float32 backend differences. This establishes numerical and artifact consistency, not a cryptographic proof of the historical execution process.

An earlier execution under 3258b1d47baa734fea102d7f8ea38fdd626d8174 had six completed primary models and a stress model recorded as running. The experimenter requested a fresh execution; earlier results were not pooled with this submitted cohort. No separate approval of the additional workload was provided. This deviation is recorded explicitly in EXPERIMENT_LOG.md and submission/provenance/prior_execution_status.json for reviewer disposition. The earlier raw archive was not supplied for this review. No claim is made that this cohort was the first or only protected execution.

## 10. Conclusion

Under the frozen protocol, fixed RFF inputs yield a small, consistent primary relative-L2 improvement and lower boundary error, while increasing held-out PDE residual and worst-case solution error. At the stress difficulty, improvement is inconsistent and an adverse vanilla seed dominates favorable averages. The main scientific finding is that near-zero collocation loss can coexist with poor Helmholtz solution recovery and substantial residual away from training points. The trial provides transparent evidence of a limited repair and a generalization failure under the approved budget. All negative outcomes remain unchanged; no post-hoc tuning was performed.

## References and contribution

[1] Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707. DOI: 10.1016/j.jcp.2018.10.045.

[2] Tancik, M. et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. Advances in Neural Information Processing Systems, 33.

[3] PyTorch SobolEngine documentation. The committed arrays, not regeneration across library versions, define the sampled inputs. No external code or dataset was copied. The approved trial specifies the PDE, hypothesis and frozen choices; the implementation contribution is the auditable paired execution, deterministic assets, condition-gated evaluation and failure-preserving evidence. CITATIONS.md supplies source links.
