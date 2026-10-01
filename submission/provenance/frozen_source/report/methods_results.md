# Methods and results — Ryan PINN failure-and-repair trial

## Status

PROTECTED RUN COMPLETE

## Problem and preregistered hypothesis

On Ω=[-1,1]², solve Δu+20²u=f with zero Dirichlet data and u*=sin(nπx)sin(nπy). The hypothesis is that fixed random Fourier features mitigate spectral bias of the tanh PINN on primary n=6. n=12 remains the preregistered stress/failure condition.

## Frozen methods

For each seed 0,1,2, both methods and both n values reuse the same committed 2,048-point scrambled Sobol set. The 512 boundary points are fixed. Both models use the normalized PDE residual plus boundary MSE, full-batch float32 Adam for exactly 8,000 updates, and cosine LR 1e-3 to 1e-4. Vanilla is 2→64→64→64→64→1 tanh (12,737 parameters); repair is a fixed seed-2026 32×2 Gaussian B (σ=3) with sin/cos features then 64→55→55→55→55→1 tanh (12,871 parameters). There is no adaptive weighting, resampling, early stopping, best-checkpoint selection, or post-hoc tuning.

Held-out 257×257 grid metrics are evaluated only after all six training runs at that n are terminal. Scientific failures are retained without restart; infrastructure interruptions have separate attempts.

## Per-seed results

| n / role | method | seed | status | rel L2 | L∞ | residual RMS | boundary RMSE | train s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 6 / primary | vanilla | 0 | completed | 1.00035 | 1.01827 | 0.49803 | 0.00763266 | 148.047 |
| 6 / primary | vanilla | 1 | completed | 1.00028 | 1.01078 | 0.497948 | 0.00784663 | 146.585 |
| 6 / primary | vanilla | 2 | completed | 1.00039 | 1.01932 | 0.498049 | 0.00756946 | 145.95 |
| 6 / primary | rff | 0 | completed | 0.954105 | 1.19423 | 0.798524 | 0.00132888 | 184.283 |
| 6 / primary | rff | 1 | completed | 0.947476 | 1.30538 | 0.791686 | 0.00120856 | 184.223 |
| 6 / primary | rff | 2 | completed | 0.948059 | 1.17295 | 0.775706 | 0.00115072 | 181.634 |
| 12 / preregistered_stress_failure | vanilla | 0 | completed | 1.00126 | 1.0542 | 0.49807 | 0.00246777 | 147.515 |
| 12 / preregistered_stress_failure | vanilla | 1 | completed | 1.94799 | 4.74922 | 0.519844 | 0.0382912 | 144.944 |
| 12 / preregistered_stress_failure | vanilla | 2 | completed | 1.00211 | 1.07556 | 0.49808 | 0.00355175 | 145.912 |
| 12 / preregistered_stress_failure | rff | 0 | completed | 1.03175 | 1.94818 | 0.607655 | 0.000786944 | 182.795 |
| 12 / preregistered_stress_failure | rff | 1 | completed | 1.03876 | 2.05628 | 0.593025 | 0.000436598 | 177.68 |
| 12 / preregistered_stress_failure | rff | 2 | completed | 1.06209 | 1.93875 | 0.571373 | 0.00140624 | 180.287 |

## Paired descriptive effects

Difference is RFF − vanilla; improvement is 100×(vanilla − RFF)/vanilla when vanilla is nonzero. Negative improvement means repair lost. These are descriptive comparisons across three seeds, not significance tests.

| n | seed | metric | difference | improvement % |
|---:|---:|---|---:|---:|
| 6 | 0 | relative_l2 | -0.04624333136026926 | 4.622721589863214 |
| 6 | 0 | linf | 0.17595946788787842 | -17.2802512614128 |
| 6 | 0 | normalized_residual_rms | 0.30049339403322134 | -60.336383045455264 |
| 6 | 0 | boundary_rmse | -0.006303783727134936 | 82.58958233053424 |
| 6 | 1 | relative_l2 | -0.052802357345252404 | 5.2787660101769385 |
| 6 | 1 | linf | 0.29460287000983953 | -29.146180423395517 |
| 6 | 1 | normalized_residual_rms | 0.29373786797777796 | -58.98961227217733 |
| 6 | 1 | boundary_rmse | -0.006638071307623029 | 84.59775424863234 |
| 6 | 2 | relative_l2 | -0.05233549163290374 | 5.231484106562941 |
| 6 | 2 | linf | 0.15363050252199173 | -15.071875486689851 |
| 6 | 2 | normalized_residual_rms | 0.27765649945233756 | -55.74882322047388 |
| 6 | 2 | boundary_rmse | -0.006418736227490429 | 84.79781904217573 |
| 12 | 0 | relative_l2 | 0.030494877954334143 | -3.045656720318601 |
| 12 | 0 | linf | 0.8939817063510418 | -84.8019853900284 |
| 12 | 0 | normalized_residual_rms | 0.10958493498003613 | -22.001894489614322 |
| 12 | 0 | boundary_rmse | -0.0016808220380035268 | 68.11107756065411 |
| 12 | 1 | relative_l2 | -0.9092338794308341 | 46.67538171340325 |
| 12 | 1 | linf | -2.692945420742035 | 56.7028711012697 |
| 12 | 1 | normalized_residual_rms | 0.07318122019979023 | -14.077532215053317 |
| 12 | 1 | boundary_rmse | -0.03785456059817368 | 98.85979532853875 |
| 12 | 2 | relative_l2 | 0.0599809986971469 | -5.985458980356233 |
| 12 | 2 | linf | 0.8631931841373444 | -80.2555574160454 |
| 12 | 2 | normalized_residual_rms | 0.07329281322384423 | -14.715059828773954 |
| 12 | 2 | boundary_rmse | -0.002145510049662353 | 60.40708128808413 |

## Mean and sample SD by condition/method

| n | method | metric | valid/planned | mean | SD |
|---:|---|---|---:|---:|---:|
| 6 | vanilla | relative_l2 | 3/3 | 1.000340590804869 | 5.8570793541181696e-05 |
| 6 | vanilla | linf | 3/3 | 1.016121723068257 | 0.0046583938702797305 |
| 6 | vanilla | normalized_residual_rms | 3/3 | 0.49800923848321177 | 5.3474419422203426e-05 |
| 6 | vanilla | boundary_rmse | 3/3 | 0.007682916482861065 | 0.00014525776405019766 |
| 6 | rff | relative_l2 | 3/3 | 0.9498801973587273 | 0.003670631407587076 |
| 6 | rff | linf | 3/3 | 1.2241860032081604 | 0.07111627661883585 |
| 6 | rff | normalized_residual_rms | 3/3 | 0.788638492304324 | 0.01171034540266031 |
| 6 | rff | boundary_rmse | 3/3 | 0.0012293860621116006 | 9.088581721732186e-05 |
| 6 | paired_absolute_difference | relative_l2 | 3/3 | -0.0504603934461418 | 0.003659535540942248 |
| 6 | paired_percentage_improvement | relative_l2 | 3/3 | 5.044323902201031 | 0.3658828732378528 |
| 6 | paired_absolute_difference | linf | 3/3 | 0.20806428013990322 | 0.0757716393255783 |
| 6 | paired_percentage_improvement | linf | 3/3 | -20.499435723832722 | 7.569271848411758 |
| 6 | paired_absolute_difference | normalized_residual_rms | 3/3 | 0.29062925382111227 | 0.011731519509674201 |
| 6 | paired_percentage_improvement | normalized_residual_rms | 3/3 | -58.35827284603549 | 2.35804333933807 |
| 6 | paired_absolute_difference | boundary_rmse | 3/3 | -0.006453530420749465 | 0.00016983822753007116 |
| 6 | paired_percentage_improvement | boundary_rmse | 3/3 | 83.99505187378077 | 1.2212759549541197 |
| 12 | vanilla | relative_l2 | 3/3 | 1.317121445921876 | 0.5463522783133186 |
| 12 | vanilla | linf | 3/3 | 2.2929923397799334 | 2.127184378514395 |
| 12 | vanilla | normalized_residual_rms | 3/3 | 0.5053316136670237 | 0.012568180057834881 |
| 12 | vanilla | boundary_rmse | 3/3 | 0.014770225585546894 | 0.0203769344850453 |
| 12 | rff | relative_l2 | 3/3 | 1.0442021116620916 | 0.015885172214412695 |
| 12 | rff | linf | 3/3 | 1.9810688296953838 | 0.06530262266100914 |
| 12 | rff | normalized_residual_rms | 3/3 | 0.5906846031349139 | 0.018254044408886064 |
| 12 | rff | boundary_rmse | 3/3 | 0.0008765946902670404 | 0.0004909997510255155 |
| 12 | paired_absolute_difference | relative_l2 | 3/3 | -0.2729193342597844 | 0.5512617419396115 |
| 12 | paired_percentage_improvement | relative_l2 | 3/3 | 12.548088670909472 | 29.59163238380427 |
| 12 | paired_absolute_difference | linf | 3/3 | -0.3119235100845496 | 2.0620829245082004 |
| 12 | paired_percentage_improvement | linf | 3/3 | -36.11822390160137 | 80.41756188780947 |
| 12 | paired_absolute_difference | normalized_residual_rms | 3/3 | 0.0853529894678902 | 0.02098555457281178 |
| 12 | paired_percentage_improvement | normalized_residual_rms | 3/3 | -16.931495511147197 | 4.4026491701241 |
| 12 | paired_absolute_difference | boundary_rmse | 3/3 | -0.013893630895279853 | 0.020752074544664434 |
| 12 | paired_percentage_improvement | boundary_rmse | 3/3 | 75.79265139242567 | 20.34472257537473 |

## Convergence, compute, failures, and seed sensitivity

| n | method | seed | first logged loss | last logged loss | last logged update |
|---:|---|---:|---:|---:|---:|
| 6 | vanilla | 0 | 0.5750411748886108 | 0.24999235570430756 | 8000 |
| 6 | vanilla | 1 | 0.2661566734313965 | 0.24980707466602325 | 8000 |
| 6 | vanilla | 2 | 0.29115915298461914 | 0.24972248077392578 | 8000 |
| 6 | rff | 0 | 2.16682767868042 | 1.7796547808757168e-06 | 8000 |
| 6 | rff | 1 | 2.630925416946411 | 1.4739038078914746e-06 | 8000 |
| 6 | rff | 2 | 1.8918583393096924 | 1.4015620308782673e-06 | 8000 |
| 12 | vanilla | 0 | 0.42371752858161926 | 0.24998091161251068 | 8000 |
| 12 | vanilla | 1 | 0.2588055431842804 | 0.16859734058380127 | 8000 |
| 12 | vanilla | 2 | 0.2723139524459839 | 0.2499413937330246 | 8000 |
| 12 | rff | 0 | 0.46687185764312744 | 5.437328240986972e-07 | 8000 |
| 12 | rff | 1 | 0.5436100959777832 | 1.847549100375545e-07 | 8000 |
| 12 | rff | 2 | 0.40172943472862244 | 1.1417773748689797e-06 | 8000 |

Measured training wall time across selected attempts: 1969.857 s for 12/12 slots. Per-attempt compute and actual accelerators are in `results/processed/compute_record.csv`.

Scientific failures: none recorded

n=6 paired primary-metric comparison: RFF lower error in 3/3, higher error in 0/3 available pairs; see every seed above.
n=12 paired primary-metric comparison: RFF lower error in 1/3, higher error in 2/3 available pairs; see every seed above.

Seed sensitivity is the per-seed spread and sample SD in the tables above; three seeds support descriptive interpretation only.

## Interpretation and scope

Training convergence and error fields are in figures/. Compute, hardware, and failure records are in results/processed/ and the per-attempt raw directories. Per-seed variability is the approved seed sensitivity evidence. An independent hyperparameter sweep was not part of the approved protected protocol and was not added post hoc. This limits claims about hyperparameter sensitivity. Preserve negative, divergent, or inconclusive outcomes.

## Reproducibility

Exact command: `python scripts/run_frozen.py --config configs/frozen.yaml`. Frozen asset hashes and resolved configs appear in raw provenance. GPU floating-point execution may vary across accelerator and library versions despite deterministic seeds and algorithm settings.

Git SHA: `0677b40869fabf50035075e240aea3b98a0c7e05`. Accelerator: `Tesla T4`. PyTorch: `2.12.0+cu130`. CUDA: `13.0`.
