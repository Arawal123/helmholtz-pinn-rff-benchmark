# Experiment log

## Hypothesis and frozen design

Fixed random Fourier features were proposed to reduce high-frequency approximation difficulty in a tanh PINN. The manufactured Helmholtz condition n=6 is primary; n=12 is the preregistered stress/falsification case. The approved values are recorded in configs/frozen.yaml. Neither the conditions nor the settings were changed in response to the submitted results.

## Preparation and interface changes

The baseline source revision was 3258b1d47baa734fea102d7f8ea38fdd626d8174. Revision 0677b40869fabf50035075e240aea3b98a0c7e05 introduced separate training/evaluation/finalization commands and training progress output. The PDE, architectures, loss, optimizer, schedule, point sets, seeds and update budget were unchanged. Frozen Sobol/RFF assets were committed before the submitted execution. Historical source documentation and the original execution journal are retained under submission/provenance/.

## Execution history and deviation

Before the submitted cohort, an execution-status record showed all six primary models completed at 8,000 updates under 3258b1d47baa734fea102d7f8ea38fdd626d8174, and a stress model with a running status. Its subsequent progress and final status are not established by the available record. The experimenter then requested a new execution from the beginning in a separate Drive directory. The submitted cohort contains 12 complete models under 0677b40869fabf50035075e240aea3b98a0c7e05; earlier outputs were not pooled, replaced or used to select checkpoints in this cohort.

The experimenter confirmed that the new execution did not have separate approval from Ryan. This is an additional-workload deviation from the approved execution history and is disclosed for reviewer disposition. The earlier raw output archive was not supplied for this review; its status record is retained in submission/provenance/prior_execution_status.json. No assertion of full historical compliance or approval is made.

## Submitted execution: 1 October 2026

| Field | Evidence |
|---|---|
| Scientific source SHA | 0677b40869fabf50035075e240aea3b98a0c7e05 |
| Executed launcher source SHA | de869e2d240439cbe24a16d96478e7e824322625 |
| Working tree during each run | Clean, recorded in all 12 run.json files |
| Accelerator | Tesla T4; one GPU; 15,637,086,208 bytes reported device memory |
| Python / PyTorch / CUDA | Python 3.13.15; PyTorch 2.12.0+cu130; CUDA 13.0 |
| Precision / deterministic configuration | float32; deterministic algorithms; TF32 disabled; CUBLAS_WORKSPACE_CONFIG=:4096:8 |
| First training start, UTC | 2026-10-01T09:46:12.082440+00:00 |
| Last training end, UTC | 2026-10-01T10:20:49.997165+00:00 |
| Final evaluation timestamp, UTC | 2026-10-01T10:20:59.274825+00:00 |
| Execution span, IST | 15:16:12 to 15:50:59 on 1 October 2026 |
| Intended slots / completed training / completed evaluation | 12 / 12 / 12 |
| Updates per model | 8,000 |
| Measured training time | 1969.857 s (32.83 min) |
| Measured evaluation time | 11.592 s |
| First training to final evaluation | 2087.192 s (34.79 min) |
| Numerical failures within submitted cohort | None recorded |
| Infrastructure-interrupted attempts within submitted cohort | None recorded |
| Retries within submitted cohort | None; one attempt_000 directory for each of the 12 slots |
| Earlier execution | Separate completed primary cohort and stress attempt, disclosed above |
| Frozen scientific changes during submitted execution | None observed in source, configs, asset hashes or provenance |

All six n=6 training records are terminal before the first primary evaluation. All six n=12 training records are terminal before the first stress evaluation. Stress training starts after primary evaluation completes. Final weights are those after update 8,000; no earlier checkpoint was selected.

## Observations and final decision

Primary relative L2 error decreased by a mean paired 5.04% with RFF, with improvement in all three pairs. Mean relative L2 remained 0.94988. RFF increased held-out PDE residual and maximum absolute error while reducing boundary error. At n=12, RFF improved relative L2 in one of three pairs; the favorable mean was driven by one poor vanilla result. The numerical outcomes and all seeds were retained unchanged.

Post-run CPU evaluation found a large RFF residual gap between training points and the held-out grid. This diagnostic used saved final weights without optimization. It did not influence training, checkpoint selection, feature selection or protocol changes.

Decision: retain the complete submitted evidence, report limited primary improvement and stress failure, and perform no post-hoc tuning or replacement training. The additional-workload deviation remains a matter for reviewer disposition.

## Submission preparation record

This final entry was assembled after execution on 1 October 2026 from the supplied raw records, tables, console logs and subsequent experimenter confirmations. It is not presented as a contemporaneous pre-run entry. The original journal is preserved byte-for-byte. Presentation changes affect documentation and supplementary figures only; raw data, saved weights, original figures and processed numerical tables are unchanged.
