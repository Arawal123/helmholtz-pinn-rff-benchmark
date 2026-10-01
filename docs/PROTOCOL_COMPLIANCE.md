# Requirements coverage

The master checklist is mapped below. Technical completion does not imply approval of the disclosed additional-workload deviation.

| Sections | Requirement | Evidence | Status |
|---|---|---|---|
| 1–2 | Task and approved Helmholtz problem | src/pde.py; configs/frozen.yaml; report sections 1–2 | Implemented; n=6 primary and n=12 stress retained |
| 3–4 | Fixed collocation and boundary inputs | data/frozen; src/data.py; raw config/hash provenance | Verified reuse; 2,048 interior and 512 unique boundary points |
| 5–6 | Held-out grid and normalized loss | src/evaluate.py; src/pde.py; run timestamps | 257×257 grid and condition gates verified |
| 7–9 | Models, RFF and seeds | src/models.py; artifacts/rff; all raw weights | Counts 12,737 / 12,871; fixed B; every seed retained |
| 10 | Protected workload | run_manifest.csv; EXPERIMENT_LOG.md | 12 submitted slots; earlier execution disclosed as an unapproved additional-workload deviation |
| 11–13 | Compute, stopping and frozen settings | src/train.py; run.json; compute_record.csv | 8,000 updates; fixed settings; no retry within submitted cohort |
| 14–16 | Metrics and paired descriptive effects | per_seed_results.csv; paired_effects.csv; summary.csv | All metrics, per-seed values, effects and sample SD retained |
| 17 | Sensitivity scope | configs/frozen.yaml; report section 8 | Seed spread reported; no independent hyperparameter sweep |
| 18–19 | Raw outputs and provenance | results/raw; submission/execution; original-file map | Raw data and scientific source SHA preserved |
| 20–21 | Tables and visualizations | results/processed; figures | Five tables, 16 original figures and clearly labeled additional figures |
| 22 | Methods/results note | report/methods_results.md and .pdf | Comprehensive methods, actual conclusions and restrained scope |
| 23 | Experiment log | EXPERIMENT_LOG.md; original journal copy | Final hardware/completion/failures/decisions entry completed from evidence |
| 24–25 | Attribution and resource use | CITATIONS.md; requirements.txt; runtime records | No external code/data copied or required private/paid service; account billing tier is not established by GPU metadata |
| 26–27 | Compute sufficiency and structure | All run records; repository tree | Full workload retained; all required source/output paths present |
| 28–29 | Reproduction and Colab | README.md; docs/COLAB_EXECUTION.md; exact CLI | Scientific SHA and source/launcher distinction explicit; actual Drive workflow disclosed |
| 30 | Smoke separation | configs/smoke.yaml; scripts/run_smoke.py; tests | Smoke is non-protected and separate from reported evidence |
| 31 | Delivery | README.md; report; submission overview; all results | Complete static technical submission with evidence and provenance |
| 32 | Scientific integrity | Original-file map; checksums; experiment log | No changed results or post-hoc tuning; historical workload exception remains for reviewer disposition |

All negative outcomes are retained. No new training, sweep or compensatory run was performed during submission preparation.
