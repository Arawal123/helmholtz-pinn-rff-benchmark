# Submission integrity

The scientific source SHA is `0677b40869fabf50035075e240aea3b98a0c7e05`. The original archive is identified by SHA-256 in `provenance/original_file_map.json`. Every original file is preserved byte-for-byte either at its active path or in a historical provenance copy. Updated documentation and supplementary figures are additional editorial artifacts.

`checksums.sha256` covers every distributed file except itself. `scripts/verify_submission.py` checks those hashes and the original-file mapping, then verifies completed run records, evaluation chronology, frozen asset hashes and table arithmetic. No optimization is performed. Optional CPU model evaluation uses saved final weights and reports float32 backend differences.

The original manuscript and execution journal are retained under `provenance/frozen_source/report/` and `provenance/original_execution_log.md`. The final experiment log is an explicitly retrospective evidence-based record. The executed notebook and environment/console logs are retained under `execution/`.

Prior protected execution is represented only by the status record supplied by the experimenter. Its original raw archive was not supplied, and reviewer approval for the additional execution was not provided. This limitation is disclosed rather than treated as a completed audit of all historical runs.
