# Base follow-up corrections and packaging

The initial capture reached the Base 30B reduction, then stopped because the adapter supplied two-dimensional tensors to a scanner interface that expects flattened vectors plus the original shape. The adapter was corrected and covered by a synthetic integration test. The complete final run reused the captured bytes with zero new tensor downloads. No numerical results from the failed run are presented as final results.

The packaged data are copied unchanged from the completed run. Its saved fetch plan describes the cached rerun, so its remaining-download count is zero; the initial 1,187,512,320-byte capture total is stated in the study README. The original independent recomputation was adapted to explicit cache/output arguments; mathematical comparison functions are unchanged and hashes/changes are recorded in `audit/portable_changes.json`. Its source-time result is retained in `audit/same_bytes_check.json`.

This is a new separate study in v0.3.0. Earlier survey, weight-map and response measurements are unchanged. There is no prediction score or added significance claim.
