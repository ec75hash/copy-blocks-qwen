# Qwen3 reporting corrections

October 26, 2026: recomputation from saved tables and per-neuron vectors corrected
these summaries of the October 25 measurements. No stored model tensor,
per-neuron vector or layer-pair cosine was changed.

| Earlier summary | Corrected reading | Evidence |
|---|---|---|
| Fewer than one in 200 neuron-pair cosines exceeds 0.5; none exceeds 0.7 in gate or up | Fractions above 0.5 are 0.518880% gate, 0.194010% up and 0.507161% down. Above 0.7 there are 109 gate, 27 up and 48 down observations, out of 614,400 per family. | Saved vectors and tables A in WEIGHT_RESULTS.md |
| Cross-checkpoint aligned gate cosine is 0.67–0.70 throughout the second block and tail | Values vary by layer: 0.276 at 14B L8 / 32B L32 and 0.692 at L31/L55; 0.67–0.70 describes the later part, not every pair. | Table C and analysis/cross_checkpoint_gate_rows.csv |
| Near-verbatim weights or an exact neuron-copy recipe | Cosines measure similarity, not equality; same-index comparisons alone do not prove every row is its nearest counterpart. C2 samples 400 higher-index rows at each listed layer and retains early-layer exceptions. | Tables C and C2 |

`WEIGHT_RESULTS.md` and the current study `README.md` state these corrected
summaries. The saved-output and selected cached-payload check is recorded in
`audit/validation.json`; it is not a fresh full-model replication.

The current page also distinguishes two different searches: section C uses
300 sampled rows for maximum-absolute-cosine searches against 14B rows; section C2
uses 400 sampled higher-index rows against other rows. Section D compares all
64 32B heads at each of seven listed layer pairs per projection family in float64.
The section-C/C2 row searches use float32 normalization and matrix products.
