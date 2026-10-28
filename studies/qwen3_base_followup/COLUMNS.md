# Base follow-up columns and labels

[Study](README.md) · [Shared glossary](../../docs/GLOSSARY.md)

## Cross-checkpoint CSVs

`data/attention_mapped_pairs.csv` has 336 rows; `data/gate_mapped_pairs.csv` has 168. `family` (attention table only) names the key or value tensor. `comparison` is `base14_to_32` (Base 14B to fixed 32B), `it14_to_32` (non-Base 14B to that same target), or `base14_to_it14` (the two 14B references).

`source_layer` and `target_layer` are zero-based original decoder indices. `map_group` is `same_index` for source 0–31 to identical target indices, `plus24` for source 8–39 to target source+24, and `same_layer` for reference calibration at layers 0–39. The primary summaries select `plus24` with source 8–31. These labels describe measured positions, not chronology.

`flat_cosine` is dot(a,b)/(norm(a)×norm(b)) after flattening the full attention tensor or the 544-row gate sample. `row_mean` and `row_median` summarize cosines of aligned output rows. `valid_rows` counts rows with finite cosine and nonzero norms; `total_rows` is the number attempted. Attention tensors have 1,024 output rows; gate samples have 544.

`data/gate_neuron_pairs.csv` has 91,392 rows. `comparison`, `source_layer`, `target_layer` have the same meanings. `neuron` is the original gate-row index (one of the 544 sampled indices). `row_cosine` compares those two 5,120-coordinate rows. It is a weight similarity, not an activation correlation. Each comparison/layer pair has exactly 544 rows.

## Cross-checkpoint arrays and summary

Six `data/{k_proj,v_proj}_{base14_to_32,it14_to_32,base14_to_it14}.npy` arrays store flattened cosines. Axis 0 is the 14B source layer; axis 1 is the target layer. Shapes are 40×64 for the 32B target and 40×40 for reference calibration. There are no auxiliary-layer indices.

`data/summary.json` contains the primary 24-pair means and Base-minus-IT differences, same-layer 40-pair calibration means, and `base_best_targets`/`it_best_targets` (argmax target per source layer, computed over every target). `sampled_gate` stores the 544-row sample size, 24-pair summaries and calibration; `row_median` there means the mean of per-pair medians. Each layer pair has equal summary weight. The Base 30B entries use the original scanner's summary schema, including `periods` at 12 and 24.

## Within-Base-30B CSVs

`data/base30_within_pairs.csv` has 2,256 rows. `family` is the attention key tensor or expert router (`mlp.gate.weight`). `layer_a < layer_b` identifies an unordered layer pair; `offset = layer_b - layer_a`. `cosine` is full flattened tensor cosine; `row_mean_cosine` averages aligned output-row cosines.

`both_inside_region` means membership in the declared **aligned block-pair set**, as in the root survey, not simply that both indices lie between 6 and 41. True pairs are a=6–17,b=a+12 or a+24, and a=18–29,b=a+12. This differs from the literal region-membership meaning used by the separate Qwen3 MLP scan. Comparisons outside this aligned set at the same offset supply the off-block means.

`data/base30_offsets.csv` has 94 rows. `offset` and `family` have the same meaning; `n_pairs` counts layer pairs at that offset. `mean_cosine` averages their full tensor cosines; `mean_row_mean_cosine` averages their aligned-row means. `rank_eligible` is True when `n_pairs >= 6`. Offsets are ranked by `mean_cosine`; no empirical p-value is computed here.

## Pins and provenance

`models.lock.json` pins repository revisions and config/index hashes. `FETCH_PLAN.json` records the seed, exact sampled indices and task byte ranges; `payload_bytes_to_fetch` describes that saved run's remaining downloads (zero for the final cached rerun). A `row_range` uses half-open bounds [start,stop); null means a full tensor. Byte ranges are inclusive HTTP byte endpoints. Shapes and dtypes describe the decoded payload, and `cache_path` identifies a local raw cache file not bundled here.

`provenance/fetched_payloads.json` records the captured payload SHA-256, byte count, raw-header hash, source shard and selected tensor/slice for each new range. `provenance/verified_legacy_payloads.json` records reused cache paths and hashes. Metadata JSON files preserve config/index content and decoded shard headers. Decoded header-file formatting is not the original safetensors header byte sequence. `audit/` contains validation and same-input recomputation records; these are checks of the recorded measurement, not additional model experiments.
