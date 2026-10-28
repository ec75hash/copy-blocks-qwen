# Qwen3 Base-reference follow-up

Using the published Base 14B instead of the non-Base 14B changes the primary mean similarities to the same Qwen3-32B target by less than 0.0002. The best matching layer stays the same for every source layer in both full key/value maps. The published Qwen3-30B-A3B-Base also has an elevated correspondence at a spacing of 12 layers.

[Measured results](RESULTS.md) · [Column definitions](COLUMNS.md) · [Corrections](CORRECTIONS.md) · [Shared glossary](../../docs/GLOSSARY.md)

## Question and inputs

Does the 14B/32B weight map change when the 14B reference is the published Base checkpoint? A second check measures repeated blocks inside the published Base 30B checkpoint. These are weight comparisons. The earlier [response study](../qwen3_response/README.md) is separate.

The recorded prediction was: “PT vs IT numbers will show PT retains higher cosine similarities between weights, but not neurons”. PT means the published Base checkpoint; IT means the published non-Base checkpoint. In this experiment, “neurons” refers to gate-weight rows, rather than responses to passages. The prediction is retained without a threshold or pass/fail score.

The four exact repository commits and config/index hashes are in [models.lock.json](models.lock.json). Here “IT 14B” is the pinned `Qwen/Qwen3-14B`; the fixed target is `Qwen/Qwen3-32B`. The new references are `Qwen/Qwen3-14B-Base` and `Qwen/Qwen3-30B-A3B-Base`. These names distinguish published artifacts; swapping a reference does not establish their construction order. All layer indices are zero-based original decoder indices; auxiliary and vision layers are excluded.

## Measurement

For the two dense 14B references and fixed 32B target, full attention key and value matrices (`k_proj`, `v_proj`) have matching shapes. Flattened cosine compares every coordinate of a tensor. Row cosine compares corresponding output rows individually. Six saved matrices cover all 40×64 cross-size layer pairs and all 40×40 Base/non-Base reference pairs.

Mapped-pair tables contain both 32-pair branches: 14B layers 0–31 to the same 32B indices, and 14B layers 8–39 to 32B layers 32–63. The primary summary uses the same 24 middle-block pairs as the earlier map: 14B layers 8–31 to 32B layers 32–55. Same-layer Base/non-Base calibration averages all 40 reference layers. Each summary mean gives equal weight to layer pairs.

The MLP gate check samples 544 indices from the 17,408 channels shared in width by both references: 17 contiguous clusters of 32 rows, seed 20261027. Eight clusters (256 rows) come from indices 0–8191 and nine (288 rows) from indices 8192–17407, proportional to their sizes. These are 17 sampled clusters, not 544 independently sampled neurons. The exact indices are in [FETCH_PLAN.json](FETCH_PLAN.json). The same indices are compared in every layer and checkpoint. Per-pair flattened gate cosine uses only this sample; per-row statistics compare corresponding 5,120-coordinate gate rows. No response measurements or uncertainty intervals are added.

Inside Base 30B, full key matrices and router matrices (`mlp.gate.weight`, which chooses experts) are compared over all 1,128 unordered pairs per family. The declared spacing K=12 and region 6–41 describe three blocks, 6–17, 18–29 and 30–41. Aligned pairs connect equal positions in those blocks. Within/off means use aligned versus remaining pairs at the same offset, for K and 2K. An offset must contain at least six pairs to be eligible for ranking. This uses the original survey scanner and does not add a significance test.

## Capture and checks

New capture read 1,187,512,320 tensor bytes through 856 byte-range requests and reused 312 previously captured full tensors. Each request uses a pinned shard, accepted only with HTTP 206, exact Content-Range and byte length; config/index hashes and per-payload SHA-256 are recorded. Computation uses float64 and checks normalized Gram diagonals within 1e-9 of one. No raw model tensors are bundled.

[Per-range provenance](provenance/fetched_payloads.json), [reused-tensor hashes](provenance/verified_legacy_payloads.json), [recorded metadata](provenance/metadata/) and [runtime](provenance/runtime.json) accompany the data. `FETCH_PLAN.json` is the final cached-rerun plan: its fetch-byte count is zero because the bytes had already been captured. Its tasks describe the full 856 new payloads; the initial total is stated above. Stored shard-header JSON files are decoded representations; the recorded raw-header hash refers to the original safetensors header bytes.

A separate implementation re-read and hash-checked the same cached bytes: all 856 new payloads and 312 reused tensors passed; the six K/V matrices had maximum absolute difference zero, and the gate and Base 30B summaries reproduced to the printed precision. This checks independent computation on the same inputs, rather than an independent download. See the [record](audit/same_bytes_check.json), [portable rerun](audit/portable_check.json), [portable checker changes](audit/portable_changes.json) and [capture validation](audit/capture_validation.json).

## Reproduce

The saved-data reduction needs Python 3.11+ and NumPy, no model, network or credentials. From the release root:

```bash
python studies/qwen3_base_followup/code/verify_and_report.py --out reproduction/qwen3_base_followup
```

It checks all 336 attention pairs against the six matrices, all 91,392 sampled-row values against the 168 gate-pair summaries, both layer maps, all 2,256 Base 30B pairs and 94 offsets, and regenerates this study's RESULTS.md. It checks saved-data consistency; it does not remeasure raw weights.

Optional raw-byte recomputation requires the full release (including the root scanner and prior Qwen3 provenance) and the retained raw caches, which are not distributed here:

```bash
python studies/qwen3_base_followup/code/check_cached_weights.py --tensor-cache /path/to/legacy-tensor-cache --range-cache /path/to/base-range-cache --out reproduction/base_cached_check.json
```

The legacy layout is `<slug>/<revision>/<sha256(tensor_name)>.bin`. The range cache uses the paths in `FETCH_PLAN.json` after removing the leading `cache/`, with same-stem `.json` sidecars. Pins, byte ranges, dtypes, shapes and hashes are recorded for a new capture; a fresh-download runner is not included. The checker itself performs no downloads and writes outside the study.
