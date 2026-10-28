# Qwen2.5 weight-table columns

These are cross-checkpoint comparisons: source/reference is Qwen2.5-14B and target
is Qwen2.5-32B. The names do not establish chronological order. Layer indices are
zero-based. See the [study](README.md) and [glossary](../../docs/GLOSSARY.md).

## Shared labels

- `family`: parameter tensor suffix, such as `self_attn.k_proj.weight`.
- `transform`: `raw` means stored parameter values; `minus_init` means subtract one
  from each RMSNorm gain before all comparisons and residual calculations.
- Pair-table `map_group`: `prefix` maps 14B 0–15 to 32B 0–15; `twin_a` maps 14B
  16–31 to 32B 16–31; `twin_b` maps 14B 16–31 to 32B 32–47; `suffix` maps 14B
  32–47 to 32B 48–63. Blank means outside this specified map.
- Summary-table `map_group` writes the same four mappings as explicit ranges
  (with “twin A” or “twin B” on the middle two rows).

## `analysis/cross_checkpoint_pairs.csv`

| Column | Meaning |
|---|---|
| `source_model`, `target_model` | Repository identifiers for 14B and 32B. |
| `source_revision`, `target_revision` | Pinned commits for those repositories. |
| `family`, `transform`, `map_group` | Labels defined above. |
| `source_layer`, `target_layer` | Decoder indices in 14B and 32B. |
| `cosine` | Cosine between the two transformed, flattened tensors. |

Every family contains all 48 × 64 = 3,072 layer comparisons; 64 are on the map.

## `analysis/mapping_summary.csv`

| Column | Meaning |
|---|---|
| `family`, `transform`, `map_group` | Parameter family, transform and explicit range mapping. |
| `n_pairs` | Number of mapped pairs in this group (16). |
| `mean_cosine`, `min_cosine`, `max_cosine` | Mean and extrema over these 16 pairs. |
| `best_rank`, `worst_rank` | Minimum and maximum mapped-target rank across the group. For each 14B layer, rank is `1 + number of 32B layers with strictly greater cosine`, among all 64 targets; 1 is strongest. |

## Reference residuals

For each 14B layer L = 16…31, let P be its transformed tensor, A the transformed
32B tensor at L, and B the transformed 32B tensor at L+16. Define
`dA = A - P` and `dB = B - P`. The norm below is the Euclidean norm of the
flattened tensor. None of these quantities dates a training update.

## `analysis/endpoint_residuals.csv`

| Column | Meaning |
|---|---|
| `family`, `transform` | Parameter family and transform. |
| `reference_layer` | Layer L in the 14B reference. |
| `target_layer_a`, `target_layer_b` | Layers L and L+16 in the 32B. |
| `relative_distance_a` | `norm(dA) / norm(P)`. |
| `relative_distance_b` | `norm(dB) / norm(P)`. |
| `endpoint_residual_cosine` | `cosine(dA, dB)`. |
| `twin_cosine` | `cosine(A, B)`, comparing the two 32B tensors directly. |

## `analysis/family_summary.csv`

| Column | Meaning |
|---|---|
| `family`, `transform` | Parameter family and transform. |
| `off_map_pairs` | Number of pairs outside the specified map (3,008). |
| `off_map_max_abs_cosine` | Largest absolute cosine over those pairs. |
| `off_map_mean_cosine` | Signed mean cosine over those pairs. |
| `reference_relative_distance_a`, `reference_relative_distance_b` | Arithmetic means of the corresponding relative distances over reference layers 16–31. |
| `endpoint_residual_cosine` | Arithmetic mean of the 16 per-layer residual cosines; not cosine of an averaged residual. |

The original capture JSON preserves older keys such as `parent_layer`, `drift`
and `rel_drift_a`. They refer to the same reference-based calculations; these
historical names do not establish parentage or chronology.
