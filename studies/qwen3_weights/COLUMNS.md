# Qwen3 weight-table columns

All layer and neuron indices are zero-based. See the [study](README.md),
[full measurement tables](WEIGHT_RESULTS.md) and [glossary](../../docs/GLOSSARY.md).
An empty numeric field means unavailable, not zero.

## `analysis/mlp_layer_pairs.csv` and `data/*_proj/pairs.csv`

The analysis table combines the three per-family pair tables and adds checkpoint
identifiers. Each family compares all pairs among 64 Qwen3-32B decoder layers.

| Column | Meaning |
|---|---|
| `model`, `revision` | Model repository and pinned commit; present in the combined analysis table. |
| `family` | MLP parameter-family suffix; present in the combined analysis table. |
| `layer_a`, `layer_b` | Decoder indices, with `layer_b > layer_a`. |
| `offset` | `layer_b - layer_a`. |
| `cosine` | Cosine of full tensors flattened into vectors. |
| `row_mean_cosine`, `row_median_cosine` | Mean and median of aligned per-neuron cosines. Neurons are stored rows for gate/up and columns for down; the scanner transposes down before computing these statistics. |
| `frac_row_gt_0.5`, `frac_row_gt_0.9` | Fraction of aligned per-neuron cosines strictly above 0.5 or 0.9; fractions, not percentages. |
| `shuffled_row_mean` | Mean per-neuron cosine after randomly permuting the target neuron indices for that layer pair. |
| `shuffled_row_p99` | 99th percentile of those shuffled per-neuron cosines. |
| `both_inside_region` | Literal region membership: both layer indices lie in 8–55 inclusive. This differs from the original survey's identically named aligned-membership field. |
| `corresponding` | True for `(layer_a, layer_b) = (L, L+24)` with L = 8…31. Use this field to select the 24 aligned block pairs. |

For example, (8,9) has `both_inside_region=True` and `corresponding=False` here.
The [original survey dictionary](../../docs/SURVEY_COLUMNS.md) describes its
different historical field meaning. Historical names are retained in both datasets.

## `data/*_proj/offsets.csv`

| Column | Meaning |
|---|---|
| `offset` | Decoder-layer separation. |
| `n_pairs` | Number of layer pairs at that separation. |
| `mean_cosine` | Mean flattened cosine across those pairs. |
| `mean_row_mean_cosine` | Mean of their aligned-neuron mean cosines. |

## `analysis/cross_checkpoint_gate_rows.csv`

Here model A is Qwen3-14B and model B is Qwen3-32B. These labels do not establish
which checkpoint came first. Only their first 17,408 gate rows are compared.

| Column | Meaning |
|---|---|
| `model_a`, `model_b` | Model repository identifiers. |
| `revision_a`, `revision_b` | Pinned commits. |
| `layer_a`, `layer_b` | Compared decoder indices in 14B and 32B. |
| `shared_rows` | Number of leading rows included in each tensor (17,408); “shared” describes the compared index range, not equal parameter values. |
| `flat_cosine_shared_rows` | Flattened cosine over the included rows. |
| `aligned_row_mean`, `aligned_row_median` | Mean and median of same-index row cosines. |
| `shuffled_row_mean` | Mean cosine after permuting the 32B rows. |

## Arrays and head labels

The K/V arrays have shape 40 × 64: row = 14B layer, column = 32B layer.
Per-neuron archives use keys such as `L8_L32` for aligned neuron-cosine vectors
and `L8_L32_shuffled` for their shuffled comparisons.

In the head tables, a KV group contains five query heads in 14B and eight in
32B. A slot is the head's zero-based position inside its group. The list
`[0,1,2,0,1,2,3,4]` maps each 32B slot to a 14B slot in the same group.
Head best matches and nearest-other-row searches use maximum absolute cosine;
the signed means in the tables retain their stated sign. Full sampled-row and
head definitions are given beside tables C, C2 and D in the measurement report.
