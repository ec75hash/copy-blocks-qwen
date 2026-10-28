# Original survey column definitions

See [METHODS](../METHODS.md) for the measurement and reporting rules and the
[glossary](GLOSSARY.md) for terms. These tables compare layers within each checkpoint.
An empty numeric field means unavailable, not zero. Indices are zero-based.

## `data/pairs.csv`

| Column | Meaning |
|---|---|
| `model` | Model repository identifier. |
| `model_slug` | Short identifier used in the lock and local paths. |
| `revision` | Pinned model-repository commit. |
| `tensor` | Parameter-family suffix; the same family is compared across layers. |
| `layer_a`, `layer_b` | Original decoder indices of the two layers, with `layer_b > layer_a`. |
| `offset` | `layer_b - layer_a`. |
| `cosine` | Full-tensor flattened cosine. |
| `row_mean_cosine` | Mean cosine over aligned rows, when available. |
| `both_inside_region` | Historical name for membership in the aligned pair set derived from the lock's segments and pair groups. It does **not** merely test whether both layer numbers lie inside the overall region. |

For example, (8,32) is an aligned Qwen3-32B pair; (8,9) is not, even though both
indices lie inside its declared region. This field differs from the identically
named field in the [Qwen3 MLP scan](../studies/qwen3_weights/COLUMNS.md).

## `data/offsets.csv`

| Column | Meaning |
|---|---|
| `model`, `model_slug`, `revision`, `tensor` | Checkpoint and parameter-family identifiers as above. |
| `offset` | Layer separation. |
| `n_pairs` | Number of available tensor pairs at this separation. |
| `mean_cosine` | Mean flattened cosine across these pairs. |
| `mean_row_mean_cosine` | Mean of their aligned-row mean cosines, when available. |
| `rank_eligible` | True when at least six pairs are available; controls inclusion in offset rankings. |

`models.lock.json` also preserves historical `note` annotations about selection
and earlier projects. Those notes are not inputs to the numerical calculation.
