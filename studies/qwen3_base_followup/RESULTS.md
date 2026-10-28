# Base-checkpoint follow-up: measured results

This separate weight study changes the 14B reference while keeping Qwen3-32B fixed. It does not measure neuron responses.

The recorded prediction was: “PT vs IT numbers will show PT retains higher cosine similarities between weights, but not neurons”. Here PT means the published Base checkpoint, IT means the published non-Base checkpoint, and the neuron measurement is sampled gate-weight rows. No threshold or prediction score is assigned.

The primary comparison uses 24 layer pairs: 14B layers 8–31 against 32B layers 32–55. Both references use the same 544 gate indices in 17 contiguous 32-row clusters (seed 20261027).

| Quantity | Base 14B reference | IT 14B reference | Base minus IT |
|---|---:|---:|---:|
| self_attn.k_proj.weight flattened cosine | 0.701420 | 0.701606 | -0.000186 |
| self_attn.v_proj.weight flattened cosine | 0.806361 | 0.806466 | -0.000105 |
| Sampled gate aligned-row mean | 0.584147 | 0.584247 | -0.000099 |

The mean changes are less than 0.0002. The best matching target layer for every source layer is unchanged in both full K/V matrices. These measurements do not determine construction order or shared activation features. The gate result covers the sampled rows.

## Reference calibration

| Base 14B versus IT 14B, same layer | Mean cosine |
|---|---:|
| self_attn.k_proj.weight flattened | 0.996146 |
| self_attn.v_proj.weight flattened | 0.997693 |
| Sampled gate aligned-row mean | 0.996124 |

## Published Base 30B checkpoint, within itself

The declared spacing K is 12 layers. The three blocks are layers 6–17, 18–29 and 30–41 (zero-based, inclusive). At K=12 there are 24 corresponding pairs and 12 other pairs. This is a within-checkpoint comparison, separate from the 14B/32B map.

| Family | Within K=12 | Other K=12 pairs | Aligned-row within | Aligned-row other | Top eligible offset |
|---|---:|---:|---:|---:|---:|
| self_attn.k_proj.weight | 0.535278 | -0.000056 | 0.478648 | 0.000206 | 12 |
| mlp.gate.weight | 0.880615 | 0.329110 | 0.885117 | 0.336251 | 12 |

All reported means weight layer pairs equally. Eligible offsets contain at least six pairs; this is a reporting rule, not a significance test. Full pairs, both layer maps, row values and denominators are in [data](data/); see [column definitions](COLUMNS.md) and [methods and reproduction](README.md).
