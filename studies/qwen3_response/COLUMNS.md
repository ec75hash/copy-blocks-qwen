# Qwen3 response-table columns

These tables describe the fixed Q8 collection, not weight cosines. See the
[study](README.md), [complete readout](READOUT.txt) and [glossary](../../docs/GLOSSARY.md).
All layer/neuron indices are zero-based. A passage in the retained field names
means one stored 256-token text window, not an independent article.

## Segments, components and signs

`segment` is a neuron-index range: A = 0–8191, B = 8192–17407,
C = 17408–25599. Four indices per segment are selected for each source layer;
equal segment sampling is not weighted by the number of neurons in each range.

| `component` | Measured scalar at each token |
|---|---|
| `gate_pre_silu` | Gate-projection output before the SiLU nonlinearity. |
| `value` | Up/value-projection output. |
| `activation` | `SiLU(gate_pre_silu) * value`, before the down projection. |

The four sign regimes describe nonzero gate/value signs at the same token.
SiLU preserves the gate's sign, so the product's sign follows this table.

| `sign_regime` label | Gate sign | Value sign | Activation sign |
|---|---|---|---|
| `positive_gate_positive_value` | + | + | + |
| `positive_gate_negative_value` | + | − | − |
| `negative_gate_positive_value` | − | + | − |
| `negative_gate_negative_value` | − | − | + |

`response_group` has six values: the four sign-regime labels above, plus
`positive_all_regimes` (the two positive-product regimes combined) and
`negative_all_regimes` (the two negative-product regimes combined). Within a
window, select the largest positive activation or the magnitude of the most
negative activation for the selected group. Windows are ranked by that strength;
partners can reach their peak on different token positions.

## Shared identifiers and ranking

- `source_layer`, `target_layer`: paired decoder indices, with target = source + 24.
- `neuron`: the selected index in both layers.
- `paired_rank`: true partner's rank among itself and usable random controls,
  with 1 best. The implementation uses `1 + number greater + half the number tied`,
  with absolute tolerance 1e-12 for ties. The 864 correlation rows contain no
  fractional ranks; overlap ranks can be fractional because tied overlaps occur.
- Empty numeric fields mean unavailable, not zero. A blank issue code means no
  exclusion for that window list.

## `data/capture/reports/correlations.csv`

| Column | Meaning |
|---|---|
| `source_layer`, `target_layer`, `neuron`, `segment` | Pair identifiers and neuron-index segment. |
| `component` | One of the three token-response scalars above. |
| `paired_correlation` | Pearson correlation of the source and true partner at the same token positions over all windows. |
| `control_mean` | Arithmetic mean correlation with usable random different-index target neurons. |
| `paired_rank` | Rank by correlation, as defined above. |
| `usable_controls` | Number of finite control correlations. |
| `token_observations` | Token positions used per channel (2,560,000). |

## `data/capture/reports/passage_overlap.csv`

Let S and T be the source and partner top-window sets of size `top_k`.

| Column | Meaning |
|---|---|
| `source_layer`, `target_layer`, `neuron`, `segment` | Pair identifiers and segment. |
| `response_group` | One of the six response groups above. |
| `top_k` | Number of strongest eligible windows in each list: 10, 100 or 500. |
| `shared_passages` | Count of shared windows, `size(S intersection T)`. |
| `paired_jaccard` | Shared fraction of the union, `size(S intersection T) / size(S union T)`. This is not the shared count or the fraction of either individual list. |
| `control_shared_mean` | Mean shared-window count between the source and each usable control list. |
| `control_jaccard_mean` | Mean Jaccard overlap between the source and each usable control list. |
| `paired_rank` | Rank of the true partner's Jaccard overlap among itself and usable controls. |
| `usable_controls` | Number of controls with valid lists when the source list is valid. |
| `source_issue`, `partner_issue` | Exclusion code for either true-pair list, as defined below. |

- `insufficient_signed_passages`: fewer than `top_k` windows have a finite,
  nonzero response with the requested sign/group.
- `tied_cutoff`: equal-strength windows straddle the cutoff, so a unique list of
  exactly `top_k` windows cannot be selected without breaking the tie.

Invalid true-pair lists leave paired count/Jaccard/rank blank. A valid source can
still have usable controls when the true partner list is invalid. Aggregate control
columns first average usable controls for each source, then average across usable
source pairs; the study reports the corresponding denominators.

## `data/capture/reports/strong_passage_references.csv`

| Column | Meaning |
|---|---|
| `layer`, `neuron` | Decoder layer and selected channel. |
| `sign_regime` | One of the four gate/value sign labels above. |
| `passage_index` | Zero-based index into the 10,000 measured windows. |
| `passage_id` | Text identifier, e.g. `p01707`. |
| `token_position` | Zero-based token position within that window. |
| `gate_pre_silu`, `value`, `activation` | Three scalar values at that token, with meanings defined above. |

The report retains up to 25 strongest finite window peaks per selected channel
and sign regime. Exact token IDs use row `passage_index + 1` of `tokens.npy`:
row 0 is the separate instrumentation window. Regime indices in array/JSON records
0, 1, 2 and 3 follow the four sign-regime rows above in order.

## `analysis/prefix_sensitivity.csv`

| Column | Meaning |
|---|---|
| `source_layer`, `neuron` | Pair identifiers; target is source + 24. |
| `drop_first_tokens` | Initial response positions omitted from each of the 512 retained raw windows: 0, 8 or 32. The model inputs are unchanged. |
| `paired_correlation` | Pearson correlation of the paired final-activation channels over the remaining positions. Raw control curves are unavailable for this check. |
