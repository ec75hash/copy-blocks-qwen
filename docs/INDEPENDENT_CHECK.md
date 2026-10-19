# Independent check against a second scanner

On 2026-10-19 a Codex (GPT-6) session working in a separate workspace wrote its own byte-range
scanner, kept its own tensor cache, and measured 33 of this release's checkpoints at the same
pinned revisions (its roster lacks DeepSeek-R1-Distill-Qwen-32B). Its per-pair table
(`current/[CURRENT] FAMILY_DATA.json` in that workspace, SHA-256 prefix `9980ce0b1acbe7b0` at the
time of comparison) was diffed against this release's `data/pairs.csv` on matching
(repository, tensor, layer_a, layer_b) keys.

| Quantity | Value |
|---|---|
| Checkpoints shared | 33 of 34 (revisions identical on all 33) |
| Layer pairs shared | 38,540 |
| Largest absolute difference in flattened cosine | 3.72 × 10⁻¹⁵ |
| Pairs differing by more than 10⁻⁵ | 0 |
| Shared tensor payload hashes | 1,552, all equal |
| Pairs only in this release | 18,410 (k_proj / conv1d families the other scanner did not select for MoEs, plus R1-Distill) |

The two scanners share an ancestor (the Qwen3.6-27B release scanner) but were extended
separately, read the Hub through separate caches, and compute the row-aligned statistic only on
this side; the agreement is on the full-tensor cosine.

## What the comparison caught

The other session's reviewer found that its float32 Gram matrix gave KAT-V1-40B L24↔L32 as
0.978407562 where a float64 dot product gives 0.983010680. A two-vector float32 check on this
side agreed with float64 to 3 × 10⁻⁸ and so missed it; the error appears only when the Gram
matrix has 32 or more rows of several million elements (NumPy 2.4.2 on macOS arm64 with
Accelerate BLAS: normalised diagonals come out 0.989–0.996 instead of 1). Reproduced here on the
full 80-row KAT stack: float32 0.978407562, float64 0.983010680. Both scanners now accumulate in
float64 and reject any Gram whose diagonal deviates from 1 by more than 10⁻⁹ (this release) or
10⁻¹⁰ (theirs). The earlier Qwen3.6-27B release computed in float32; its 16-row k_proj and
40,960-element conv1d matrices were inside the safe regime (0.858496 → 0.858501, 0.309258 →
0.309261), while its 48-row SOLAR matrix was not (0.870722 → 0.871450).

The same session supplied the primary source for KAT-V1-40B's parentage (its technical report,
arXiv:2507.08297 §2), and the representation of KAT's two independent block pairs used in
`models.lock.json` (`segments` + `pair_groups`) follows its data model.

Agreement between two measurements of the same bytes is evidence that the measurement is
implemented correctly; it is not additional evidence about the checkpoints.
