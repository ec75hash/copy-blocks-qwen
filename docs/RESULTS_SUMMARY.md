> Historical within-checkpoint survey summary from v0.1.2. Statements below about unmeasured MLPs or weight-only scope describe that survey. For the later Qwen2.5/Qwen3 14B–32B weight comparisons and Q8 responses, see [the v0.2.0 update](UPDATE_20261027.md) and [documentation history](DOCUMENTATION_HISTORY.md).

# Qwen copy-block survey — results summary

Jeffrey Shorthill · release v0.1.2 · documentation checked 2026-10-21

The pinned survey measures repeated block patterns in 16 Qwen or Qwen-derived checkpoints,
plus SOLAR as a disclosed depth-upscaling control; 17 other checkpoints show no repeated
block under the stated reporting rules. These are within-checkpoint weight comparisons.
Same-layer cross-checkpoint and activation experiments are separate studies.

## Checkpoints with a repeated block

Within/off = corresponding-pair mean and remaining-pair mean at offset K. Attention and
convolution values are flattened float64 cosines; router values are aligned row-mean
cosines for both columns. Every checkpoint has its own row. KAT lists its two independent
within-group means. Indices are zero-based decoder blocks.

| Checkpoint | Layers | Corresponding blocks | K | k_proj within / off | conv1d within / off | Router rows within / off |
|---|---:|---|---:|---|---|---|
| Qwen3-32B | 64 | 8–31 ↔ 32–55 | 24 | 0.254 / -0.000 | — | — |
| Qwen3.6-35B-A3B | 40 | 8–19 ↔ 20–31 | 12 | 0.524 / -0.000 | 0.896 / -0.001 | 0.795 / 0.260 |
| Qwen3.5-35B-A3B | 40 | 8–19 ↔ 20–31 | 12 | 0.522 / -0.001 | 0.896 / -0.001 | 0.794 / 0.264 |
| Qwen2.5-32B | 64 | 16–31 ↔ 32–47 | 16 | 0.424 / 0.000 | — | — |
| KAT-V1-40B (grown from Qwen2.5-32B) | 80 | 24–31 ↔ 32–39; 40–47 ↔ 48–55 | 8 | 0.984, 0.978 / -0.000 | — | — |
| SOLAR-10.7B-v1.0 | 48 | 8–23 ↔ 24–39 | 16 | 0.871 / 0.000 | — | — |
| Qwen3-30B-A3B | 48 | 6–17 ↔ 18–29 ↔ 30–41 | 12 | 0.532 / -0.000 | — | 0.882 / 0.333 |
| Qwen3-30B-A3B-Instruct-2507 | 48 | 6–17 ↔ 18–29 ↔ 30–41 | 12 | 0.530 / -0.000 | — | 0.883 / 0.329 |
| Qwen3-Next-80B-A3B | 48 | 4–15 ↔ 16–27 ↔ 28–39 | 12 | 0.632 / -0.000 | 0.940 / -0.005 | 0.921 / 0.284 |
| Qwen3-235B-A22B | 94 | 13–46 ↔ 47–80 | 34 | 0.528 / -0.000 | — | 0.883 / 0.236 |
| Qwen3.5-27B | 64 | 8–23 ↔ 24–39 ↔ 40–55 | 16 | 0.310 / -0.000 | 0.860 / 0.001 | — |
| Qwen3.6-27B | 64 | 8–23 ↔ 24–39 ↔ 40–55 | 16 | 0.309 / -0.000 | 0.859 / 0.001 | — |
| Qwen3.8-27B | 64 | 8–23 ↔ 24–39 ↔ 40–55 | 16 | 0.303 / -0.000 | 0.858 / 0.001 | — |
| Qwen3.8-Flash-Next | 48 | 8–23 ↔ 24–39 | 16 | 0.830 / -0.001 | 0.969 / -0.001 | 0.900 / 0.078 |
| Qwen3.8-2.4T-A95B | 92 | 12–43 ↔ 44–75 | 32 | 0.532 / -0.000 | 0.950 / 0.000 | 0.839 / 0.098 |
| DeepSeek-R1-Distill-Qwen-32B | 64 | 16–31 ↔ 32–47 | 16 | 0.422 / 0.000 | — | — |
| Qwen3.5-35B-A3B-Base | 40 | 8–19 ↔ 20–31 | 12 | 0.522 / -0.001 | 0.897 / -0.001 | 0.794 / 0.264 |

For tripled blocks, the table pools consecutive corresponding pairs; the full table
and pair CSV retain their individual measurements. For example, Qwen3.5-27B has consecutive
k_proj means 0.250 / 0.369 and convolution means 0.842 / 0.877.

## Checkpoints without a confirmed repeated block

Qwen2.5-7B, Qwen3.5-9B, Qwen3-Coder-480B-A35B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B, Qwen2.5-14B, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B, Qwen2.5-Math-7B.

The largest eligible neighboring-offset spike is 0.010175 (below the 0.05 reporting
threshold). Two small-hybrid attention families lack six same-offset pairs and are
explicitly unranked; their convolution families support the model-level result.
Family totals: 37 confirmed, 26 no-block and 2 with no eligible offsets.

## Validation and scope

- 34 pinned checkpoints; 2,464 selected tensors; 60,982 layer pairs.
- All 38,540 shared full-tensor pairs agree with a separately extended scanner within
  4 × 10⁻¹⁵; all 1,552 shared tensor payload hashes match.
- Both scanners use float64 normalization/accumulation and a Gram-diagonal guard.
  The largest earlier release correction is SOLAR: 0.870722 → 0.871450.
- The verifier checked 96 non-manifest files in v0.1.2; the current total is the
  number of entries in `checksums.json`. It also checks roster/config coverage,
  scanner/lock hashes and exact regeneration of RESULTS.md.
- Routers have nonzero backgrounds, and additional offsets are present in some
  checkpoints. The evidence is the localized aligned-versus-baseline contrast.
- These measurements are consistent with paired blocks having started from the same
  weights, as in the disclosed SOLAR and KAT up-scalings. The weights alone cannot
  establish that history. No activation or behavioral measurement is part of this package.

## Files

[Chart](../figures/family_blocks.png) · [full table](../RESULTS.md) · [methods](../METHODS.md)
· [corrections](CHART_CORRECTIONS.md) · [independent check](INDEPENDENT_CHECK.md)

Addendum 2026-10-21: the value projection (`v_proj`) was scanned for Qwen3-32B and Qwen2.5-32B as a second attention family; within-block 0.447 and 0.544 against 0.000 off-block at the same offsets, spectrum peaking only at K, higher than the key projection in both. Trained together with `k_proj`, so corroborating rather than independent; the rest of each layer is unmeasured.
