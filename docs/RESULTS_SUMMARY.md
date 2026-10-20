# Qwen copy-block survey — finalized results

Jeffrey Shorthill · 2026-10-19 · release v0.1.0 (`/Volumes/ExternalSSD/copy-blocks-qwen-release`)

## The finding in one paragraph

Fifteen Qwen-family checkpoints contain a stretch of decoder blocks whose weights repeat, block
for block, at a fixed offset. Inside that stretch, layer *l* and layer *l + K* have the same
attention, DeltaNet, and router weights to a cosine of 0.25–0.98 depending on the model and
tensor; outside it, and in every other layer pair, the cosine is zero. Nineteen other Qwen
checkpoints show nothing of the kind. SOLAR-10.7B, which Upstage built by copying a block and
training further, shows the same signature at 0.87. The measurement reads raw weights only; it
says the paired blocks were once the same weights, and nothing about how or when, or what they
do now.

## Checkpoints with a repeated block

Within = mean cosine over corresponding layer pairs (flattened weights, float64). Off = mean over
the other pairs at the same offset. Router rows = the same measure per expert row. All indices
are zero-based decoder blocks. 3.5/3.6-35B-A3B Instruct and Base are identical to these digits.

| Checkpoint | Layers | Corresponding blocks | K | k_proj within / off | DeltaNet conv1d within / off | Router rows within / off |
|---|---:|---|---:|---|---|---|
| **Qwen3-32B** (Goodfire) | 64 | 8–31 ↔ 32–55 | 24 | 0.254 / 0.000 | — | — |
| **Qwen3.6-35B-A3B** (Goodfire; E114 paper) | 40 | 8–19 ↔ 20–31 | 12 | 0.524 / 0.000 | 0.896 / 0.000 | 0.795 / 0.25 |
| **Qwen2.5-32B** (Dadfar) | 64 | 16–31 ↔ 32–47 | 16 | 0.424 / 0.000 | — | — |
| DeepSeek-R1-Distill-Qwen-32B (fine-tune of the above) | 64 | 16–31 ↔ 32–47 | 16 | 0.422 / 0.000 | — | — |
| KAT-V1-40B (Qwen2.5-32B expanded; report §2) | 80 | 24–31 ↔ 32–39, and 40–47 ↔ 48–55 | 8 | 0.984, 0.978 / 0.000 | — | — |
| Qwen3-30B-A3B (and Instruct-2507) | 48 | 6–17 ↔ 18–29 ↔ 30–41 | 12 | 0.532 / 0.000 | — | 0.882 / 0.33 |
| Qwen3-Next-80B-A3B | 48 | 4–15 ↔ 16–27 ↔ 28–39 | 12 | 0.632 / 0.000 | 0.940 / 0.000 | 0.921 / 0.28 |
| Qwen3-235B-A22B | 94 | 13–46 ↔ 47–80 (router also spikes at offsets 20 and 54) | 34 | 0.528 / 0.000 | — | 0.883 / 0.23 |
| **Qwen3.6-27B** (J++ Lens; also 3.5-27B, 3.8-27B) | 64 | 8–23 ↔ 24–39 ↔ 40–55 | 16 | 0.309 / 0.000 | 0.859 / 0.001 | — |
| Qwen3.8-Flash-Next | 48 | 8–23 ↔ 24–39 | 16 | 0.830 / 0.000 | 0.969 / 0.000 | 0.900 / 0.07 |
| Qwen3.8-2.4T-A95B | 92 | 12–43 ↔ 44–75 | 32 | 0.532 / 0.000 | 0.950 / 0.000 | 0.839 / 0.10 |
| SOLAR-10.7B-v1.0 (disclosed up-scaling; calibration) | 48 | 8–23 ↔ 24–39 | 16 | 0.871 / 0.000 | — | — |

For the tripled blocks the two consecutive pairs differ a little (Qwen3.5-27B k_proj 0.250 then
0.369; conv1d 0.840 then 0.877; Qwen3-30B k_proj 0.512 then 0.552).

## Checkpoints with no repeated block

Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B, Qwen2.5-7B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen3-0.6B / 1.7B /
4B / 8B / 14B, Qwen3-Coder-480B-A35B, Qwen3.5-0.8B / 2B / 4B / 9B, Qwen3.5-122B-A10B,
Qwen3.5-397B-A17B. In every one, no offset's mean stands more than 0.010 above its neighbouring
offsets in any tensor. (MoE routers everywhere share a smooth 0.3–0.5 similarity between
neighbouring layers; that is drift, and the test is the spike above it, not the raw value.)

## Why the numbers can be trusted

- **Pinned.** Every checkpoint is read at a full commit SHA; config and index files are hashed;
  every tensor's shard, byte range and payload SHA-256 are recorded (`data/provenance.json`).
- **No model runs.** The scanner pulls only the needed tensors by HTTP byte range (9.8 GB over
  2,336 tensors for all 34 checkpoints); anyone can rerun it on a laptop, no GPU.
- **Independently reproduced.** A second scanner, written separately by Codex with its own cache,
  agrees on all 38,540 shared layer pairs to within 4 × 10⁻¹⁵ and on all 1,552 shared payload hashes.
- **One real bug found and fixed on both sides.** float32 Gram matrices of 32+ rows are wrong by
  ~0.005 on Apple Accelerate; both scanners now use float64 with a diagonal guard. Only one
  earlier published number moves: SOLAR in the Qwen3.6-27B release, 0.8707 → 0.8715.
- **Verifier.** `scripts/verify_release.py` checks 95 file hashes, the lock and scanner hashes,
  block coverage against each pinned config, roster completeness, and that `RESULTS.md`
  regenerates byte for byte. It passes.

## What changed against the September chart

Two small hybrids had their layer counts wrong (Qwen3.5-0.8B/2B are 24, 4B/9B are 32); the
footer's family-wide rule was contradicted by two of its own rows and is gone; KAT's parentage
now cites its technical report (arXiv:2507.08297 §2) instead of a model-card inference;
"source / copy" labels became "block A / B / C" because the weights give no direction; and the
Qwen3-Next annotation was a statistic from a different tensor and is superseded. Every other
September number is reproduced.

## What this does not claim

No construction recipe, no training chronology, no statement that the paired blocks compute the
same function today. Cosine is not a clock. Routers are judged against their same-offset
baseline. No confidence intervals: one checkpoint is one measurement.

## Files

- Chart: `figures/family_blocks.png` · full table: `RESULTS.md` · method: `METHODS.md`
- Data: `data/pairs.csv` (56,950 pairs), `data/offsets.csv`, `data/summary.json`, `data/provenance.json`
- Corrections: `docs/CHART_CORRECTIONS.md` · independent check: `docs/INDEPENDENT_CHECK.md`
- Lock: `models.lock.json` (34 checkpoints) · verifier: `scripts/verify_release.py`
