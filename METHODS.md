# Methods

## What is measured

For each checkpoint in `models.lock.json`, the scanner reads one or more *parameter families*
from every decoder block that carries them, flattens each tensor, and computes the cosine
similarity between every pair of blocks. Three families are used:

| Family | Tensor suffix | Where it exists | Why it is used |
|---|---|---|---|
| Attention key projection | `self_attn.k_proj.weight` | every block of a standard transformer; the full-attention blocks of a hybrid | present in every model in the survey, so it is the common yardstick |
| DeltaNet short convolution | `linear_attn.conv1d.weight` | the linear-attention blocks of Qwen3.5/3.6/3.8 and Qwen3-Next hybrids | small (tens of thousands of parameters), sharply block-local, and the family that showed the pattern first |
| MoE router | `mlp.gate.weight` | every block of a mixture-of-experts model | one row per expert, so its cosine is also computed row-aligned: a copied router keeps its expert ordering |

Two statistics are reported per block pair. The **flattened cosine** treats the whole tensor
as one vector; it is the release's primary number and matches the definition used in the
Qwen3.6-27B release. For two-dimensional tensors the **row-mean cosine** is the mean over
rows of the per-row cosine between the same row of the two blocks; for a router this is the
mean over experts of the agreement between expert *i*'s gate vector at the two depths. The
two differ when row norms vary across rows.

## Offsets, regions and periods

An *offset* is the difference between two zero-based decoder block indices, enumerated from
1 to depth−1 using the original block IDs (not positions in the sparse list of blocks that
carry a tensor). For each family and offset the scanner records the mean cosine over all
pairs at that offset. Offsets with fewer than six pairs stay in `data/offsets.csv` but are
excluded from rankings and figures; that is a reporting rule, not a significance test.

Each lock entry carries a *period* K and a *region* [first, last]: the block hypothesis for
that checkpoint, taken from the September 2026 survey of the family (the chart this release
re-measures). The scanner does not use them to select data; it uses them only to split the
pairs at offsets K and 2K into *within-block* pairs (both ends inside the region) and
*off-block* pairs, so the two means can be compared. For checkpoints without a proposed block
the region is the whole decoder.

`scripts/make_report.py` turns the measurements into a verdict per family with fixed rules:

- a proposed block is **confirmed** when the top-ranked offset for the family is K or 2K, the
  within-block mean at K is at least 0.10, and off-block pairs at K average below 0.05;
- a no-block row **holds** when no rank-eligible offset mean reaches 0.05.

The thresholds are stated in the script and in `RESULTS.md`. They are conservative for the
positive cases in this survey (within-block means run from about 0.15 to 0.98) and loose for
the negative ones (off-block and no-block means sit at the third decimal place).

## Capture

The scanner downloads only `config.json` and `model.safetensors.index.json` (or, for a
single-file checkpoint, the safetensors header) through `huggingface_hub`, checks their
SHA-256 against the lock, and then reads each selected tensor's bytes from its shard with an
HTTP `Range` request. A response is accepted only with status 206, the requested
`Content-Range`, and the exact byte length; a server that returns the whole shard is rejected
before its body is read. Each tensor is recorded with its source shard, dtype, shape,
absolute byte range, shard-header hash and payload hash in `data/provenance.json`. BF16 is
decoded exactly to float32 and the linear algebra runs in float32 NumPy.

Tensor selection is anchored to the declared decoder namespace (`model.layers.N.` or
`model.language_model.layers.N.`), so auxiliary multi-token-prediction blocks (`mtp.layers.*`)
and vision towers cannot enter the analysis. For hybrids the selected block IDs for each
family must match the configuration's attention layout, given either as `layer_types` or as
`full_attention_interval`.

No model is executed and no complete shard is downloaded. Tensor bytes are cached locally
(`--cache`) so a rerun recomputes from disk; the cache is not part of the release.

## Controls and calibration

- **SOLAR-10.7B-v1.0** has a published construction (depth up-scaling: duplicate a block
  range, then continue pretraining). Its within-block k_proj cosine (0.87) calibrates what a
  copied-then-trained block looks like in the same tensor family.
- **DeepSeek-R1-Distill-Qwen-32B** is, per its model card, fine-tuned from Qwen2.5-32B; it
  shows that a block pattern survives downstream fine-tuning.
- **KAT-V1-40B** is, per its technical report (§2, arXiv:2507.08297), Qwen2.5-32B "scaled
  to a 40B-parameter variant via a targeted layer-wise expansion strategy": a second disclosed
  up-scaling, this time of a checkpoint that already carries a block. Only KAT's own
  within-checkpoint structure is measured here; cross-model similarity is not.
- **Qwen2.5-7B, Qwen3.5-9B** and the other no-block rows are same-family checkpoints scanned
  with the same families; they show the off-block floor.

## What the measurement does and does not establish

A block-localized similarity peak at one offset, absent elsewhere and absent in same-family
controls, supports shared ancestry between the paired blocks: at some point they were the
same weights. It does not establish how the model was built, when the copying happened
relative to training, how much training followed it, or whether the blocks now compute the
same function. Cosine magnitude is not a clock. Claims of that kind are outside this release.

Rows within a checkpoint share weights and are not independent samples; there are no
confidence intervals here. The scan is a deterministic measurement of a pinned artifact,
and the same bytes give the same numbers up to BLAS rounding.

## Reproduction

Offline (`python scripts/verify_release.py`, `python scripts/make_report.py`) checks the
released files against `checksums.json` and regenerates every table and figure from
`data/`. Online (`python scripts/scan_weights.py --out reproduction/data --provenance
reproduction/metadata`) re-reads the pinned tensors from the Hub and recomputes everything;
compare its `data/` to the release numerically. The run needs Python 3.11+, NumPy, requests,
huggingface_hub and matplotlib, a few gigabytes of network transfer, and no GPU.
