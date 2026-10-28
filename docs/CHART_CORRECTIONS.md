# Corrections to the September 2026 family chart

The author’s September 2026 chart was an unpublished working chart, not a required
release artifact. The comparison below transcribes its annotations. That chart's rows were
typed in from several exploratory scans; this one is generated from `data/` produced by the
pinned scanner, with each checkpoint's depth taken from its pinned `config.json`. Differences:

1. **Layer counts for the small hybrids.** The September chart listed Qwen3.5-0.8B / 2B at
   18 layers and Qwen3.5-4B / 9B at 24. Those are the counts of DeltaNet (linear-attention)
   blocks; the decoders have **24** and **32** blocks respectively (18 + 6 and 24 + 8 full
   attention). The remaining 27 rows' depths agree with the pinned configurations.
2. **Footer generalization removed.** The September footer read "Every Qwen3-generation MoE
   … carr[ies] a block; every checkpoint at or below 14B and the three largest do not." Two of
   the chart's own rows contradict it: Qwen3-Coder-480B-A35B is a Qwen3-generation MoE with
   no block, and Qwen3.8-2.4T-A95B, the largest checkpoint, has one. This release states no
   family-level rule; each row carries its own measurement and verdict.
3. **Every number is a pinned measurement.** The September cosines came from earlier captures that were not hash-pinned. The values
   here come from `scripts/scan_weights.py` at the revisions in `models.lock.json`, with byte
   ranges and payload hashes in `data/provenance.json`. Where the two disagree, `RESULTS.md`
   is authoritative; see the comparison table below.
4. **Repository identities.** `Qwen/Qwen3-Next-80B-A3B` has no base checkpoint on the Hub;
   the Instruct release is scanned. `Qwen/Qwen3-Coder-480B-A35B-Instruct` likewise.
   `Qwen/Qwen3-30B-A3B-Instruct-2507` is scanned alongside the April `Qwen3-30B-A3B` because
   the September chart merged them into one row.
5. **KAT-V1-40B.** The September row labeled it "grown from Qwen2.5-32B". Its Hub model card
   does not say so, but its technical report does: "we scaled the Qwen2.5-32B model to a
   40B-parameter variant via a targeted layer-wise expansion strategy" (KAT-V1 report, §2,
   arXiv:2507.08297). The label stands, with that citation; this release still measures only
   KAT's own within-checkpoint block structure. KAT's `model.safetensors.index.json` also names
   the wrong shard for 76 of its 80 `k_proj` tensors; the scanner resolves tensors against the
   shard headers and records the mismatch in `data/provenance.json`.
6. **Two statistics for routers.** The September chart quoted "router rows" (a per-expert
   row-aligned mean). This release reports both the flattened router cosine and the
   row-aligned mean, labeled separately.

## Comparison with the September values

September numbers are the chart's hand-typed annotations; this release's numbers are means over
the corresponding-block pairs in `data/pairs.csv` (A↔B; B↔C where a third block exists).

| Checkpoint | September annotation | This release | Status |
|---|---|---|---|
| Qwen2.5-32B | k_proj copies 0.31–0.52 | k_proj 0.424 (per-pair range 0.31–0.52) | agrees |
| KAT-V1-40B | two K=8 copies at 0.98 | k_proj 0.984 and 0.978 | agrees |
| Qwen3-30B-A3B | copies 0.51 / 0.55; router rows 0.87 | k_proj 0.512 / 0.552; router rows 0.876 / 0.887 | block pattern confirmed; router statistic clarified |
| Qwen3-32B | copies 0.25 | k_proj 0.254 | agrees |
| Qwen3-Next-80B-A3B | aligned cos 0.71 / 0.56 | conv1d 0.934 / 0.947; k_proj 0.594 / 0.669; router rows 0.901 / 0.941 | superseded: the September statistic is not one of these families and was not reproducible; block positions agree |
| Qwen3-235B-A22B | K=34, nested sub-blocks at offsets 20 and 54 | router: offset 34 0.600; spikes also at 20 (0.576 vs 0.37 neighbours) and 54 (0.483 vs 0.29) | agrees; the nested offsets are real but not drawn |
| Qwen3.5/3.6-27B | copies 0.25 / 0.37 | 3.5: k_proj 0.250 / 0.369, conv1d 0.842 / 0.877; 3.6: k_proj 0.250 / 0.369, conv1d 0.840 / 0.877 | agrees at the original annotation precision |
| Qwen3.5/3.6-35B-A3B | copies 0.52; router rows 0.79; 3.5 = 3.6 to 0.004 | k_proj 0.522 / 0.524; router rows 0.794 / 0.795; Base 0.794 | agrees |
| Qwen3.8-Flash-Next | K=16 | conv1d 0.97; k_proj 0.83; router rows 0.90 | agrees, now with numbers |
| Qwen3.8-2.4T-A95B | copies 0.53; router rows 0.84 | k_proj 0.53; router rows 0.839 | agrees |
| all "no block" rows | no block | largest eligible neighboring-offset spike 0.010175; two small-hybrid attention families have no eligible offsets | agrees |

The pinned scan confirms the reported block locations. The table gives the current
statistics explicitly; the hand-typed router annotation and the superseded Qwen3-Next
statistic should not be treated as exact reproductions.
