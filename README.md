---
license: cc-by-4.0
language:
  - en
pretty_name: Repeated blocks in Qwen-family checkpoints
tags:
  - interpretability
  - model-analysis
  - depth-upscaling
  - qwen
configs:
  - config_name: layer_pairs
    data_files:
      - split: analysis
        path: data/pairs.csv
  - config_name: offset_spectrum
    data_files:
      - split: analysis
        path: data/offsets.csv
---

# Repeated blocks in Qwen-family checkpoints

**Several Qwen checkpoints contain a stretch of decoder blocks whose weights repeat, block for
block, at a fixed offset: layers 32–55 of Qwen3-32B line up with layers 8–31, layers 20–31 of
Qwen3.5/3.6-35B-A3B line up with 8–19, layers 32–47 of Qwen2.5-32B line up with 16–31, and so on
across the family.** The similarity is confined to those blocks, sits at zero for every other
layer pair, and looks like what a disclosed depth up-scaling (SOLAR-10.7B) looks like in the same
tensor. This release measures that pattern in 34 pinned checkpoints with nothing but HTTP byte-range
reads of their public weight files, and ships the data, the scanner and a verifier so anyone can
check it on a laptop.

Jeffrey Shorthill · independent interpretability research · 2026-10-19 · v0.1.0

![Repeated blocks across the Qwen family](figures/family_blocks.png)

## The rows people have asked about

Within-block mean cosine between corresponding layers (flattened weights, float64), with the
mean over the remaining pairs at the same offset for comparison. Full table: [RESULTS.md](RESULTS.md).

| Checkpoint | Layers | Corresponding blocks | Tensor | Within-block | Same-offset, off-block |
|---|---:|---|---|---:|---:|
| Qwen3-32B | 64 | 8–31 ↔ 32–55 | attention k_proj | 0.254 | 0.000 |
| Qwen3.6-35B-A3B (3.5 Instruct and Base within 0.002) | 40 | 8–19 ↔ 20–31 | DeltaNet conv1d / k_proj / router rows | 0.896 / 0.524 / 0.795 | 0.00 / 0.00 / 0.25 |
| Qwen2.5-32B (and DeepSeek-R1-Distill-Qwen-32B) | 64 | 16–31 ↔ 32–47 | attention k_proj | 0.424 / 0.422 | 0.000 |
| Qwen3.6-27B (and 3.5-27B, 3.8-27B) | 64 | 8–23 ↔ 24–39 ↔ 40–55 | DeltaNet conv1d / k_proj | 0.859 / 0.309 | 0.001 / 0.000 |
| SOLAR-10.7B-v1.0 (disclosed up-scaling, calibration) | 48 | 8–23 ↔ 24–39 | attention k_proj | 0.871 | 0.000 |
| KAT-V1-40B (Qwen2.5-32B expanded, per its report) | 80 | 24–31 ↔ 32–39 and 40–47 ↔ 48–55 | attention k_proj | 0.984 / 0.978 | 0.000 |

Routers in every Qwen MoE share a smooth 0.3–0.5 similarity between neighbouring layers whether
or not a block repeats; that is drift, and the table and chart compare each router's within-block
value against the same-offset off-block value for that reason. The checkpoints with no repeat
(every model at or below 14B, Qwen3.5-122B and 397B, Qwen3-Coder-480B, Qwen1.5/2 MoEs) show no
offset standing above its neighbours by more than 0.01 in any tensor.

## What this does and does not establish

A block-localised similarity peak at one offset, absent everywhere else and absent in same-family
checkpoints of the same tensor, supports shared ancestry between the paired blocks: at some point
they were the same weights, as in a depth up-scaling. It does not establish how or when the model
was built, how much training followed the copy, or whether the paired blocks now compute the same
function. Cosine is not a clock. The release makes no claim beyond the weights.

## Check it yourself

Offline, about a minute, no network: verifies every file hash, the lock and scanner hashes, the
block coverage against each pinned `config.json`, and that `RESULTS.md` regenerates byte for byte.

```bash
git clone https://github.com/ec75hash/copy-blocks-qwen && cd copy-blocks-qwen
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/verify_release.py
python scripts/make_report.py --out /tmp/figs --report /tmp/RESULTS.md
```

Online, the real test: re-read the pinned tensors from the Hub and recompute everything. The
scanner downloads only `config.json`, the safetensors index (or header) and the selected tensors'
byte ranges; it refuses any response that is not an honoured `Range` request. No model runs.
No GPU. The three checkpoints in the emails plus the calibration are about 1.5 GB:

```bash
python scripts/scan_weights.py --models qwen3-32b qwen36-35b-a3b qwen25-32b solar-10p7b \
    --out reproduction/data --provenance reproduction/metadata
python scripts/make_report.py --data reproduction/data --out reproduction/figures --report reproduction/RESULTS.md
diff <(cut -d, -f1-8 data/pairs.csv | grep -E 'Qwen3-32B|Qwen3.6-35B|Qwen2.5-32B|SOLAR') \
     <(cut -d, -f1-8 reproduction/data/pairs.csv)
```

All 34 checkpoints are 9.8 GB of range reads (2,336 tensors) and took about 40 minutes on a
home connection; the recompute from the local cache takes 13 minutes. Hardware used here:
Apple M5 Max, 48 GB, macOS 27.0.1, Python 3.14.4, NumPy 2.4.2 (Accelerate BLAS). Any machine
with a few GB of RAM will do; the largest matrix is 80 × 5.2 M values in float64 (3.4 GB).

## Independent reproduction

A second scanner, written separately in a Codex (GPT-6) session with its own tensor cache and
its own pair table, measured 33 of these checkpoints at the same revisions. All 38,540 layer
pairs the two tables share agree to within 4 × 10⁻¹⁵, and all 1,552 shared tensor payload hashes
match. Details in [docs/INDEPENDENT_CHECK.md](docs/INDEPENDENT_CHECK.md). That comparison also
caught a real bug: on this platform a float32 Gram matrix of 32 or more multi-million-element
rows is wrong by about 0.005 (normalised diagonals come out 0.99). Both scanners now accumulate in
float64 and refuse to proceed if a diagonal deviates from 1 by more than 10⁻⁹. The only published
number that moves is SOLAR's within-block mean in the earlier Qwen3.6-27B release, 0.8707 → 0.8715.

## Limitations

- Weights only. No activations, no behaviour, no claim about function.
- Shared ancestry, not a recipe. Qwen has not published how these checkpoints were built; the
  reading "copied, then trained" rests on the SOLAR calibration and the off-block floor.
- One tensor family per claim. Dense models are read through `k_proj`; hybrids add the DeltaNet
  `conv1d`; MoEs add the router. Other tensors may tell a more or less similar story.
- Routers have a non-zero baseline between any two layers, so their numbers are only meaningful
  against the same-offset off-block value printed beside them.
- No confidence intervals: rows within a checkpoint share weights and are not independent samples.
  The measurement is deterministic; the same bytes give the same numbers.
- Pinned revisions only (`models.lock.json`). `Qwen3-Next-80B-A3B` and `Qwen3-Coder-480B-A35B`
  are read from their Instruct repositories because no base checkpoint is published.
- KAT-V1-40B's Hub index names the wrong shard for 76 of its 80 `k_proj` tensors; the scanner
  resolves tensors against the shard headers and records the mismatch in `data/provenance.json`.

## Files and provenance

- `models.lock.json`: 34 checkpoints, full commit SHAs, SHA-256 of each `config.json` and index,
  tensor families, and the block hypothesis (period and region, or explicit segments) per row.
- `data/pairs.csv`: 56,950 layer pairs: model, revision, tensor, layer_a, layer_b, offset,
  cosine, row-mean cosine, whether the pair is a corresponding-block pair.
- `data/offsets.csv`, `data/summary.json`: per-offset means and the per-family summaries.
- `data/provenance.json`: every tensor's shard, dtype, shape, absolute byte range, payload and
  header SHA-256, plus scanner and lock hashes, package versions and run time.
- `provenance/metadata/`: the pinned `config.json` and index files (no tensor data).
- `checksums.json`: SHA-256 of every tracked file; `scripts/verify_release.py` checks it.
- `figures/`: the family chart, per-family offset spectra, and six heat maps.
- [METHODS.md](METHODS.md) · [RESULTS.md](RESULTS.md) · [docs/CHART_CORRECTIONS.md](docs/CHART_CORRECTIONS.md)
  (what changed against the September 2026 hand-assembled chart) · [NOTICE](NOTICE) · [CITATION.cff](CITATION.cff)

Related: the Qwen3.6-27B release with the J++ readout check,
https://github.com/ec75hash/qwen-depth-upscaling-jpp-check.

Code Apache-2.0 ([LICENSE](LICENSE)); data, figures and prose CC BY 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)).
The scanner derives from the 27B release; Claude assisted with its extensions, the report code
and this documentation; the independent check was a Codex session. The survey, its direction and
its interpretation are the author's.
