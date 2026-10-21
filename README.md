---
license: cc-by-4.0
language:
  - en
pretty_name: Repeated blocks in Qwen-family checkpoints
tags:
  - interpretability
  - model-analysis
  - weight-similarity
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

**Several Qwen checkpoints contain a stretch of decoder blocks whose weights are far more similar,
block for block, to the stretch a fixed offset away than any other pair of blocks at that offset:
layers 32–55 of Qwen3-32B line up with layers 8–31, layers 20–31 of
Qwen3.5/3.6-35B-A3B line up with 8–19, layers 32–47 of Qwen2.5-32B line up with 16–31, and so on
across the surveyed checkpoints.** Corresponding pairs show elevated cosine relative to
the other pairs at the same offset. Routers also have a nonzero background, and some checkpoints
have additional structure at other offsets. SOLAR-10.7B provides a disclosed depth-upscaling
comparison in the same tensor family. This release measures that pattern in 34 pinned checkpoints with nothing but HTTP byte-range
reads of their public weight files, and ships the data, the scanner and a verifier so anyone can
check it on a laptop. For survey counting, grouping related variants and derivatives gives
about **9 Qwen base/lineage groups with a block and 16 without**, plus SOLAR as a separate
calibration. The grouped positive rows are Qwen3.5-35B Base/Instruct with Qwen3.6-35B,
Qwen3.5/3.6/3.8-27B, Qwen3-30B with its 2507 variant, and Qwen2.5-32B with R1 and KAT;
the negative rows group Qwen2.5-7B with Math-7B. Every checkpoint is measured separately.

Jeffrey Shorthill · independent interpretability research · 2026-10-21 · v0.1.2

![Repeated blocks across the Qwen family](figures/family_blocks.png)

## The rows people have asked about

Within-block mean cosine between corresponding layers, with the mean over the remaining
pairs at the same offset for comparison. Both are flattened float64 cosines except the
explicitly labeled router-row values, which use aligned per-row cosines for both means. Full table: [RESULTS.md](RESULTS.md).

| Checkpoint | Layers | Corresponding blocks | Tensor | Within-block | Same-offset, off-block |
|---|---:|---|---|---:|---:|
| Qwen3-32B | 64 | 8–31 ↔ 32–55 | attention k_proj / v_proj | 0.254 (n=24) / 0.447 (n=24) | 0.000 (n=16) / 0.000 (n=16) |
| Qwen3.6-35B-A3B (3.5 Instruct and Base within 0.002) | 40 | 8–19 ↔ 20–31 | DeltaNet conv1d / k_proj / router rows | 0.896 (n=9) / 0.524 (n=3) / 0.795 (n=12) | -0.001 (n=12) / 0.000 (n=4) / 0.260 (n=16) |
| Qwen2.5-32B (and DeepSeek-R1-Distill-Qwen-32B) | 64 | 16–31 ↔ 32–47 | attention k_proj; v_proj (2.5-32B only) | 0.424 / 0.422 (n=16 each); v_proj 0.544 (n=16) | 0.000 (n=32 each); v_proj 0.000 (n=32) |
| Qwen3.6-27B (3.5/3.8 measured separately; values here are 3.6) | 64 | 8–23 ↔ 24–39 ↔ 40–55 | DeltaNet conv1d / k_proj | 0.859 (n=24) / 0.309 (n=8) | 0.001 (n=12) / 0.000 (n=4) |
| SOLAR-10.7B-v1.0 (disclosed up-scaling, calibration) | 48 | 8–23 ↔ 24–39 | attention k_proj | 0.871 (n=16) | 0.000 (n=16) |
| KAT-V1-40B (Qwen2.5-32B expanded, per its report) | 80 | 24–31 ↔ 32–39 and 40–47 ↔ 48–55 | attention k_proj | 0.984 / 0.978 (n=8 each; 16 total) | 0.000 (n=56) |

Counts refer to the displayed means. The 27B row pools the two consecutive block comparisons:
12 convolution pairs and 4 attention pairs per comparison. KAT displays its two independent
8-pair means separately; Qwen2.5-32B, its R1 derivative, and SOLAR contribute 16 pairs
per 16-layer segment comparison.
The value projection (`v_proj`) was added for Qwen3-32B and Qwen2.5-32B on 2026-10-21 at a
reviewer's request as a second attention tensor family. Measured against its own same-offset
background, it shows elevated similarity at the same corresponding positions as the key
projection (within-block 0.447 and 0.544, off-block 0.000, offset spectrum peaking only at K),
and in both checkpoints, corresponding value projections have higher within-block cosine
similarity than corresponding key projections. Keys and values are trained together, so this
corroborates rather than independently replicates the key-projection result; the query and output projections and the
MLP of those layers remain unmeasured.

Under the inherited-block interpretation, KAT also predicts offsets 16+8=24 and
16+2×8=32; the four observed 8-pair correspondences are listed in
[RESULTS.md](RESULTS.md#kat-inherited-block-arithmetic).

Across the surveyed MoEs, flattened router means at offset 1 round to 0.26–0.51.
The table and chart compare within-block similarity against the same-offset off-block
background; the raw-rank gate's margins are reported in [RESULTS.md](RESULTS.md#router-raw-rank-gate-margins). The checkpoints with no repeat
(every model at or below 14B, Qwen3.5-122B and 397B, Qwen3-Coder-480B, Qwen1.5/2 MoEs) show no
eligible neighboring-offset spike above 0.010175. Two small-hybrid attention families
have no offset with six pairs and remain explicitly unranked; their convolution families
support the model-level no-block result.

## What this does and does not establish

Elevated similarity between aligned block pairs, relative to other pairs at the same offset
and to the surveyed controls, is consistent with the paired blocks having started from the same
weights, as happens in the two disclosed depth up-scalings measured here (SOLAR, KAT). The weights
alone cannot show that this is what happened, how or when the model was built, how much training
followed any copy, or whether the paired blocks now compute the same function. Cosine is not a
clock. The release makes no claim beyond the weights.

## Check it yourself

Offline, about a minute, no network: verifies every file hash, the lock and scanner hashes, the
block coverage against each pinned `config.json`, and that `RESULTS.md` regenerates byte for byte.

```bash
git clone https://huggingface.co/datasets/ec75hash/copy-blocks-qwen
cd copy-blocks-qwen
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/verify_release.py
python scripts/make_report.py --out /tmp/figs --report /tmp/RESULTS.md
```

Online, the real test: re-read the pinned tensors from the Hub and recompute everything. The
scanner downloads only `config.json`, the safetensors index (or header) and the selected tensors'
byte ranges; it refuses any response that is not an honoured `Range` request. No model runs.
No GPU. This example selects Qwen3-32B, Qwen3.6-35B-A3B, Qwen2.5-32B and the SOLAR
calibration (about 1.8 GB):

```bash
python scripts/scan_weights.py --models qwen3-32b qwen36-35b-a3b qwen25-32b solar-10p7b \
    --out reproduction/data --provenance reproduction/metadata
python scripts/make_report.py --data reproduction/data --out reproduction/figures --report reproduction/RESULTS.md
python - <<'PY'
import csv, math
selected = {"Qwen/Qwen3-32B", "Qwen/Qwen3.6-35B-A3B",
            "Qwen/Qwen2.5-32B", "upstage/SOLAR-10.7B-v1.0"}
def read(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["model"] in selected]
    pairs = {(r["model"], r["revision"], r["tensor"], r["layer_a"], r["layer_b"]):
             float(r["cosine"]) for r in rows}
    assert len(pairs) == len(rows), "duplicate pairs"
    return pairs
a, b = read("data/pairs.csv"), read("reproduction/data/pairs.csv")
assert a.keys() == b.keys(), "missing or extra pairs"
assert all(math.isfinite(b[k]) and abs(a[k] - b[k]) <= 1e-9 for k in a)
print("Matched", len(a), "pairs within 1e-9")
PY
```

The current release covers 34 checkpoints and 2,464 tensors (11.1 GB of tensor payloads).
The original v0.1.0 capture covered 2,336 tensors (9.8 GB) and took about 40 minutes on a
home connection; its recompute from the local cache took 13 minutes. The v0.1.1 extension
adds 128 value-projection tensors (1.34 GB). Hardware used here:
Apple M5 Max, 48 GB, macOS 27.0.1, Python 3.14.4, NumPy 2.4.2 (Accelerate BLAS). The largest normalized input stack is 80 × 5,242,880 values in float64 (3.36 GB);
peak memory is higher because decoded weights, stacked copies and conversion temporaries
coexist. The complete survey was run with 48 GB of RAM.

## Independent reproduction

A second scanner, written separately in a Codex (GPT-6) session with its own tensor cache and
its own pair table, measured 33 of these checkpoints at the same revisions; its roster omitted
DeepSeek-R1-Distill-Qwen-32B. All 38,540 layer
pairs the two tables share agree to within 4 × 10⁻¹⁵, and all 1,552 shared tensor payload hashes
match. Details in [docs/INDEPENDENT_CHECK.md](docs/INDEPENDENT_CHECK.md). That comparison also
caught a real bug: on this platform a float32 Gram matrix of 32 or more multi-million-element
rows is wrong by about 0.005 (normalised diagonals come out 0.99). Both scanners now accumulate in
float64, with diagonal tolerances of 10⁻⁹ here and 10⁻¹⁰ in the independent scan. The
largest correction in the earlier Qwen3.6-27B release is SOLAR's within-block mean,
0.870722 → 0.871450; the smaller 27B changes are listed in the independent-check note.

## Limitations

- Weights only. No activations, no behaviour, no claim about function.
- Consistent with shared initialisation, not a recipe. Qwen has not published how these checkpoints were built; the
  reading "copied, then trained" rests on the SOLAR calibration and the off-block floor.
- One tensor family per claim (two, `k_proj` and `v_proj`, for Qwen3-32B and Qwen2.5-32B). Dense models are read through `k_proj`; hybrids add the DeltaNet
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
- `data/pairs.csv`: 60,982 layer pairs: model, revision, tensor, layer_a, layer_b, offset,
  cosine, row-mean cosine, whether the pair is a corresponding-block pair.
- `data/offsets.csv`, `data/summary.json`: per-offset means and the per-family summaries.
  The summary's `inside_row_mean` and `outside_row_mean` provide the matched row-mean
  comparison shown in RESULTS and the README router row (0.795 versus 0.260 for Qwen3.6-35B).
- `data/provenance.json`: every tensor's shard, dtype, shape, absolute byte range, payload and
  header SHA-256, plus scanner and lock hashes, package versions and run time.
  Capture-time hashes are retained alongside recorded reporting amendments; the added
  off-block row means were aggregated from the saved pair table without a new weight scan.
- `provenance/metadata/`: the pinned `config.json` and index files (no tensor data).
- `checksums.json`: SHA-256 of every tracked file except the manifest itself; `scripts/verify_release.py` checks it.
- `figures/`: the family chart, per-family offset spectra, and six heat maps.
- [METHODS.md](METHODS.md) · [RESULTS.md](RESULTS.md) · [docs/CHART_CORRECTIONS.md](docs/CHART_CORRECTIONS.md)
  (what changed against the September 2026 hand-assembled chart) · [NOTICE](NOTICE) · [CITATION.cff](CITATION.cff)

Related: the Qwen3.6-27B release with the J++ readout check,
https://github.com/ec75hash/qwen-depth-upscaling-jpp-check.

Code Apache-2.0 ([LICENSE](LICENSE)); data, figures and prose CC BY 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)).
The scanner derives from the 27B release; Claude assisted with its extensions, the report code
and this documentation; the independent check was a Codex session. The survey, its direction and
its interpretation are the author's.
