# Reproduce the v0.3.0 evidence

Use Python 3.11+ and the root `requirements.txt`. The full release contains about
3.6 GB of evidence and documentation, including roughly 3.5 GB (3.3 GiB) of
response-study files. Model weights are not included. Read tables and plots online
without downloading the large arrays.

## Weight studies only (including the Base follow-up)

The weight studies' offline numerical reductions need under 70 MB of study
files, plus installed Python dependencies. A selective download omits the response
archive. This command uses `hf` from `huggingface_hub` 1.x and the `v0.3.0` tag;
if the tag is unavailable, use a verified release commit rather than an unpinned
`main` revision. The version tag is created when the release is published:

```bash
hf download ec75hash/copy-blocks-qwen --repo-type dataset --revision v0.3.0 --include "studies/qwen25_weights/**" --include "studies/qwen3_weights/**" --include "studies/qwen3_base_followup/**" --include "requirements.txt" --local-dir copy-blocks-qwen
cd copy-blocks-qwen
python studies/qwen25_weights/code/verify_and_plot.py --out reproduction/qwen25_weights_readout
python studies/qwen3_weights/code/verify_and_plot.py --out reproduction/qwen3_weights_readout
python studies/qwen3_base_followup/code/verify_and_report.py --out reproduction/qwen3_base_followup
```

These checks verify retained matrices and tables. Fresh tensor-byte checks are
separate: the Qwen2.5 study offers a roughly 4 MB small-family route or a 1.18 GB
six-family route. Its [README](../studies/qwen25_weights/README.md#reproduce) gives
the commands. The existing-cache/fresh-scanner routes also need the root scanner,
lock and metadata files; the selective command above is for offline reductions.

## Complete package

The root checksum verifier checks every released file, so it requires the full
snapshot. Re-deriving response results also needs the retained capture arrays.
From the complete repository root:

```bash
python -m unittest discover -s tests
python scripts/verify_release.py
python studies/qwen25_weights/code/verify_and_plot.py --out reproduction/qwen25_weights_readout
python studies/qwen3_weights/code/verify_and_plot.py --out reproduction/qwen3_weights_readout
python studies/qwen3_base_followup/code/verify_and_report.py --out reproduction/qwen3_base_followup
python studies/qwen3_response/code/reduce_and_plot.py --out reproduction/qwen3_response
python studies/qwen3_response/code/verify_capture.py
python scripts/verify_update.py --reproduced reproduction
```

The Qwen2.5 reducer checks all 18,432 cross-checkpoint values, 24 mapped
means, 96 reference-residual rows and 672 provenance entries. Its study README
also documents checked-cache and optional fresh byte-range reproduction.

The original survey verifier checks all released hashes and regenerates the original
survey result table. The weight reducer checks all 6,048 MLP table rows, 189 offsets,
saved aligned-neuron vectors, cross-checkpoint K/V matrices, gate comparisons and head counts.
The response reducer uses saved centered covariance and per-window peaks,
recomputing all 864 correlation summaries and 5,184 overlap rows; it also reduces all 512
retained raw windows for the prefix-position check. The capture verifier checks all
saved peak regimes and positions and reduces 32 raw windows.
The final update verifier compares the regenerated statistics to the released summaries
and checks that all 52 selected annotated events match saved peak values and token IDs.
It also checks documented model pins and package-source preservation.

All these checks run on CPU without model execution, network access or credentials.
Fresh byte-range tensor scans require network/disk space; fresh response inference
requires the separate pinned Q8 model and the documented macOS/Metal runtime.
Portable paths do not establish fresh cross-platform inference reproducibility.

`data/` and `studies/**/data/` are immutable evidence inputs. Generated checks and
figures are written into `reproduction/`. The completed as-run records under each
study are historical; later corrections and metadata path redactions are recorded
separately in [amendments](AMENDMENTS.json).

The Base-study reducer independently checks saved cross matrices, mapped-pair tables, all 91,392 sampled row values and both Base 30B families, and regenerates its RESULTS.md. An optional checker remeasures the same raw cached payloads; those model-weight caches are not bundled. See the [Base study](../studies/qwen3_base_followup/README.md#reproduce) for its exact requirements.
