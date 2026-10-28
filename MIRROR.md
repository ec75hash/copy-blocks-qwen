# Lightweight source mirror

The canonical release is [HF v0.3.0](https://huggingface.co/datasets/ec75hash/copy-blocks-qwen/tree/7c0de5a888abb984eed09e9f43bca0a0d3860144), commit `7c0de5a888abb984eed09e9f43bca0a0d3860144`.
This repository includes the code, documentation, original survey tables, the
small weight studies, response CSV summaries and small capture metadata/text.
The response `.npy`/`.npz` arrays and all retained raw-window files are omitted,
along with the Hub-specific Git LFS configuration. No model weights are included.

For full checksum verification and response re-reduction, download the complete
Hub snapshot as described in the main README. The copied `checksums.json` is the
canonical Hub manifest; it deliberately includes files absent from this mirror.
`MIRROR_MANIFEST.json` lists included/omitted files and hashes, with link-only
Markdown adaptations recorded separately. Weight-study reductions run directly
from this mirror without the response arrays.

Publication history and numerical results belong to the linked canonical release.
The Base follow-up, when included in the canonical version, remains a separate study under studies/qwen3_base_followup/.
