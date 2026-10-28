# Fresh exact-input replay

The original collection used macOS 27.0.1, Apple M5 Max, llama-cpp-python 0.3.36
with Metal, gguf 0.19.0 and NumPy 2.5.4. Exact original runtime library hashes are
retained in `../data/capture/instrumentation.json`; private llama.cpp APIs are used.
This is a platform-specific inference path, separate from the CPU-only offline reductions.

Install `../requirements-capture.txt` into a separate environment, building
llama-cpp-python with Metal as appropriate for the local toolchain. Obtain the official
`Qwen/Qwen3-32B-GGUF` Q8_0 file at revision
`938a7432affaec9157f883a87164e2646ae17555` separately. The replay verifies its full
SHA-256 before loading it. Model weights are not bundled with the measurements.

From the release root:

```bash
python studies/qwen3_response/code/replay_saved_inputs.py --model /path/to/Qwen3-32B-Q8_0.gguf --out reproduction/new_capture
```

Use `--limit 16` and a different new output directory for a short instrumentation
replay. This wrapper reads the exact frozen token IDs and channel layout; it does not
reuse the old title parser or depend on locally installed `hf` for corpus preparation.
It refuses existing output directories, supports neither resume nor detached lifecycle
control, and does not overwrite released evidence. A full run requires the 34.8 GB
model, enough RAM for the model/runtime, approximately 3.5 GB (3.3 GiB) for output and suitable
free-disk reserve. The original run used a 48 GB Mac and took about 1 hour 54 minutes.

The original complete as-run runner and its historical preparation code are in
`../data/capture/code/`. They are retained for provenance, with the known parser
defect documented separately. Do not treat those local as-run paths as a portable launcher.

This wrapper produces a fresh capture, not a complete fresh-report pipeline.
The packaged reducer reads the released `data/capture/` directory; it does not
accept this new output directory as an input. Generating the full set of reports
from a fresh capture requires additional integration.

The new wrapper was checked for syntax, CLI operation and correspondence with the
original capture loop. No second full inference run was performed for this update;
cross-platform or bitwise inference reproducibility is not asserted.
