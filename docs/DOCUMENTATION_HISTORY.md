# Reading past and current documentation

The original survey and later studies answer different questions. A dated
statement that a tensor was unmeasured describes the work available then; it does
not replace a later, explicitly linked measurement.

| Material | Scope | Current reading |
|---|---|---|
| v0.1.2 survey tables and original scanner | Within-checkpoint similarities across 34 models | Numerical evidence is retained unchanged; root METHODS and RESULTS describe this survey. |
| `docs/RESULTS_SUMMARY.md` | Historical survey summary dated October 21 | Its weight-only and unmeasured-MLP descriptions apply to that earlier survey. |
| Qwen2.5 weight study | Cross-checkpoint keys, biases and norms | The two mapped blocks are measured and the doubled MLP width is read from pinned configs; the channel-by-channel MLP transformation remains unmeasured. |
| Qwen3 weight study | Within-model MLPs plus cross-checkpoint K/V, gate rows and sampled heads | Within 32B, layers 32–55 repeat layers 8–31. Between checkpoints, 14B layers 0–9 match 32B 0–9 best, and 14B 10–39 match 32B 34–63 best. |
| Qwen3 response study | Actual activations under a separately pinned Q8 model | Response similarity is measured; a shared causal function is not established. |
| `code/as_run/` and frozen capture records | Historical code and recorded inputs | Preserved for provenance, including superseded descriptions and machine-specific paths. Use the current study instructions for reproduction. |

The current response description is 10,000 distinct text windows. Historical
independent-article wording is superseded by the separate corpus correction;
the captured tokens and measurements are preserved unchanged.

Current text distinguishes observations from the interpretation that blocks were
copied and trained further. The two cross-checkpoint studies add external trained
references beyond the original survey, while leaving construction direction and
timing unresolved. Cosine is not a measure of elapsed training tokens.

File sizes use decimal GB; GiB values are shown explicitly where useful.
Weight-table reductions, fresh tensor-byte measurement and response reprocessing
have different download requirements; see [reproduction](REPRODUCTION_V020.md).

Old releases and numerical as-run evidence are preserved. Public metadata path
redactions and descriptive/report-format amendments are listed in
[AMENDMENTS.json](AMENDMENTS.json), with before/after hashes. Current navigation and
scope notes explain the additions instead of silently rewriting those records.
