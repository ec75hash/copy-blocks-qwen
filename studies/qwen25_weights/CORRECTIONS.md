# Qwen2.5 reporting correction, October 27, 2026

The original off-map summary discarded the two largest absolute entries in every
source row. Only layers 16–31 have two mapped targets; other layers have one.
The shortcut therefore also discarded the strongest off-map entry in those rows.
The corrected calculation explicitly masks the 64 mapped pairs and measures all
3,008 remaining pairs per family. Tensor values and mapped-pair means did not change.

| Family | Earlier reported maximum absolute off-map cosine | Corrected maximum |
|---|---:|---:|
| Key weight | 0.003277328 | 0.004314988 |
| Query bias | 0.160863093 | 0.178093753 |
| Key bias | 0.236404405 | 0.243581607 |
| Value bias | 0.275195909 | 0.304903260 |
| Input norm, minus one | 0.996862179 | 0.998541235 |
| Post-attention norm, minus one | 0.999219204 | 0.999647701 |

This release includes the corrected original code and capture JSON under
`code/as_run/` and `data/`, the portable cache reproducer, and corrected summaries
in `analysis/family_summary.csv` and `RESULTS.md`. The off-map mask is checked by
`code/verify_and_plot.py`. The revised maximum leaves the layer mapping unchanged.
The separate-session check re-decoded the same cached payloads; it did not fetch
fresh tensors or run the models.
