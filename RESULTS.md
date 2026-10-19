# Numerical results

Regenerated from `data/summary.json`, `data/offsets.csv` and `data/pairs.csv` by `scripts/make_report.py`.

Cosines use flattened decoder-only tensors; the row-mean column is the mean over aligned rows (experts, for a router). Within-block means use the aligned pairs between corresponding blocks of the lock's hypothesis; off-block means use the remaining pairs at the same offset. A proposed block counts as confirmed when the top-ranked offset is K or 2K, the within-block mean at K is ≥ 0.10 and exceeds the off-block mean at K by ≥ 0.10. A no-block row holds when no offset's mean stands more than 0.05 above the mean of its two neighbouring offsets (the spike column); routers in every Qwen MoE share a smooth 0.3–0.5 similarity between neighbouring layers, which is drift, not repetition. These are reporting rules, not significance tests.

| Checkpoint | Role | Tensor | Blocks with tensor | K | Top offset (mean, pairs) | Within-block mean at K (pairs) | Off-block mean at K | Row-mean at K | Max spike | Verdict |
|---|---|---|---:|---:|---|---|---:|---:|---:|---|
| Qwen1.5-MoE-A2.7B | no_block_reported | `self_attn.k_proj.weight` | 24/24 | 8 | 1 (0.000, n=23) | 0.000 (16) | — | 0.000 | 0.001 | **no block** |
| Qwen1.5-MoE-A2.7B | no_block_reported | `mlp.gate.weight` | 24/24 | 8 | 1 (0.389, n=23) | 0.258 (16) | — | 0.260 | 0.003 | **no block** |
| Qwen2-57B-A14B | no_block_reported | `self_attn.k_proj.weight` | 28/28 | 8 | 7 (0.001, n=21) | -0.000 (16) | -0.001 | 0.000 | 0.001 | **no block** |
| Qwen2-57B-A14B | no_block_reported | `mlp.gate.weight` | 28/28 | 8 | 1 (0.323, n=27) | 0.058 (16) | 0.105 | 0.049 | 0.003 | **no block** |
| Qwen2.5-7B | negative_control | `self_attn.k_proj.weight` | 28/28 | 16 | 6 (0.001, n=22) | — (0) | 0.000 | — | 0.000 | **no block** |
| Qwen2.5-Math-7B | no_block_reported | `self_attn.k_proj.weight` | 28/28 | 16 | 21 (0.001, n=7) | — (0) | 0.000 | — | 0.001 | **no block** |
| Qwen2.5-14B | no_block_reported | `self_attn.k_proj.weight` | 48/48 | 16 | 16 (0.000, n=32) | 0.000 (32) | — | 0.000 | 0.000 | **no block** |
| Qwen2.5-32B | primary | `self_attn.k_proj.weight` | 64/64 | 16 | 16 (0.142, n=48) | 0.424 (16) | 0.000 | 0.352 | 0.142 | **confirmed** |
| DeepSeek-R1-Distill-Qwen-32B | derivative_of_qwen25_32b | `self_attn.k_proj.weight` | 64/64 | 16 | 16 (0.141, n=48) | 0.422 (16) | 0.000 | 0.351 | 0.141 | **confirmed** |
| KAT-V1-40B (grown from Qwen2.5-32B) | disclosed_upscale_of_qwen25_32b | `self_attn.k_proj.weight` | 80/80 | 8 | 8 (0.218, n=72) | 0.981 (16) | -0.000 | 0.974 | 0.218 | **confirmed** |
| Qwen3-0.6B | no_block_reported | `self_attn.k_proj.weight` | 28/28 | 8 | 14 (0.001, n=14) | -0.000 (16) | 0.000 | -0.000 | 0.001 | **no block** |
| Qwen3-1.7B | no_block_reported | `self_attn.k_proj.weight` | 28/28 | 8 | 11 (0.000, n=17) | 0.000 (16) | -0.000 | 0.000 | 0.000 | **no block** |
| Qwen3-4B | no_block_reported | `self_attn.k_proj.weight` | 36/36 | 12 | 30 (0.001, n=6) | -0.000 (24) | — | -0.000 | 0.001 | **no block** |
| Qwen3-8B | no_block_reported | `self_attn.k_proj.weight` | 36/36 | 12 | 30 (0.000, n=6) | -0.000 (24) | — | -0.000 | 0.000 | **no block** |
| Qwen3-14B | no_block_reported | `self_attn.k_proj.weight` | 40/40 | 8 | 3 (0.000, n=37) | -0.000 (32) | — | -0.000 | 0.000 | **no block** |
| Qwen3-30B-A3B | primary | `self_attn.k_proj.weight` | 48/48 | 12 | 12 (0.355, n=36) | 0.532 (24) | -0.000 | 0.475 | 0.355 | **confirmed** |
| Qwen3-30B-A3B | primary | `mlp.gate.weight` | 48/48 | 12 | 12 (0.694, n=36) | 0.878 (24) | 0.326 | 0.882 | 0.297 | **confirmed** |
| Qwen3-30B-A3B-Instruct-2507 | primary_post_trained_variant | `self_attn.k_proj.weight` | 48/48 | 12 | 12 (0.354, n=36) | 0.530 (24) | -0.000 | 0.474 | 0.354 | **confirmed** |
| Qwen3-30B-A3B-Instruct-2507 | primary_post_trained_variant | `mlp.gate.weight` | 48/48 | 12 | 12 (0.693, n=36) | 0.878 (24) | 0.322 | 0.883 | 0.301 | **confirmed** |
| Qwen3-32B | primary | `self_attn.k_proj.weight` | 64/64 | 24 | 24 (0.152, n=40) | 0.254 (24) | -0.000 | 0.227 | 0.152 | **confirmed** |
| Qwen3-Next-80B-A3B | primary | `linear_attn.conv1d.weight` | 36/48 | 12 | 12 (0.625, n=27) | 0.940 (18) | -0.005 | — | 0.623 | **confirmed** |
| Qwen3-Next-80B-A3B | primary | `self_attn.k_proj.weight` | 12/48 | 12 | 12 (0.421, n=9) | 0.632 (6) | -0.000 | 0.590 | — | **confirmed** |
| Qwen3-Next-80B-A3B | primary | `mlp.gate.weight` | 48/48 | 12 | 12 (0.705, n=36) | 0.919 (24) | 0.278 | 0.921 | 0.315 | **confirmed** |
| Qwen3-235B-A22B | primary | `self_attn.k_proj.weight` | 94/94 | 34 | 34 (0.299, n=60) | 0.528 (34) | -0.000 | 0.484 | 0.299 | **confirmed** |
| Qwen3-235B-A22B | primary | `mlp.gate.weight` | 94/94 | 34 | 34 (0.600, n=60) | 0.881 (34) | 0.231 | 0.883 | 0.250 | **confirmed** |
| Qwen3-Coder-480B-A35B | no_block_reported | `self_attn.k_proj.weight` | 62/62 | 16 | 39 (0.000, n=23) | 0.000 (32) | 0.000 | 0.000 | 0.000 | **no block** |
| Qwen3-Coder-480B-A35B | no_block_reported | `mlp.gate.weight` | 62/62 | 16 | 1 (0.513, n=61) | 0.144 (32) | 0.298 | 0.146 | 0.001 | **no block** |
| Qwen3.5-0.8B | no_block_reported | `linear_attn.conv1d.weight` | 18/24 | 8 | 13 (0.008, n=6) | 0.000 (12) | — | — | 0.005 | **no block** |
| Qwen3.5-0.8B | no_block_reported | `self_attn.k_proj.weight` | 6/24 | 8 | — | 0.000 (4) | — | -0.002 | — | **no eligible offsets** |
| Qwen3.5-2B | no_block_reported | `linear_attn.conv1d.weight` | 18/24 | 8 | 13 (0.006, n=6) | -0.003 (12) | — | — | 0.006 | **no block** |
| Qwen3.5-2B | no_block_reported | `self_attn.k_proj.weight` | 6/24 | 8 | — | -0.000 (4) | — | -0.001 | — | **no eligible offsets** |
| Qwen3.5-4B | no_block_reported | `linear_attn.conv1d.weight` | 24/32 | 8 | 19 (0.007, n=6) | -0.003 (18) | — | — | 0.010 | **no block** |
| Qwen3.5-4B | no_block_reported | `self_attn.k_proj.weight` | 8/32 | 8 | 4 (0.001, n=7) | 0.001 (6) | — | 0.000 | — | **no block** |
| Qwen3.5-9B | negative_control | `linear_attn.conv1d.weight` | 24/32 | 16 | 19 (0.007, n=6) | -0.000 (12) | — | — | 0.010 | **no block** |
| Qwen3.5-9B | negative_control | `self_attn.k_proj.weight` | 8/32 | 16 | 8 (0.001, n=6) | 0.000 (4) | — | -0.000 | — | **no block** |
| Qwen3.5-27B | primary | `linear_attn.conv1d.weight` | 48/64 | 16 | 16 (0.574, n=36) | 0.860 (24) | 0.001 | — | 0.573 | **confirmed** |
| Qwen3.5-27B | primary | `self_attn.k_proj.weight` | 16/64 | 16 | 16 (0.206, n=12) | 0.310 (8) | -0.000 | 0.290 | — | **confirmed** |
| Qwen3.6-27B | primary | `linear_attn.conv1d.weight` | 48/64 | 16 | 16 (0.573, n=36) | 0.859 (24) | 0.001 | — | 0.572 | **confirmed** |
| Qwen3.6-27B | primary | `self_attn.k_proj.weight` | 16/64 | 16 | 16 (0.206, n=12) | 0.309 (8) | -0.000 | 0.290 | — | **confirmed** |
| Qwen3.5-35B-A3B-Base | primary_base_variant | `linear_attn.conv1d.weight` | 30/40 | 12 | 12 (0.384, n=21) | 0.897 (9) | -0.001 | — | 0.382 | **confirmed** |
| Qwen3.5-35B-A3B-Base | primary_base_variant | `self_attn.k_proj.weight` | 10/40 | 12 | 12 (0.224, n=7) | 0.522 (3) | -0.001 | 0.475 | — | **confirmed** |
| Qwen3.5-35B-A3B-Base | primary_base_variant | `mlp.gate.weight` | 40/40 | 12 | 12 (0.485, n=28) | 0.789 (12) | 0.256 | 0.794 | 0.169 | **confirmed** |
| Qwen3.5-35B-A3B | primary | `linear_attn.conv1d.weight` | 30/40 | 12 | 12 (0.383, n=21) | 0.896 (9) | -0.001 | — | 0.381 | **confirmed** |
| Qwen3.5-35B-A3B | primary | `self_attn.k_proj.weight` | 10/40 | 12 | 12 (0.223, n=7) | 0.522 (3) | -0.001 | 0.475 | — | **confirmed** |
| Qwen3.5-35B-A3B | primary | `mlp.gate.weight` | 40/40 | 12 | 12 (0.485, n=28) | 0.789 (12) | 0.257 | 0.794 | 0.169 | **confirmed** |
| Qwen3.6-35B-A3B | primary | `linear_attn.conv1d.weight` | 30/40 | 12 | 12 (0.383, n=21) | 0.896 (9) | -0.001 | — | 0.381 | **confirmed** |
| Qwen3.6-35B-A3B | primary | `self_attn.k_proj.weight` | 10/40 | 12 | 12 (0.224, n=7) | 0.524 (3) | -0.000 | 0.477 | — | **confirmed** |
| Qwen3.6-35B-A3B | primary | `mlp.gate.weight` | 40/40 | 12 | 12 (0.483, n=28) | 0.791 (12) | 0.251 | 0.795 | 0.175 | **confirmed** |
| Qwen3.5-122B-A10B | no_block_reported | `linear_attn.conv1d.weight` | 36/48 | 16 | 35 (0.004, n=6) | -0.000 (24) | — | — | 0.005 | **no block** |
| Qwen3.5-122B-A10B | no_block_reported | `self_attn.k_proj.weight` | 12/48 | 16 | 4 (0.000, n=11) | -0.000 (8) | — | -0.000 | — | **no block** |
| Qwen3.5-122B-A10B | no_block_reported | `mlp.gate.weight` | 48/48 | 16 | 1 (0.409, n=47) | 0.107 (32) | — | 0.108 | 0.003 | **no block** |
| Qwen3.5-397B-A17B | no_block_reported | `linear_attn.conv1d.weight` | 45/60 | 16 | 35 (0.004, n=12) | 0.001 (24) | 0.001 | — | 0.005 | **no block** |
| Qwen3.5-397B-A17B | no_block_reported | `self_attn.k_proj.weight` | 15/60 | 16 | 4 (0.000, n=14) | -0.001 (8) | 0.000 | -0.000 | — | **no block** |
| Qwen3.5-397B-A17B | no_block_reported | `mlp.gate.weight` | 60/60 | 16 | 1 (0.476, n=59) | 0.239 (32) | 0.233 | 0.245 | 0.006 | **no block** |
| Qwen3.8-27B | primary_post_trained_variant | `linear_attn.conv1d.weight` | 48/64 | 16 | 16 (0.572, n=36) | 0.858 (24) | 0.001 | — | 0.571 | **confirmed** |
| Qwen3.8-27B | primary_post_trained_variant | `self_attn.k_proj.weight` | 16/64 | 16 | 16 (0.202, n=12) | 0.303 (8) | -0.000 | 0.283 | — | **confirmed** |
| Qwen3.8-Flash-Next | primary | `linear_attn.conv1d.weight` | 36/48 | 16 | 16 (0.484, n=24) | 0.969 (12) | -0.001 | — | 0.486 | **confirmed** |
| Qwen3.8-Flash-Next | primary | `self_attn.k_proj.weight` | 12/48 | 16 | 16 (0.415, n=8) | 0.830 (4) | -0.001 | 0.810 | — | **confirmed** |
| Qwen3.8-Flash-Next | primary | `mlp.gate.weight` | 48/48 | 16 | 16 (0.486, n=32) | 0.898 (16) | 0.074 | 0.900 | 0.322 | **confirmed** |
| Qwen3.8-2.4T-A95B | primary | `linear_attn.conv1d.weight` | 69/92 | 32 | 32 (0.507, n=45) | 0.950 (24) | 0.000 | — | 0.507 | **confirmed** |
| Qwen3.8-2.4T-A95B | primary | `self_attn.k_proj.weight` | 23/92 | 32 | 32 (0.284, n=15) | 0.532 (8) | -0.000 | 0.503 | — | **confirmed** |
| Qwen3.8-2.4T-A95B | primary | `mlp.gate.weight` | 92/92 | 32 | 32 (0.491, n=60) | 0.837 (32) | 0.095 | 0.839 | 0.236 | **confirmed** |
| SOLAR-10.7B-v1.0 | disclosed_duplication_control | `self_attn.k_proj.weight` | 48/48 | 16 | 16 (0.436, n=32) | 0.871 (16) | 0.000 | 0.821 | 0.436 | **confirmed** |

## Strongest offsets per family

- **Qwen1.5-MoE-A2.7B**, `self_attn.k_proj.weight`: k=1: 0.0003 (n=23); k=3: 0.0003 (n=21); k=8: 0.0003 (n=16); k=7: 0.0003 (n=17)
- **Qwen1.5-MoE-A2.7B**, `mlp.gate.weight`: k=1: 0.3891 (n=23); k=2: 0.3681 (n=22); k=3: 0.3452 (n=21); k=4: 0.3254 (n=20)
- **Qwen2-57B-A14B**, `self_attn.k_proj.weight`: k=7: 0.0006 (n=21); k=20: 0.0004 (n=8); k=21: 0.0004 (n=7); k=18: 0.0004 (n=10)
- **Qwen2-57B-A14B**, `mlp.gate.weight`: k=1: 0.3233 (n=27); k=2: 0.2585 (n=26); k=3: 0.1979 (n=25); k=4: 0.1493 (n=24)
- **Qwen2.5-7B**, `self_attn.k_proj.weight`: k=6: 0.0006 (n=22); k=7: 0.0005 (n=21); k=21: 0.0005 (n=7); k=20: 0.0004 (n=8)
- **Qwen2.5-Math-7B**, `self_attn.k_proj.weight`: k=21: 0.0006 (n=7); k=7: 0.0006 (n=21); k=10: 0.0004 (n=18); k=13: 0.0003 (n=15)
- **Qwen2.5-14B**, `self_attn.k_proj.weight`: k=16: 0.0003 (n=32); k=6: 0.0002 (n=42); k=28: 0.0002 (n=20); k=40: 0.0002 (n=8)
- **Qwen2.5-32B**, `self_attn.k_proj.weight`: k=16: 0.1416 (n=48); k=56: 0.0003 (n=8); k=53: 0.0002 (n=11); k=22: 0.0002 (n=42)
- **DeepSeek-R1-Distill-Qwen-32B**, `self_attn.k_proj.weight`: k=16: 0.1410 (n=48); k=56: 0.0003 (n=8); k=53: 0.0002 (n=11); k=22: 0.0002 (n=42)
- **KAT-V1-40B (grown from Qwen2.5-32B)**, `self_attn.k_proj.weight`: k=8: 0.2177 (n=72); k=32: 0.1375 (n=48); k=24: 0.1185 (n=56); k=72: 0.0003 (n=8)
- **Qwen3-0.6B**, `self_attn.k_proj.weight`: k=14: 0.0007 (n=14); k=5: 0.0007 (n=23); k=21: 0.0006 (n=7); k=18: 0.0005 (n=10)
- **Qwen3-1.7B**, `self_attn.k_proj.weight`: k=11: 0.0004 (n=17); k=1: 0.0002 (n=27); k=19: 0.0002 (n=9); k=16: 0.0001 (n=12)
- **Qwen3-4B**, `self_attn.k_proj.weight`: k=30: 0.0005 (n=6); k=13: 0.0003 (n=23); k=5: 0.0003 (n=31); k=10: 0.0003 (n=26)
- **Qwen3-8B**, `self_attn.k_proj.weight`: k=30: 0.0004 (n=6); k=29: 0.0004 (n=7); k=13: 0.0003 (n=23); k=8: 0.0003 (n=28)
- **Qwen3-14B**, `self_attn.k_proj.weight`: k=3: 0.0003 (n=37); k=5: 0.0003 (n=35); k=31: 0.0002 (n=9); k=6: 0.0002 (n=34)
- **Qwen3-30B-A3B**, `self_attn.k_proj.weight`: k=12: 0.3545 (n=36); k=24: 0.1806 (n=24); k=14: 0.0010 (n=34); k=26: 0.0008 (n=22)
- **Qwen3-30B-A3B**, `mlp.gate.weight`: k=12: 0.6940 (n=36); k=24: 0.5269 (n=24); k=1: 0.4616 (n=47); k=2: 0.4497 (n=46)
- **Qwen3-30B-A3B-Instruct-2507**, `self_attn.k_proj.weight`: k=12: 0.3535 (n=36); k=24: 0.1800 (n=24); k=14: 0.0009 (n=34); k=26: 0.0009 (n=22)
- **Qwen3-30B-A3B-Instruct-2507**, `mlp.gate.weight`: k=12: 0.6929 (n=36); k=24: 0.5247 (n=24); k=1: 0.4553 (n=47); k=2: 0.4434 (n=46)
- **Qwen3-32B**, `self_attn.k_proj.weight`: k=24: 0.1524 (n=40); k=1: 0.0003 (n=63); k=6: 0.0003 (n=58); k=5: 0.0002 (n=59)
- **Qwen3-Next-80B-A3B**, `linear_attn.conv1d.weight`: k=12: 0.6254 (n=27); k=24: 0.4504 (n=18); k=37: 0.0064 (n=6); k=25: 0.0035 (n=12)
- **Qwen3-Next-80B-A3B**, `self_attn.k_proj.weight`: k=12: 0.4209 (n=9); k=24: 0.2674 (n=6); k=4: 0.0002 (n=11); k=16: 0.0001 (n=8)
- **Qwen3-Next-80B-A3B**, `mlp.gate.weight`: k=12: 0.7052 (n=36); k=24: 0.5546 (n=24); k=1: 0.4605 (n=47); k=2: 0.4335 (n=46)
- **Qwen3-235B-A22B**, `self_attn.k_proj.weight`: k=34: 0.2989 (n=60); k=20: 0.2406 (n=74); k=54: 0.2136 (n=40); k=14: 0.0934 (n=80)
- **Qwen3-235B-A22B**, `mlp.gate.weight`: k=34: 0.5996 (n=60); k=20: 0.5761 (n=74); k=54: 0.4832 (n=40); k=1: 0.4670 (n=93)
- **Qwen3-Coder-480B-A35B**, `self_attn.k_proj.weight`: k=39: 0.0002 (n=23); k=56: 0.0002 (n=6); k=3: 0.0002 (n=59); k=29: 0.0002 (n=33)
- **Qwen3-Coder-480B-A35B**, `mlp.gate.weight`: k=1: 0.5128 (n=61); k=2: 0.4818 (n=60); k=3: 0.4579 (n=59); k=4: 0.4354 (n=58)
- **Qwen3.5-0.8B**, `linear_attn.conv1d.weight`: k=13: 0.0082 (n=6); k=5: 0.0058 (n=10); k=9: 0.0029 (n=8); k=16: 0.0012 (n=6)
- **Qwen3.5-0.8B**, `self_attn.k_proj.weight`: 
- **Qwen3.5-2B**, `linear_attn.conv1d.weight`: k=13: 0.0063 (n=6); k=5: 0.0056 (n=10); k=6: 0.0024 (n=9); k=9: 0.0022 (n=8)
- **Qwen3.5-2B**, `self_attn.k_proj.weight`: 
- **Qwen3.5-4B**, `linear_attn.conv1d.weight`: k=19: 0.0074 (n=6); k=6: 0.0048 (n=13); k=21: 0.0028 (n=6); k=12: 0.0009 (n=15)
- **Qwen3.5-4B**, `self_attn.k_proj.weight`: k=4: 0.0011 (n=7); k=8: 0.0008 (n=6)
- **Qwen3.5-9B**, `linear_attn.conv1d.weight`: k=19: 0.0070 (n=6); k=6: 0.0040 (n=13); k=21: 0.0025 (n=6); k=9: 0.0007 (n=12)
- **Qwen3.5-9B**, `self_attn.k_proj.weight`: k=8: 0.0009 (n=6); k=4: 0.0005 (n=7)
- **Qwen3.5-27B**, `linear_attn.conv1d.weight`: k=16: 0.5737 (n=36); k=32: 0.4008 (n=24); k=53: 0.0036 (n=6); k=39: 0.0036 (n=12)
- **Qwen3.5-27B**, `self_attn.k_proj.weight`: k=16: 0.2063 (n=12); k=32: 0.0961 (n=8); k=20: 0.0005 (n=11); k=4: 0.0004 (n=15)
- **Qwen3.6-27B**, `linear_attn.conv1d.weight`: k=16: 0.5728 (n=36); k=32: 0.3999 (n=24); k=39: 0.0036 (n=12); k=53: 0.0035 (n=6)
- **Qwen3.6-27B**, `self_attn.k_proj.weight`: k=16: 0.2061 (n=12); k=32: 0.0956 (n=8); k=4: 0.0005 (n=15); k=20: 0.0004 (n=11)
- **Qwen3.5-35B-A3B-Base**, `linear_attn.conv1d.weight`: k=12: 0.3836 (n=21); k=32: 0.0038 (n=6); k=22: 0.0028 (n=9); k=3: 0.0026 (n=18)
- **Qwen3.5-35B-A3B-Base**, `self_attn.k_proj.weight`: k=12: 0.2236 (n=7); k=8: 0.0009 (n=8); k=4: 0.0005 (n=9); k=16: 0.0001 (n=6)
- **Qwen3.5-35B-A3B-Base**, `mlp.gate.weight`: k=12: 0.4847 (n=28); k=1: 0.4502 (n=39); k=2: 0.4231 (n=38); k=3: 0.4042 (n=37)
- **Qwen3.5-35B-A3B**, `linear_attn.conv1d.weight`: k=12: 0.3835 (n=21); k=32: 0.0038 (n=6); k=22: 0.0028 (n=9); k=3: 0.0026 (n=18)
- **Qwen3.5-35B-A3B**, `self_attn.k_proj.weight`: k=12: 0.2235 (n=7); k=8: 0.0009 (n=8); k=4: 0.0005 (n=9); k=16: 0.0001 (n=6)
- **Qwen3.5-35B-A3B**, `mlp.gate.weight`: k=12: 0.4845 (n=28); k=1: 0.4501 (n=39); k=2: 0.4230 (n=38); k=3: 0.4041 (n=37)
- **Qwen3.6-35B-A3B**, `linear_attn.conv1d.weight`: k=12: 0.3833 (n=21); k=32: 0.0038 (n=6); k=22: 0.0027 (n=9); k=3: 0.0027 (n=18)
- **Qwen3.6-35B-A3B**, `self_attn.k_proj.weight`: k=12: 0.2242 (n=7); k=8: 0.0009 (n=8); k=4: 0.0005 (n=9); k=16: 0.0002 (n=6)
- **Qwen3.6-35B-A3B**, `mlp.gate.weight`: k=12: 0.4828 (n=28); k=1: 0.4400 (n=39); k=2: 0.4136 (n=38); k=3: 0.3950 (n=37)
- **Qwen3.5-122B-A10B**, `linear_attn.conv1d.weight`: k=35: 0.0041 (n=6); k=34: 0.0040 (n=7); k=26: 0.0038 (n=11); k=17: 0.0032 (n=16)
- **Qwen3.5-122B-A10B**, `self_attn.k_proj.weight`: k=4: 0.0000 (n=11); k=24: -0.0001 (n=6); k=12: -0.0001 (n=9); k=16: -0.0003 (n=8)
- **Qwen3.5-122B-A10B**, `mlp.gate.weight`: k=1: 0.4089 (n=47); k=2: 0.3739 (n=46); k=3: 0.3421 (n=45); k=4: 0.3192 (n=44)
- **Qwen3.5-397B-A17B**, `linear_attn.conv1d.weight`: k=35: 0.0042 (n=12); k=47: 0.0026 (n=6); k=6: 0.0023 (n=27); k=49: 0.0023 (n=6)
- **Qwen3.5-397B-A17B**, `self_attn.k_proj.weight`: k=4: 0.0003 (n=14); k=8: 0.0002 (n=13); k=28: 0.0000 (n=8); k=32: -0.0000 (n=7)
- **Qwen3.5-397B-A17B**, `mlp.gate.weight`: k=1: 0.4757 (n=59); k=2: 0.4306 (n=58); k=3: 0.3922 (n=57); k=4: 0.3607 (n=56)
- **Qwen3.8-27B**, `linear_attn.conv1d.weight`: k=16: 0.5721 (n=36); k=32: 0.3984 (n=24); k=23: 0.0036 (n=20); k=39: 0.0036 (n=12)
- **Qwen3.8-27B**, `self_attn.k_proj.weight`: k=16: 0.2016 (n=12); k=32: 0.0917 (n=8); k=8: 0.0005 (n=14); k=20: 0.0005 (n=11)
- **Qwen3.8-Flash-Next**, `linear_attn.conv1d.weight`: k=16: 0.4839 (n=24); k=28: 0.0038 (n=15); k=5: 0.0030 (n=22); k=9: 0.0030 (n=20)
- **Qwen3.8-Flash-Next**, `self_attn.k_proj.weight`: k=16: 0.4148 (n=8); k=20: 0.0011 (n=7); k=4: 0.0010 (n=11); k=12: 0.0005 (n=9)
- **Qwen3.8-Flash-Next**, `mlp.gate.weight`: k=16: 0.4859 (n=32); k=1: 0.2626 (n=47); k=2: 0.2251 (n=46); k=3: 0.2012 (n=45)
- **Qwen3.8-2.4T-A95B**, `linear_attn.conv1d.weight`: k=32: 0.5066 (n=45); k=74: 0.0037 (n=9); k=78: 0.0035 (n=7); k=59: 0.0032 (n=16)
- **Qwen3.8-2.4T-A95B**, `self_attn.k_proj.weight`: k=32: 0.2837 (n=15); k=40: 0.0005 (n=13); k=8: 0.0004 (n=21); k=64: 0.0003 (n=7)
- **Qwen3.8-2.4T-A95B**, `mlp.gate.weight`: k=32: 0.4909 (n=60); k=1: 0.4112 (n=91); k=2: 0.3850 (n=90); k=3: 0.3637 (n=89)
- **SOLAR-10.7B-v1.0**, `self_attn.k_proj.weight`: k=16: 0.4358 (n=32); k=3: 0.0005 (n=45); k=19: 0.0005 (n=29); k=13: 0.0003 (n=35)
