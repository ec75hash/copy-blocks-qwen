# Qwen2.5-14B / 32B measured correspondence

Generated from the six retained 48×64 cosine matrices and corrected source JSON.

All indices are zero-based decoder indices. Norm rows subtract the stored all-ones initialization.

| Family | 14B → 32B layer group | Pairs | Mean cosine | Min | Max |
|---|---|---:|---:|---:|---:|
| `self_attn.k_proj.weight` | 0-15 -> 0-15 | 16 | 0.580718 | 0.512289 | 0.733044 |
| `self_attn.k_proj.weight` | 16-31 -> 16-31 (twin A) | 16 | 0.484755 | 0.433466 | 0.561691 |
| `self_attn.k_proj.weight` | 16-31 -> 32-47 (twin B) | 16 | 0.738111 | 0.424927 | 0.840021 |
| `self_attn.k_proj.weight` | 32-47 -> 48-63 | 16 | 0.683175 | 0.536873 | 0.811241 |
| `self_attn.q_proj.bias` | 0-15 -> 0-15 | 16 | 0.995877 | 0.987255 | 0.998570 |
| `self_attn.q_proj.bias` | 16-31 -> 16-31 (twin A) | 16 | 0.993225 | 0.989162 | 0.996348 |
| `self_attn.q_proj.bias` | 16-31 -> 32-47 (twin B) | 16 | 0.997633 | 0.996030 | 0.998720 |
| `self_attn.q_proj.bias` | 32-47 -> 48-63 | 16 | 0.998167 | 0.995916 | 0.998929 |
| `self_attn.k_proj.bias` | 0-15 -> 0-15 | 16 | 0.984978 | 0.967788 | 0.998920 |
| `self_attn.k_proj.bias` | 16-31 -> 16-31 (twin A) | 16 | 0.951968 | 0.920303 | 0.969551 |
| `self_attn.k_proj.bias` | 16-31 -> 32-47 (twin B) | 16 | 0.991127 | 0.980403 | 0.997650 |
| `self_attn.k_proj.bias` | 32-47 -> 48-63 | 16 | 0.986441 | 0.938712 | 0.997137 |
| `self_attn.v_proj.bias` | 0-15 -> 0-15 | 16 | 0.819312 | 0.543575 | 0.983748 |
| `self_attn.v_proj.bias` | 16-31 -> 16-31 (twin A) | 16 | 0.653286 | 0.325424 | 0.859913 |
| `self_attn.v_proj.bias` | 16-31 -> 32-47 (twin B) | 16 | 0.946377 | 0.856446 | 0.981383 |
| `self_attn.v_proj.bias` | 32-47 -> 48-63 | 16 | 0.987820 | 0.976629 | 0.997235 |
| `input_layernorm.weight` | 0-15 -> 0-15 | 16 | 0.999081 | 0.997608 | 0.999927 |
| `input_layernorm.weight` | 16-31 -> 16-31 (twin A) | 16 | 0.985030 | 0.972178 | 0.997781 |
| `input_layernorm.weight` | 16-31 -> 32-47 (twin B) | 16 | 0.991917 | 0.981741 | 0.998172 |
| `input_layernorm.weight` | 32-47 -> 48-63 | 16 | 0.987199 | 0.957278 | 0.998937 |
| `post_attention_layernorm.weight` | 0-15 -> 0-15 | 16 | 0.997615 | 0.987650 | 0.999855 |
| `post_attention_layernorm.weight` | 16-31 -> 16-31 (twin A) | 16 | 0.998227 | 0.997014 | 0.999425 |
| `post_attention_layernorm.weight` | 16-31 -> 32-47 (twin B) | 16 | 0.991016 | 0.987239 | 0.996569 |
| `post_attention_layernorm.weight` | 32-47 -> 48-63 | 16 | 0.984253 | 0.950781 | 0.992561 |

## Off-map background

The 64 specified mapped pairs are excluded explicitly; 3,008 pairs remain per family.

| Family | Off-map max absolute cosine | Off-map signed mean |
|---|---:|---:|
| `self_attn.k_proj.weight` | 0.004315 | -0.000013 |
| `self_attn.q_proj.bias` | 0.178094 | 0.000173 |
| `self_attn.k_proj.bias` | 0.243582 | 0.000249 |
| `self_attn.v_proj.bias` | 0.304903 | 0.000244 |
| `input_layernorm.weight` | 0.998541 | 0.852944 |
| `post_attention_layernorm.weight` | 0.999648 | 0.735191 |

High cross-layer norm similarity is common; the key and layer-specific bias maps locate the correspondence.
Endpoint residuals are in `analysis/endpoint_residuals.csv`; these differences from the 14B reference do not date training updates.
