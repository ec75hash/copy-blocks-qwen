#!/usr/bin/env python3
"""Cross-checkpoint comparison, Qwen2.5-14B against Qwen2.5-32B, from the local tensor cache only (no downloads).

Kept separate from the within-checkpoint survey (AGENTS.md rule). Both checkpoints have hidden size 5120, 40 query
heads and 8 key/value heads, so k_proj [1024, 5120], the q/k/v biases and the RMSNorm gains are directly comparable.
Writes cross_qwen25_14b_32b.json next to this script.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scan_weights as scan  # noqa: E402

LOCK = {m["slug"]: m for m in json.loads((ROOT / "models.lock.json").read_text())["models"]}
CACHES = [Path.home() / ".cache/copy-blocks-qwen-tensors", ROOT / ".cache" / "tensors"]
SMALL, LARGE, DEPTH_S, DEPTH_L, K = "qwen25-14b", "qwen25-32b", 48, 64, 16
MAPPING = {"0-15 -> 0-15": [(l, l) for l in range(16)],
           "16-31 -> 16-31 (twin A)": [(l, l) for l in range(16, 32)],
           "16-31 -> 32-47 (twin B)": [(l, l + K) for l in range(16, 32)],
           "32-47 -> 48-63": [(l, l + K) for l in range(32, 48)]}
FAMILIES = [("self_attn.k_proj.weight", [1024 * 5120], 0.0), ("self_attn.q_proj.bias", [5120], 0.0),
            ("self_attn.k_proj.bias", [1024], 0.0), ("self_attn.v_proj.bias", [1024], 0.0),
            ("input_layernorm.weight", [5120], 1.0), ("post_attention_layernorm.weight", [5120], 1.0)]


def load(slug, name, shape):
    for cache in CACHES:
        path = cache / slug / LOCK[slug]["revision"] / (hashlib.sha256(name.encode()).hexdigest() + ".bin")
        if path.exists():
            raw = path.read_bytes()
            return scan.decode_tensor(raw, "BF16", shape).astype(np.float64), scan.digest(raw)
    raise FileNotFoundError(name)


def main():
    out = {"small": {"slug": SMALL, "revision": LOCK[SMALL]["revision"]}, "large": {"slug": LARGE, "revision": LOCK[LARGE]["revision"]},
           "families": {}, "payload_sha256": {}}
    for fam, shape, init in FAMILIES:
        small = {l: load(SMALL, f"model.layers.{l}.{fam}", shape) for l in range(DEPTH_S)}
        large = {l: load(LARGE, f"model.layers.{l}.{fam}", shape) for l in range(DEPTH_L)}
        for slug, tensors in ((SMALL, small), (LARGE, large)):
            for l, (_, sha) in tensors.items():
                out["payload_sha256"][f"{slug}/model.layers.{l}.{fam}"] = sha
        a = np.stack([(small[l][0] - init).ravel() for l in range(DEPTH_S)])
        b = np.stack([(large[l][0] - init).ravel() for l in range(DEPTH_L)])
        ua = a / np.linalg.norm(a, axis=1)[:, None]
        ub = b / np.linalg.norm(b, axis=1)[:, None]
        cross = ua @ ub.T                                     # 48 x 64, float64
        best = [(int(np.argmax(cross[l])), float(cross[l].max())) for l in range(DEPTH_S)]
        mapped = np.zeros(cross.shape, dtype=bool)                       # explicit off-map mask (audit 2026-10-27, board 733)
        for pairs in MAPPING.values():
            for s_, t_ in pairs:
                mapped[s_, t_] = True
        off = float(np.abs(cross[~mapped]).max())                        # max |cos| over the 3,008 unmapped pairs
        means = {name: float(np.mean([cross[s, t] for s, t in pairs])) for name, pairs in MAPPING.items()}
        # learned-update fingerprint: each twin's drift from the parent block, and whether the two drifts agree
        drift = []
        for s in range(16, 32):
            parent, twin_a, twin_b = a[s], b[s], b[s + K]
            d_a, d_b = twin_a - parent, twin_b - parent
            drift.append({"parent_layer": s, "rel_drift_a": float(np.linalg.norm(d_a) / np.linalg.norm(parent)),
                          "rel_drift_b": float(np.linalg.norm(d_b) / np.linalg.norm(parent)),
                          "drift_cosine_a_b": float(d_a @ d_b / np.linalg.norm(d_a) / np.linalg.norm(d_b)),
                          "twin_cosine": float(ub[s] @ ub[s + K])})
        out["families"][fam] = {"init_subtracted": init, "best_match": best, "off_mapping_max_abs_cosine": off,
                                "mapping_means": means, "drift": drift,
                                "drift_summary": {k: float(np.mean([d[k] for d in drift])) for k in ("rel_drift_a", "rel_drift_b", "drift_cosine_a_b", "twin_cosine")}}
        if fam == "self_attn.k_proj.weight":
            out["k_proj_cross_matrix"] = cross.tolist()
        print(f"{fam:34s} " + " ".join(f"{name}: {means[name]:.3f}" for name in MAPPING) + f" | off-map max {off:.4f} | "
              f"drift A {out['families'][fam]['drift_summary']['rel_drift_a']:.3f}, B {out['families'][fam]['drift_summary']['rel_drift_b']:.3f}, "
              f"cos(dA,dB) {out['families'][fam]['drift_summary']['drift_cosine_a_b']:.3f}", flush=True)
    out["scan_weights_sha256"] = scan.digest((ROOT / "scripts/scan_weights.py").read_bytes())
    (Path(__file__).resolve().parent / "cross_qwen25_14b_32b.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
