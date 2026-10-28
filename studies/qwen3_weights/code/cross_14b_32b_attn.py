#!/usr/bin/env python3
"""CROSS-CHECKPOINT comparison (kept separate from the within-checkpoint survey): Qwen3-14B (40 layers) vs Qwen3-32B (64 layers)
attention k_proj/v_proj at the pinned revisions. Both are [1024, 5120] (8 KV heads x 128, hidden 5120), so cosines are direct.
Hypothesis under test: 32B = 14B grown by the 24-layer block copy (8-31 -> 32-55), i.e. 14B L -> 32B L for L < 32 and 14B L -> 32B L+24 for L >= 32.
Writes out/cross_qwen3_14b/{cosine_<fam>.npy, summary.json, fig_cross.png, provenance.json}."""
import sys, os, json, struct, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts")); import scan_weights as sw
from huggingface_hub import hf_hub_download, hf_hub_url
OUT = Path(os.environ.get("COPY_BLOCKS_OUTPUT", str(Path(__file__).resolve().parents[3] / "reproduction/qwen3_weights"))) / 'cross_qwen3_14b'; OUT.mkdir(parents=True, exist_ok=True); CACHE = Path(os.environ.get("COPY_BLOCKS_TENSOR_CACHE", str(Path.home() / ".cache/copy-blocks-qwen-tensors")))
MODELS = {"qwen3-14b": ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", 40), "qwen3-32b": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", 64)}
FAMS = ["self_attn.k_proj.weight", "self_attn.v_proj.weight"]; CACHE.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def log(*x): print(f"[{time.time()-t0:5.0f}s]", *x, flush=True)
prov = {}
def get_family(slug, fam):
    repo, rev, depth = MODELS[slug]
    wm = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=rev, token=False)).read_text())["weight_map"]
    names = sw.decoder_names(wm, "model.layers.", fam, depth); headers = {}
    def load_header(shard):
        if shard not in headers:
            url = hf_hub_url(repo, shard, revision=rev); hl = struct.unpack("<Q", sw.read_range(url, 0, 7))[0]; raw = sw.read_range(url, 8, 7 + hl)
            headers[shard] = (url, json.loads(raw), 8 + hl, sw.digest(raw))
        return headers[shard][1]
    allnames = [names[i] for i in range(depth)]
    for sh in sorted({wm[n] for n in allnames}): load_header(sh)
    resolved = sw.resolve_shards(allnames, wm, load_header)
    def fetch(name):
        url, header, base, hh = headers[resolved[name]]; e = header[name]; s, t = e["data_offsets"]; p = CACHE / slug / rev / (sw.digest(name.encode()) + ".bin")
        if not (p.exists() and p.stat().st_size == t - s):
            raw = sw.read_range(url, base + s, base + t - 1); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
        raw = p.read_bytes(); prov.setdefault(slug, {})[name] = {"sha256": sw.digest(raw), "shape": e["shape"], "dtype": e["dtype"], "shard": resolved[name]}
        return sw.decode_tensor(raw, e["dtype"], e["shape"]).reshape(e["shape"])
    with ThreadPoolExecutor(max_workers=6) as pool: arrs = list(pool.map(fetch, allnames))
    log(f"{slug} {fam}: {len(arrs)} tensors {arrs[0].shape}")
    return arrs
summary = {}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for fi, fam in enumerate(FAMS):
    A = get_family("qwen3-14b", fam); B = get_family("qwen3-32b", fam)
    An = np.stack([a.astype(np.float64).ravel() for a in A]); Bn = np.stack([b.astype(np.float64).ravel() for b in B])
    An /= np.linalg.norm(An, axis=1, keepdims=True); Bn /= np.linalg.norm(Bn, axis=1, keepdims=True)
    C = An @ Bn.T   # [40, 64]
    np.save(OUT / f"cosine_{fam.split('.')[1]}.npy", C)
    best = C.argmax(1); bestv = C.max(1)
    hyp = [L if L < 32 else L + 24 for L in range(40)]
    hypv = np.array([C[L, hyp[L]] for L in range(40)])
    # the twin of the copied region in 32B: 14B L (8..31) vs 32B L+24 (32..55)
    twin = np.array([C[L, L + 24] for L in range(8, 32)])
    offdiag = np.array([C[L, j] for L in range(40) for j in range(64) if j != hyp[L] and not (8 <= L <= 31 and j == L + 24)])
    summary[fam] = {"best_match_per_14B_layer": [int(b) for b in best], "best_value": [round(float(v), 4) for v in bestv],
                    "hypothesis_map_value": [round(float(v), 4) for v in hypv], "hypothesis_map_mean": float(hypv.mean()),
                    "hypothesis_match_rate": float(np.mean(best == np.array(hyp))), "twin_region_32B_L+24_mean": float(twin.mean()),
                    "offdiag_mean": float(offdiag.mean()), "offdiag_max": float(offdiag.max())}
    log(f"{fam}: hypothesis map mean cosine {hypv.mean():.4f} (match rate {np.mean(best == np.array(hyp)):.2f}); 14B L vs 32B L+24 in copied region {twin.mean():.4f}; off-diagonal mean {offdiag.mean():.4f} max {offdiag.max():.4f}")
    log("  per-layer hypothesis cosines:", " ".join(f"{v:.2f}" for v in hypv))
    ax = axes[fi]; im = ax.imshow(C, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.05, float(C.max()))); ax.set_xlabel("Qwen3-32B layer"); ax.set_ylabel("Qwen3-14B layer"); ax.set_title(f"{fam}: cosine of flattened weights"); plt.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(OUT / "fig_cross.png", dpi=140)
json.dump({"models": MODELS, "families": FAMS, "summary": summary, "provenance": prov}, open(OUT / "summary.json", "w"), indent=1)
log("done")
