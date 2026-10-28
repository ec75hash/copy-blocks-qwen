#!/usr/bin/env python3
"""CROSS-CHECKPOINT: Qwen3-14B gate_proj [17408, 5120] vs Qwen3-32B gate_proj [25600, 5120] rows 0-17407, at the layer map found in the
attention cross-check (14B L -> 32B L for L<8 and L in {8,9}; -> 32B L+24 for 10<=L<=39; both candidates reported for 8<=L<=31).
Per-neuron aligned cosine (row i vs row i), flattened cosine over the shared rows, and a novelty check for 32B rows 17408-25599
(best cosine against any 14B row of the mapped layer, on a sample). Also the 14B's own row statistics by index segment."""
import sys, os, json, struct, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts")); import scan_weights as sw
from huggingface_hub import hf_hub_download, hf_hub_url
OUT = Path(os.environ.get("COPY_BLOCKS_OUTPUT", str(Path(__file__).resolve().parents[3] / "reproduction/qwen3_weights"))) / 'cross_qwen3_14b/mlp_gate'; OUT.mkdir(parents=True, exist_ok=True); CACHE = Path(os.environ.get("COPY_BLOCKS_TENSOR_CACHE", str(Path.home() / ".cache/copy-blocks-qwen-tensors")))
P14 = ("qwen3-14b", "Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", 40); P32 = ("qwen3-32b", "Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", 64)
FAM = "mlp.gate_proj.weight"; CACHE.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def log(*x):
    m = f"[{time.time()-t0:5.0f}s] " + " ".join(str(v) for v in x); print(m, flush=True); open(OUT / "log.txt", "a").write(m + "\n")
prov = {}
def family(slug, repo, rev, depth, fetch_missing=True):
    wm = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=rev, token=False)).read_text())["weight_map"]
    names = sw.decoder_names(wm, "model.layers.", FAM, depth); headers = {}
    def load_header(shard):
        if shard not in headers:
            url = hf_hub_url(repo, shard, revision=rev); hl = struct.unpack("<Q", sw.read_range(url, 0, 7))[0]; raw = sw.read_range(url, 8, 7 + hl)
            headers[shard] = (url, json.loads(raw), 8 + hl, sw.digest(raw))
        return headers[shard][1]
    allnames = [names[i] for i in range(depth)]
    for sh in sorted({wm[n] for n in allnames}): load_header(sh)
    resolved = sw.resolve_shards(allnames, wm, load_header)
    def ensure(name):
        url, header, base, hh = headers[resolved[name]]; e = header[name]; s, t = e["data_offsets"]; p = CACHE / slug / rev / (sw.digest(name.encode()) + ".bin")
        if not (p.exists() and p.stat().st_size == t - s):
            if not fetch_missing: raise RuntimeError(f"missing cached payload {name}")
            raw = sw.read_range(url, base + s, base + t - 1); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
        return name, p, e
    with ThreadPoolExecutor(max_workers=6) as pool: entries = list(pool.map(ensure, allnames))
    def loader(L):
        name, p, e = entries[L]; raw = p.read_bytes(); prov.setdefault(slug, {})[name] = {"sha256": sw.digest(raw), "shape": e["shape"], "dtype": e["dtype"]}
        return sw.decode_tensor(raw, e["dtype"], e["shape"]).reshape(e["shape"])
    return loader
log("ensuring 32B gate payloads are cached (from the within-checkpoint scan)"); load32 = family(*P32, fetch_missing=True)
log("fetching 14B gate_proj (40 x 178 MB)"); load14 = family(*P14); log("14B fetched")
def rowcos(A, B):
    out = np.empty(A.shape[0])
    for s in range(0, A.shape[0], 4096):
        a = A[s:s+4096].astype(np.float64); b = B[s:s+4096].astype(np.float64); out[s:s+4096] = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return out
def flatcos(A, B):
    num = 0.0; na = 0.0; nb = 0.0
    for s in range(0, A.shape[0], 4096):
        a = A[s:s+4096].astype(np.float64); b = B[s:s+4096].astype(np.float64); num += (a * b).sum(); na += (a * a).sum(); nb += (b * b).sum()
    return num / np.sqrt(na * nb)
segs = {"A_0_8191": (0, 8192), "B_8192_17407": (8192, 17408)}
rng = np.random.default_rng(0); results = []; per_neuron = {}
for L in range(40):
    W14 = load14(L); cands = []
    if L < 8 or L in (8, 9): cands.append(L)
    if 8 <= L <= 31: cands += ([L] if L not in (8, 9) else []) + [L + 24]
    if L >= 32: cands.append(L + 24)
    cands = sorted(set(cands)); rec = {"L14": L, "candidates": {}}
    for c in cands:
        W32 = load32(c); shared = W32[:17408]
        r = rowcos(W14, shared); fc = flatcos(W14, shared)
        d = {"flat_cosine_shared_rows": float(fc), "row_mean": float(r.mean()), "row_median": float(np.median(r)), "frac_gt_0.5": float((r > 0.5).mean()), "frac_gt_0.8": float((r > 0.8).mean()),
             "seg_row_mean": {k: float(r[a:b].mean()) for k, (a, b) in segs.items()}}
        # shuffled null
        perm = rng.permutation(17408); sh = rowcos(W14, shared[perm]); d["shuffled_row_mean"] = float(sh.mean()); d["shuffled_p99"] = float(np.quantile(sh, 0.99))
        # novelty of 32B rows 17408-25599: best cosine against any 14B row (sample 300 rows)
        idx = rng.choice(np.arange(17408, 25600), 300, replace=False); X = W32[idx].astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y = W14.astype(np.float32); Y /= np.linalg.norm(Y, axis=1, keepdims=True); best_new = np.abs(X @ Y.T).max(1)
        # reference: best |cosine| of a shared 32B row against any 14B row other than its own index (sample 300)
        idx2 = rng.choice(17408, 300, replace=False); X2 = shared[idx2].astype(np.float32); X2 /= np.linalg.norm(X2, axis=1, keepdims=True); M2 = np.abs(X2 @ Y.T); M2[np.arange(300), idx2] = 0; best_other = M2.max(1)
        d["new_rows_best_abs_cos_vs_any_14B_row_median"] = float(np.median(best_new)); d["new_rows_best_abs_cos_p95"] = float(np.quantile(best_new, 0.95))
        d["shared_rows_best_abs_cos_vs_other_14B_rows_median"] = float(np.median(best_other))
        rec["candidates"][str(c)] = d; per_neuron[f"L14_{L}_vs_L32_{c}"] = r.astype(np.float32)
        log(f"14B L{L:2d} vs 32B L{c:2d}: flat {fc:.3f} | row mean {r.mean():.3f} median {np.median(r):.3f} | >0.5 {(r>0.5).mean():.3f} >0.8 {(r>0.8).mean():.3f} | segA {r[:8192].mean():.3f} segB {r[8192:].mean():.3f} | null {sh.mean():.4f} | new-row best|cos| median {np.median(best_new):.3f} (shared-row vs other rows {np.median(best_other):.3f})")
    if L in (2, 14, 30):   # 14B's own segment statistics
        st = {}
        for k, (a, b) in segs.items():
            S = W14[a:b].astype(np.float64); rn = np.sqrt((S * S).sum(1)); st[k] = {"row_norm_mean": float(rn.mean()), "row_norm_sd": float(rn.std()), "kurtosis": float(((S - S.mean()) ** 4).mean() / S.var() ** 2)}
        rec["own_segment_stats"] = st; log(f"  14B L{L} own segment stats: {json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in st.items()})}")
    results.append(rec)
json.dump({"parent": P14, "child": P32, "family": FAM, "results": results, "provenance": prov}, open(OUT / "results.json", "w"), indent=1)
np.savez_compressed(OUT / "per_neuron_cosines.npz", **per_neuron); log("done")
