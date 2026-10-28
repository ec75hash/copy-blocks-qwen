#!/usr/bin/env python3
"""CROSS-CHECKPOINT, attention heads: Qwen3-14B q_proj [5120, 5120] (40 heads x 128) vs Qwen3-32B q_proj [8192, 5120] (64 heads x 128).
Tests: (1) parent head h vs child head h (rows 128h..128h+127) at the mapped layer; (2) child heads 40-63 vs child heads 0-23 aligned
(duplication hypothesis, by analogy with the MLP neurons 17408+j = copy of j); (3) best-matching child head for each added head.
o_proj (columns are heads) is run only if >25 GB remain free. Outputs out/cross_qwen3_14b/heads/."""
import sys, os, json, struct, time, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts")); import scan_weights as sw
from huggingface_hub import hf_hub_download, hf_hub_url
OUT = Path(os.environ.get("COPY_BLOCKS_OUTPUT", str(Path(__file__).resolve().parents[3] / "reproduction/qwen3_weights"))) / 'cross_qwen3_14b/heads'; OUT.mkdir(parents=True, exist_ok=True); CACHE = Path(os.environ.get("COPY_BLOCKS_TENSOR_CACHE", str(Path.home() / ".cache/copy-blocks-qwen-tensors")))
P14 = ("qwen3-14b", "Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", 40); P32 = ("qwen3-32b", "Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", 64)
CACHE.mkdir(parents=True, exist_ok=True)
t0 = time.time()
def log(*x):
    m = f"[{time.time()-t0:5.0f}s] " + " ".join(str(v) for v in x); print(m, flush=True); open(OUT / "log.txt", "a").write(m + "\n")
prov = {}
def family(slug, repo, rev, depth, fam):
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
    def ensure(name):
        url, header, base, hh = headers[resolved[name]]; e = header[name]; s, t = e["data_offsets"]; p = CACHE / slug / rev / (sw.digest(name.encode()) + ".bin")
        if not (p.exists() and p.stat().st_size == t - s):
            if shutil.disk_usage(CACHE).free / 1e9 < 12: raise RuntimeError("low disk")
            raw = sw.read_range(url, base + s, base + t - 1); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
        return name, p, e
    with ThreadPoolExecutor(max_workers=6) as pool: entries = list(pool.map(ensure, allnames))
    def loader(L):
        name, p, e = entries[L]; raw = p.read_bytes(); prov.setdefault(slug, {})[name] = {"sha256": sw.digest(raw), "shape": e["shape"], "dtype": e["dtype"]}
        return sw.decode_tensor(raw, e["dtype"], e["shape"]).reshape(e["shape"])
    return loader
def heads(W, axis):  # -> [n_heads, 128*other] unit vectors; axis 0: rows are heads (q_proj), axis 1: columns are heads (o_proj)
    if axis == 1: W = np.ascontiguousarray(W.T)
    H = W.reshape(W.shape[0] // 128, 128 * W.shape[1]).astype(np.float64); return H / np.linalg.norm(H, axis=1, keepdims=True)
def run(fam, axis):
    log(f"=== {fam} ==="); l14 = family(*P14, fam); l32 = family(*P32, fam); rows = []
    for L in range(40):
        cands = ([L] if (L < 10) else []) + ([L + 24] if L >= 8 else [])
        H14 = heads(l14(L), axis)                       # [40, D]
        for c in cands:
            H32 = heads(l32(c), axis)                   # [64, D]
            pc = (H14 * H32[:40]).sum(1)                # parent head h vs child head h
            dup = (H32[40:64] * H32[0:24]).sum(1)       # child heads 40-63 vs 0-23
            M = np.abs(H32[40:64] @ H32[:40].T); best = M.argmax(1); bestv = M.max(1)
            Mp = np.abs(H32[40:64] @ H14.T); bestp = Mp.argmax(1)
            rec = {"L14": L, "L32": c, "parent_child_head_mean": float(pc.mean()), "parent_child_frac_gt_0.5": float((pc > 0.5).mean()),
                   "added_vs_first24_aligned_mean": float(dup.mean()), "added_best_child_head_is_h_minus_40": float((best == np.arange(24)).mean()),
                   "added_best_child_head_median_abscos": float(np.median(bestv)), "added_best_parent_head_is_h_minus_40": float((bestp == np.arange(24)).mean())}
            rows.append(rec)
            log(f"14B L{L:2d} vs 32B L{c:2d}: parent-child head cos {pc.mean():.3f} (>0.5 {(pc>0.5).mean():.2f}) | added heads 40-63 vs 0-23 aligned {dup.mean():.3f}; best child head == h-40: {(best==np.arange(24)).mean():.2f}; best parent head == h-40: {(bestp==np.arange(24)).mean():.2f}")
    json.dump(rows, open(OUT / f"{fam.split('.')[1]}.json", "w"), indent=1)
run("self_attn.q_proj.weight", 0)
if shutil.disk_usage(CACHE).free / 1e9 > 25: run("self_attn.o_proj.weight", 1)
else: log("skipping o_proj: low disk")
json.dump(prov, open(OUT / "provenance.json", "w"), indent=1); log("done")
