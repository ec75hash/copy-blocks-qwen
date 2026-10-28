#!/usr/bin/env python3
"""Streaming scan of Qwen3-32B MLP families at every layer offset (pinned revision, byte-range reads, no model execution).
Reuses the release scanner's range/decode/shard-resolution code read-only. Memory-bounded: tensors are cached raw on disk and
processed in blocks of --block layers. Per layer pair: flattened float64 cosine and per-neuron aligned cosine summaries.
  python scan_mlp_32b.py --family mlp.gate_proj.weight --out out/gate_proj
"""
import sys, os, json, time, argparse, struct, hashlib, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, "/Volumes/ExternalSSD/copy-blocks-qwen-release/scripts")
import scan_weights as sw   # read-only reuse: read_range, decode_tensor, resolve_shards, digest
from huggingface_hub import hf_hub_download, hf_hub_url

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3-32B"); ap.add_argument("--revision", default="9216db5781bf21249d130ec9da846c4624c16137")
ap.add_argument("--slug", default="qwen3-32b"); ap.add_argument("--family", required=True)
ap.add_argument("--depth", type=int, default=64); ap.add_argument("--period", type=int, default=24); ap.add_argument("--region", default="8,55")
ap.add_argument("--neuron-axis", type=int, default=None, help="0: rows are neurons (gate/up); 1: columns are neurons (down). Default by name.")
ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "copy-blocks-qwen-tensors")
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--block", type=int, default=8)
ap.add_argument("--min-free-gb", type=float, default=20.0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
def log(*x):
    msg = f"[{time.time()-t0:6.0f}s] " + " ".join(str(v) for v in x); print(msg, flush=True)
    with open(a.out / "scan.log", "a") as f: f.write(msg + "\n")
neuron_axis = a.neuron_axis if a.neuron_axis is not None else (1 if "down_proj" in a.family else 0)
region = tuple(int(x) for x in a.region.split(",")); prefix = "model.layers."

# ---- pinned metadata ----
cfg_path = Path(hf_hub_download(a.repo, "config.json", revision=a.revision, token=False))
idx_path = Path(hf_hub_download(a.repo, "model.safetensors.index.json", revision=a.revision, token=False))
config = json.loads(cfg_path.read_text()); weight_map = json.loads(idx_path.read_text())["weight_map"]
assert int(config["num_hidden_layers"]) == a.depth
names = sw.decoder_names(weight_map, prefix, a.family, a.depth); assert set(names) == set(range(a.depth)), "family must cover every block"
headers = {}
def load_header(shard):
    if shard not in headers:
        url = hf_hub_url(a.repo, shard, revision=a.revision)
        hl = struct.unpack("<Q", sw.read_range(url, 0, 7))[0]; assert hl < 64 * 1024 * 1024
        raw = sw.read_range(url, 8, 7 + hl); headers[shard] = (url, json.loads(raw), 8 + hl, sw.digest(raw))
    return headers[shard][1]
all_names = [names[i] for i in range(a.depth)]
for shard in sorted({weight_map[n] for n in all_names}): load_header(shard)
resolved = sw.resolve_shards(all_names, weight_map, load_header)
log(f"{a.repo}@{a.revision[:12]} family {a.family}: {len(all_names)} tensors across {len({resolved[n] for n in all_names})} shards; neuron_axis={neuron_axis}")

# ---- phase 1: fetch to cache (raw bf16 bytes), manifest with hashes ----
def local_path(name): return a.cache / a.slug / a.revision / (sw.digest(name.encode()) + ".bin")
def fetch(name):
    shard = resolved[name]; url, header, base, hh = headers[shard]; entry = header[name]; s, e = entry["data_offsets"]; p = local_path(name)
    if p.exists() and p.stat().st_size == e - s: raw = None
    else:
        free_gb = shutil.disk_usage(a.cache).free / 1e9
        if free_gb < a.min_free_gb: raise RuntimeError(f"low disk: {free_gb:.1f} GB free")
        raw = sw.read_range(url, base + s, base + e - 1); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
    data = p.read_bytes() if raw is None else raw
    return {"tensor_name": name, "shard": shard, "dtype": entry["dtype"], "shape": entry["shape"], "n_bytes": len(data), "sha256": sw.digest(data),
            "shard_header_sha256": hh, "absolute_byte_start": base + s, "absolute_byte_end": base + e - 1, "cached": raw is None}
log("phase 1: fetching"); 
with ThreadPoolExecutor(max_workers=a.workers) as pool: manifests = list(pool.map(fetch, all_names))
json.dump({"repo_id": a.repo, "revision": a.revision, "family": a.family, "tensors": manifests}, open(a.out / "provenance.json", "w"), indent=1)
shape = manifests[0]["shape"]; dtype = manifests[0]["dtype"]; log(f"phase 1 done: {sum(m['n_bytes'] for m in manifests)/1e9:.1f} GB, shape {shape} {dtype}")

# ---- phase 2: streaming cosines ----
def load(i):
    m = manifests[i]; raw = local_path(m["tensor_name"]).read_bytes(); v = sw.decode_tensor(raw, m["dtype"], m["shape"]).reshape(m["shape"])
    return v if neuron_axis == 0 else np.ascontiguousarray(v.T)      # rows = neurons after this
def row_norms(v):  # float64 per-row norms, chunked
    out = np.empty(v.shape[0]); 
    for s in range(0, v.shape[0], 4096): c = v[s:s+4096].astype(np.float64); out[s:s+4096] = np.sqrt((c * c).sum(1))
    return out
def pair_stats(A, nA, B, nB, rng):
    # flattened cosine in float64
    flat = 0.0
    for s in range(0, A.shape[0], 4096): flat += float(np.einsum("ij,ij->", A[s:s+4096].astype(np.float64), B[s:s+4096].astype(np.float64)))
    flat /= (np.sqrt((nA * nA).sum()) * np.sqrt((nB * nB).sum()))
    rows = np.empty(A.shape[0])
    for s in range(0, A.shape[0], 4096): rows[s:s+4096] = np.einsum("ij,ij->i", A[s:s+4096].astype(np.float64), B[s:s+4096].astype(np.float64))
    rows = rows / (nA * nB)
    perm = rng.permutation(A.shape[0]); sh = np.empty(A.shape[0])
    for s in range(0, A.shape[0], 4096):
        idx = perm[s:s+4096]; sh[s:s+4096] = np.einsum("ij,ij->i", A[s:s+4096].astype(np.float64), B[idx].astype(np.float64)) / (nA[s:s+4096] * nB[idx])
    return flat, rows, sh
rng = np.random.default_rng(a.seed)
N = a.depth; blocks = [list(range(s, min(s + a.block, N))) for s in range(0, N, a.block)]
rows_out = []; neuron_vectors = {}; norms = {}
log(f"phase 2: {N*(N-1)//2} pairs in {len(blocks)} blocks of {a.block}")
for bi, I in enumerate(blocks):
    TI = {i: load(i) for i in I}
    for i in I: norms[i] = row_norms(TI[i])
    for bj in range(bi, len(blocks)):
        J = blocks[bj]; TJ = TI if bj == bi else {j: load(j) for j in J}
        for j in J:
            if j not in norms: norms[j] = row_norms(TJ[j])
        for i in I:
            for j in J:
                if j <= i: continue
                flat, r, sh = pair_stats(TI[i], norms[i], TJ[j], norms[j], rng)
                inside = region[0] <= i <= region[1] and region[0] <= j <= region[1]
                corr = (j - i == a.period) and inside and (region[0] <= i <= region[1] - a.period)
                rows_out.append({"layer_a": i, "layer_b": j, "offset": j - i, "cosine": flat, "row_mean_cosine": float(r.mean()), "row_median_cosine": float(np.median(r)),
                                 "frac_row_gt_0.5": float((r > 0.5).mean()), "frac_row_gt_0.9": float((r > 0.9).mean()), "shuffled_row_mean": float(sh.mean()),
                                 "shuffled_row_p99": float(np.quantile(sh, 0.99)), "both_inside_region": inside, "corresponding": corr})
                if j - i == a.period: neuron_vectors[f"L{i}_L{j}"] = r.astype(np.float32); neuron_vectors[f"L{i}_L{j}_shuffled"] = sh.astype(np.float32)
        if bj != bi: del TJ
        log(f"  block {bi}x{bj} done ({len(rows_out)} pairs)")
    del TI
# ---- outputs ----
import csv
with open(a.out / "pairs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
spectrum = []
for off in range(1, N):
    sel = [r for r in rows_out if r["offset"] == off]
    spectrum.append({"offset": off, "n_pairs": len(sel), "mean_cosine": float(np.mean([r["cosine"] for r in sel])), "mean_row_mean_cosine": float(np.mean([r["row_mean_cosine"] for r in sel]))})
with open(a.out / "offsets.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(spectrum[0].keys())); w.writeheader(); w.writerows(spectrum)
def wo(off):
    sel = [r for r in rows_out if r["offset"] == off]; ins = [r for r in sel if r["corresponding"]]; outs = [r for r in sel if not r["corresponding"]]
    f = lambda rs, k: (float(np.mean([r[k] for r in rs])) if rs else None)
    return {"n_corresponding": len(ins), "n_remaining": len(outs), "within_cosine": f(ins, "cosine"), "off_cosine": f(outs, "cosine"),
            "within_row_mean": f(ins, "row_mean_cosine"), "off_row_mean": f(outs, "row_mean_cosine"), "within_frac_gt_0.5": f(ins, "frac_row_gt_0.5"),
            "off_frac_gt_0.5": f(outs, "frac_row_gt_0.5"), "within_shuffled_row_mean": f(ins, "shuffled_row_mean")}
ranked = sorted(spectrum, key=lambda s: -s["mean_cosine"])[:5]
summary = {"repo_id": a.repo, "revision": a.revision, "family": a.family, "shape": shape, "dtype": dtype, "neuron_axis": neuron_axis, "region": region, "period": a.period,
           "top_offsets": ranked, "by_period": {str(a.period): wo(a.period), str(2 * a.period): wo(2 * a.period)},
           "global_pair_mean": float(np.mean([r["cosine"] for r in rows_out])), "elapsed_s": round(time.time() - t0, 1)}
json.dump(summary, open(a.out / "summary.json", "w"), indent=1); np.savez_compressed(a.out / "neuron_cosines_offset24.npz", **neuron_vectors)
log("summary:", json.dumps(summary["by_period"]), "| top offsets:", [(s["offset"], round(s["mean_cosine"], 4)) for s in ranked])
