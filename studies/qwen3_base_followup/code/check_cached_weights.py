#!/usr/bin/env python3
"""Recompute the Base-reference comparisons from the same cached payloads.

Reads only. Verifies every payload hash (follow-up sidecars, fetched_payloads.json, and the release/study provenance for
reused tensors), recomputes the k/v cross matrices, the 544-row gate sample and the Base30 within-scan with the released scanner and compares against the study data. Writes only to --out.
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

import argparse
STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
parser = argparse.ArgumentParser(description="Recompute the Base study from retained raw tensor/range caches; no downloads.")
parser.add_argument('--tensor-cache', type=Path, required=True)
parser.add_argument('--range-cache', type=Path, required=True)
parser.add_argument('--out', type=Path, default=ROOT/'reproduction/base_cached_check.json')
args = parser.parse_args()
F = STUDY
C = ROOT
LEGACY = args.tensor_cache
OUT = args.out.resolve()
if OUT == STUDY or STUDY in OUT.parents:
    raise ValueError('Write checks outside the released study')
OUT.parent.mkdir(parents=True, exist_ok=True)
def range_path(task):
    return args.range_cache / Path(task['cache_path']).relative_to('cache')
sys.path.insert(0, str(ROOT / "scripts"))
import scan_weights as sw  # noqa: E402

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

report = {"checks": {}, "comparisons": {}}
lock = {m["slug"]: m for m in json.loads((F / "models.lock.json").read_text())["models"]}
plan = json.loads((F / "FETCH_PLAN.json").read_text())
fetched = {r["cache_path"]: r for r in json.loads((F / "provenance/fetched_payloads.json").read_text())}
rows = np.array(plan["sample_rows"])
chunks = rows.reshape(-1, 32)
assert len(rows) == 544 and all((c == c[0] + np.arange(32)).all() for c in chunks) and rows.max() < 17408 and len(set(rows)) == 544
report["checks"]["sample_rows"] = "544 rows, 17 contiguous 32-row chunks inside 0-17407, all distinct"

# 1. payload hashes of everything the follow-up fetched
bad = 0
for t in plan["tasks"]:
    p = range_path(t); side = json.loads(p.with_suffix(".json").read_text())
    h = sha(p)
    ok = p.stat().st_size == t["n_bytes"] and h == side["sha256"] == fetched[t["cache_path"]]["sha256"]
    bad += (not ok)
report["checks"]["followup_payloads"] = {"tasks": len(plan["tasks"]), "hash_or_size_failures": bad}

# expected hashes for reused release tensors (release survey + Qwen3 study provenance)
expected = {}
prov = json.loads((C / "data/provenance.json").read_text())
slug_of = {m["repo_id"]: m["slug"] for m in json.loads((C / "models.lock.json").read_text())["models"]}
for m in prov["models"]:
    for t in m["selected_tensors"]:
        expected[(slug_of[m["repo_id"]], t["tensor_name"])] = t
for t in json.loads((C / "studies/qwen3_weights/data/gate_proj/provenance.json").read_text())["tensors"]:
    expected[("qwen3-32b", t["tensor_name"])] = t
for name in ["summary.json", "mlp_gate/results.json"]:
    for slug, recs in json.loads((C / "studies/qwen3_weights/data/cross_qwen3_14b" / name).read_text())["provenance"].items():
        for tensor, rec in recs.items():
            expected[(slug, tensor)] = rec

def legacy_path(slug, name):
    return LEGACY / slug / lock[slug]["revision"] / (hashlib.sha256(name.encode()).hexdigest() + ".bin")

verified_legacy = 0
def load_full(slug, family, layer):
    """Full tensor as float64 2-D array, from the follow-up cache (new models) or the legacy cache (release models)."""
    name = f"model.layers.{layer}.{family}"
    lp = legacy_path(slug, name)
    if lp.exists() and (slug, name) in expected:
        exp = expected[(slug, name)]
        global verified_legacy
        assert sha(lp) == exp["sha256"], (slug, name)
        verified_legacy += 1
        return sw.decode_tensor(lp.read_bytes(), exp["dtype"], exp["shape"]).astype(np.float64).reshape(exp["shape"])
    t = next(x for x in plan["tasks"] if x["slug"] == slug and x["tensor_name"] == name and x["row_range"] is None)
    return sw.decode_tensor(range_path(t).read_bytes(), t["dtype"], t["shape"]).astype(np.float64).reshape(t["shape"])

def load_rows(slug, layer):
    """The 544 sampled gate rows, float64 [544, 5120]."""
    name = f"model.layers.{layer}.mlp.gate_proj.weight"
    lp = legacy_path(slug, name)
    if lp.exists() and (slug, name) in expected:
        exp = expected[(slug, name)]
        assert sha(lp) == exp["sha256"], (slug, name)
        global verified_legacy
        verified_legacy += 1
        mm = np.memmap(lp, dtype="<u2", mode="r", shape=tuple(exp["shape"]))
        raw = np.ascontiguousarray(mm[rows]).tobytes()
        return sw.decode_tensor(raw, "BF16", [544, exp["shape"][1]]).astype(np.float64).reshape(544, -1)
    parts = []
    for c in chunks:
        t = next(x for x in plan["tasks"] if x["slug"] == slug and x["tensor_name"] == name and x["row_range"] == [int(c[0]), int(c[-1]) + 1])
        parts.append(sw.decode_tensor(range_path(t).read_bytes(), t["dtype"], t["shape"]).astype(np.float64).reshape(t["shape"]))
    return np.concatenate(parts)

def unit_stack(slug, family, depth):
    out = None
    for l in range(depth):
        a = load_full(slug, family, l).ravel()
        if out is None: out = np.empty((depth, a.size))
        out[l] = a / np.sqrt(a @ a)
    assert np.max(np.abs(np.einsum("ij,ij->i", out, out) - 1)) <= 1e-9
    return out

def row_cos(a, b):
    na, nb = np.sqrt(np.einsum("ij,ij->i", a, a)), np.sqrt(np.einsum("ij,ij->i", b, b))
    return np.einsum("ij,ij->i", a, b) / (na * nb)

summary = json.loads((F / "data/summary.json").read_text())
# 2. k/v cross matrices and primary means
for fam, stem in [("self_attn.k_proj.weight", "k_proj"), ("self_attn.v_proj.weight", "v_proj")]:
    base, it, tgt = unit_stack("qwen3-14b-base", fam, 40), unit_stack("qwen3-14b", fam, 40), unit_stack("qwen3-32b", fam, 64)
    mats = {"base14_to_32": base @ tgt.T, "it14_to_32": it @ tgt.T, "base14_to_it14": base @ it.T}
    comp = {}
    for label, m in mats.items():
        theirs = np.load(F / "data" / f"{stem}_{label}.npy")
        comp[label + "_max_abs_diff_vs_codex"] = float(np.max(np.abs(m - theirs)))
    pb = np.array([mats["base14_to_32"][s, s + 24] for s in range(8, 32)]); pi = np.array([mats["it14_to_32"][s, s + 24] for s in range(8, 32)])
    comp.update(base_mean=float(pb.mean()), it_mean=float(pi.mean()), paired_difference_mean=float((pb - pi).mean()),
                base_to_it_same_layer_mean=float(np.diag(mats["base14_to_it14"]).mean()),
                codex_base_mean=summary[fam]["base_mean"], codex_it_mean=summary[fam]["it_mean"],
                codex_calibration=summary[fam]["base_to_it_same_layer_mean"])
    report["comparisons"][fam] = comp
    del base, it, tgt, mats

# 3. sampled gate rows
gates = {s: [load_rows(s, l) for l in range(d)] for s, d in [("qwen3-14b-base", 40), ("qwen3-14b", 40), ("qwen3-32b", 64)]}
means = {}
for label, sa, sb in [("base14_to_32", "qwen3-14b-base", "qwen3-32b"), ("it14_to_32", "qwen3-14b", "qwen3-32b")]:
    means[label] = float(np.mean([row_cos(gates[sa][s], gates[sb][s + 24]).mean() for s in range(8, 32)]))
cal = float(np.mean([row_cos(gates["qwen3-14b-base"][l], gates["qwen3-14b"][l]).mean() for l in range(40)]))
sg = summary["sampled_gate"]
report["comparisons"]["sampled_gate"] = {"base_row_mean": means["base14_to_32"], "it_row_mean": means["it14_to_32"],
    "paired_row_mean_difference": means["base14_to_32"] - means["it14_to_32"], "base_to_it_same_layer_row_mean": cal,
    "codex_base_row_mean": sg["base14_to_32"]["row_mean"], "codex_it_row_mean": sg["it14_to_32"]["row_mean"],
    "codex_calibration": sg["base_to_it_same_layer_row_mean"]}
del gates

# 4. Base30 within-scan with this checkout's scanner
for fam in ["self_attn.k_proj.weight", "mlp.gate.weight"]:
    full = {l: load_full("qwen3-30b-a3b-base", fam, l) for l in range(48)}
    shape = list(full[0].shape)
    tensors = {l: a.ravel() for l, a in full.items()}  # the scanner takes flat vectors plus the 2-D shape
    r, o, s = sw.similarities(tensors, 48, 12, (6, 41), 6, shape=shape)
    k = s["periods"]["12"]; ck = summary["base30_" + fam]["periods"]["12"]
    report["comparisons"]["base30_" + fam] = {"inside_mean": k["inside_mean"], "outside_mean": k["outside_mean"],
        "inside_row_mean": k.get("inside_row_mean"), "codex_inside_mean": ck["inside_mean"], "codex_outside_mean": ck["outside_mean"],
        "top_offset": s["top_offsets"][0]["offset"], "codex_top_offset": summary["base30_" + fam]["top_offsets"][0]["offset"]}
    del tensors

report["checks"]["reused_release_tensors_hash_verified"] = verified_legacy
report["scanner_sha256"] = sha(ROOT / "scripts/scan_weights.py")
assert report['checks']['followup_payloads']['hash_or_size_failures'] == 0
for family, comp in report['comparisons'].items():
    for key, value in comp.items():
        if key.endswith('_max_abs_diff_vs_codex'):
            assert value < 1e-12, (family, key)
    pairs = [('base_mean','codex_base_mean'), ('it_mean','codex_it_mean'),
             ('base_to_it_same_layer_mean','codex_calibration'),
             ('base_row_mean','codex_base_row_mean'), ('it_row_mean','codex_it_row_mean'),
             ('base_to_it_same_layer_row_mean','codex_calibration'),
             ('inside_mean','codex_inside_mean'), ('outside_mean','codex_outside_mean')]
    for a,b in pairs:
        if a in comp:
            assert np.isclose(comp[a],comp[b],rtol=0,atol=1e-12),(family,a)
report['status'] = 'passed'
OUT.write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(report, indent=1))
