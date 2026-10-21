#!/usr/bin/env python3
"""Pinned, decoder-only weight similarities. No model execution or full shard download.

Derived from the Qwen3.6-27B release scanner (ec75hash/qwen-depth-upscaling-jpp-check v0.2.0).
Changes for the family scan: tensor families that exist on every decoder block (for example an
MoE router `mlp.gate.weight`) are accepted alongside attention-only families, each tensor pair
also reports an aligned per-row cosine when the tensor is two-dimensional, and `--models`
restricts a run to a subset of lock slugs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import struct
import time

from huggingface_hub import hf_hub_download, hf_hub_url
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def decoder_names(weight_map: dict, prefix: str, suffix: str, depth: int) -> dict[int, str]:
    # Anchored namespace: mtp.layers.0 must never masquerade as decoder layer 0.
    pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)\." + re.escape(suffix) + r"$")
    result = {}
    for name in weight_map:
        match = pattern.fullmatch(name)
        if match:
            layer = int(match.group(1))
            if not 0 <= layer < depth:
                raise ValueError(f"Decoder layer {layer} outside declared depth {depth}")
            result[layer] = name
    if not result:
        raise ValueError(f"No tensors for {prefix}*.{suffix}")
    return dict(sorted(result.items()))


def expected_layer_kind(suffix: str) -> str | None:
    """Which config.layer_types entries should carry this family; None means every block."""
    if suffix.startswith("linear_attn."):
        return "linear_attention"
    if suffix.startswith("self_attn."):
        return "full_attention"
    return None


def decode_tensor(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F32":
        values = np.frombuffer(raw, dtype="<f4").copy()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError("Wrong tensor size or nonfinite tensor")
    return values


def read_range(url: str, start: int, end: int) -> bytes:
    # Stream and reject a full-file response before reading its body. These shards are GBs.
    for attempt in range(3):
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                actual = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not actual.startswith(f"bytes {start}-{end}/"):
                    raise ValueError(f"Server did not honor requested byte range: status {response.status_code}")
                chunks = []
                size = 0
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > end - start + 1:
                        raise ValueError("Range response too large")
                    chunks.append(chunk)
                data = b"".join(chunks)
                if len(data) != end - start + 1:
                    raise ValueError("Truncated range response")
                return data
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                # Avoid exposing temporary signed URLs in network exceptions.
                raise RuntimeError(f"Byte-range request failed ({type(exc).__name__})") from None
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def resolve_shards(names, weight_map: dict, load_header) -> dict[str, str]:
    """Map each tensor to the shard whose header actually contains it.

    The Hub index is the first guess; when it is wrong (KAT-V1-40B's index names the wrong shard
    for most attention tensors) the other shards listed by the index are searched. The shard
    headers, not the index, are the ground truth for byte offsets.
    """
    all_shards = sorted(set(weight_map.values()))
    resolved = {}
    for name in names:
        indexed = weight_map[name]
        if name in load_header(indexed):
            resolved[name] = indexed
            continue
        for shard in all_shards:
            if shard != indexed and name in load_header(shard):
                resolved[name] = shard
                break
        else:
            raise ValueError(f"{name} is not present in any shard header")
    return resolved


def row_mean_cosines(tensors: dict[int, np.ndarray], ids: list[int], shape: list[int]) -> dict | None:
    """Mean over aligned rows of the per-row cosine (rows = experts for a router, kv rows for k_proj)."""
    if len(shape) != 2 or shape[0] < 2:
        return None
    rows, cols = int(shape[0]), int(shape[1])
    unit = {}
    for i in ids:
        matrix = tensors[i].reshape(rows, cols).astype(np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        if not np.isfinite(norms).all() or (norms == 0).any():
            return None
        unit[i] = matrix / norms[:, None]
    result = {}
    for a, left in enumerate(ids):
        for right in ids[a + 1:]:
            result[(left, right)] = float(np.einsum("ij,ij->i", unit[left], unit[right]).mean())
    return result


def derive_segments(region: tuple[int, int], period: int) -> list[list[int]]:
    """Consecutive length-`period` windows from the region start that fit inside the region."""
    start, end = region
    segments = []
    while start + period - 1 <= end:
        segments.append([start, start + period - 1])
        start += period
    return segments


def aligned_pairs(segments: list[list[int]], pair_groups: list[list[int]]) -> set[tuple[int, int]]:
    """(block, block) pairs at corresponding positions of two segments in the same group."""
    pairs = set()
    for group in pair_groups:
        for x in range(len(group)):
            for y in range(x + 1, len(group)):
                s, t = segments[group[x]], segments[group[y]]
                for k in range(min(s[1] - s[0], t[1] - t[0]) + 1):
                    pairs.add((s[0] + k, t[0] + k))
    return pairs


def similarities(tensors: dict[int, np.ndarray], depth: int, period: int,
                 region: tuple[int, int], min_pairs: int,
                 shape: list[int] | None = None,
                 segments: list[list[int]] | None = None,
                 pair_groups: list[list[int]] | None = None) -> tuple[list, list, dict]:
    # The block hypothesis: explicit segments (+ which segments correspond to which), or derived
    # from (region, period) as a single chain of consecutive copies. "Inside" pairs are the
    # aligned positions between corresponding segments; everything else at the same offset is
    # "outside" and serves as the floor.
    if segments is None:
        segments = derive_segments(tuple(region), period)
    if pair_groups is None:
        pair_groups = [list(range(len(segments)))]
    aligned = aligned_pairs(segments, pair_groups)
    ids = sorted(tensors)
    sizes = {v.size for v in tensors.values()}
    if len(sizes) != 1:
        raise ValueError("Tensor dimensions differ across layers; choose a homogeneous parameter family")
    # float64 throughout: on macOS arm64 the float32 BLAS Gram of 32+ multi-million-element rows
    # drifts by ~5e-3 (normalized diagonals come out 0.994), while small stacks are exact.
    matrix = np.stack([tensors[i] for i in ids]).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Zero or nonfinite norm")
    matrix /= norms[:, None]
    gram = matrix @ matrix.T
    diagonal_error = float(np.abs(np.diag(gram) - 1.0).max())
    if diagonal_error > 1e-9:
        raise ValueError(f"Gram diagonal off by {diagonal_error:.2e}; BLAS accumulation is not trustworthy here")
    gram = np.clip(gram, -1.0, 1.0)
    del matrix
    row_means = row_mean_cosines(tensors, ids, shape) if shape is not None else None
    rows = []
    for a, left in enumerate(ids):
        for b in range(a + 1, len(ids)):
            right = ids[b]
            inside = (left, right) in aligned
            rows.append({"layer_a": left, "layer_b": right, "offset": right - left,
                         "cosine": float(gram[a, b]),
                         "row_mean_cosine": (row_means[(left, right)] if row_means else None),
                         "both_inside_region": inside})
    spectrum = []
    for offset in range(1, depth):
        selected = [r for r in rows if r["offset"] == offset]
        if selected:
            spectrum.append({"offset": offset, "n_pairs": len(selected),
                             "mean_cosine": float(np.mean([r["cosine"] for r in selected])),
                             "mean_row_mean_cosine": (float(np.mean([r["row_mean_cosine"] for r in selected]))
                                                      if row_means else None),
                             "rank_eligible": len(selected) >= min_pairs})
    ranked = sorted((r for r in spectrum if r["rank_eligible"]),
                    key=lambda r: (-r["mean_cosine"], r["offset"]))
    by_period = {}
    for offset in (period, 2 * period):
        selected = [r for r in rows if r["offset"] == offset]
        inside = [r["cosine"] for r in selected if r["both_inside_region"]]
        outside = [r["cosine"] for r in selected if not r["both_inside_region"]]
        inside_rows = [r["row_mean_cosine"] for r in selected if r["both_inside_region"]] if row_means else []
        outside_rows = [r["row_mean_cosine"] for r in selected if not r["both_inside_region"]] if row_means else []
        by_period[str(offset)] = {
            "n_pairs": len(selected),
            "all_valid_mean": float(np.mean([r["cosine"] for r in selected])) if selected else None,
            "n_inside": len(inside), "inside_mean": float(np.mean(inside)) if inside else None,
            "inside_min": min(inside) if inside else None,
            "inside_max": max(inside) if inside else None,
            "inside_row_mean": float(np.mean(inside_rows)) if inside_rows else None,
            "outside_row_mean": float(np.mean(outside_rows)) if outside_rows else None,
            "n_outside": len(outside), "outside_mean": float(np.mean(outside)) if outside else None}
    return rows, spectrum, {"top_offsets": ranked[:4], "periods": by_period,
                             "segments": segments, "pair_groups": pair_groups,
                             "n_aligned_pairs_measured": sum(1 for r in rows if r["both_inside_region"]),
                             "global_pair_mean": float(np.mean([r["cosine"] for r in rows])),
                             "tensor_layers": ids}


def scan_model(spec: dict, cache: Path, provenance: Path, workers: int,
               expected_hashes: dict) -> tuple[list, list, list, dict]:
    repo, revision = spec["repo_id"], spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full pinned Hub commit SHA")
    metadata = {}
    single_file = "model.safetensors.index.json" not in spec["metadata_sha256"]
    (provenance / spec["slug"]).mkdir(parents=True, exist_ok=True)
    for filename in ("config.json",) if single_file else ("config.json", "model.safetensors.index.json"):
        path = Path(hf_hub_download(repo, filename, revision=revision, token=False))
        data = path.read_bytes()
        expected = spec["metadata_sha256"][filename]
        if digest(data) != expected:
            raise ValueError(f"Metadata hash mismatch: {repo}/{filename}")
        (provenance / spec["slug"] / filename).write_bytes(data)
        metadata[filename] = json.loads(data)
    config = metadata["config.json"]
    text = config.get("text_config", config)
    depth = int(text["num_hidden_layers"])
    if depth != spec["depth"]:
        raise ValueError("Pinned config depth mismatch")
    if single_file:
        # Checkpoints shipped as one model.safetensors have no index; the safetensors header is
        # the pinned metadata instead (hashed like the index), and every tensor maps to that file.
        url = hf_hub_url(repo, "model.safetensors", revision=revision)
        header_len = struct.unpack("<Q", read_range(url, 0, 7))[0]
        if header_len > 64 * 1024 * 1024:
            raise ValueError("Unreasonable safetensors header")
        raw_header = read_range(url, 8, 7 + header_len)
        if digest(raw_header) != spec["metadata_sha256"]["model.safetensors.header"]:
            raise ValueError(f"Metadata hash mismatch: {repo}/model.safetensors header")
        (provenance / spec["slug"] / "model.safetensors.header.json").write_bytes(raw_header)
        weight_map = {name: "model.safetensors" for name in json.loads(raw_header) if name != "__metadata__"}
    else:
        weight_map = metadata["model.safetensors.index.json"]["weight_map"]
    # Hybrid checkpoints declare their attention blocks either explicitly (layer_types) or by a
    # period (full_attention_interval: every interval-th block, 1-based, is full attention).
    layer_types = text.get("layer_types")
    if layer_types is None and text.get("full_attention_interval"):
        interval = int(text["full_attention_interval"])
        layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(depth)]
    groups = {suffix: decoder_names(weight_map, spec["decoder_prefix"], suffix, depth)
              for suffix in spec["tensors"]}
    all_names = sorted({name for names in groups.values() for name in names.values()})
    # Verify each selected family's block coverage against the configured layer types when available.
    if layer_types:
        for suffix, names in groups.items():
            wanted = expected_layer_kind(suffix)
            expected_ids = {i for i, kind in enumerate(layer_types) if wanted is None or kind == wanted}
            if set(names) != expected_ids:
                raise ValueError(f"Tensor coverage differs from config.layer_types: {suffix}")
    else:
        for suffix, names in groups.items():
            if set(names) != set(range(depth)):
                raise ValueError(f"Tensor family does not cover every decoder block: {suffix}")
    headers = {}

    def load_header(shard):
        if shard not in headers:
            url = hf_hub_url(repo, shard, revision=revision)
            header_len = struct.unpack("<Q", read_range(url, 0, 7))[0]
            if header_len > 64 * 1024 * 1024:
                raise ValueError("Unreasonable safetensors header")
            raw_header = read_range(url, 8, 7 + header_len)
            headers[shard] = (url, json.loads(raw_header), 8 + header_len, digest(raw_header))
        return headers[shard][1]

    for shard in sorted({weight_map[name] for name in all_names}):
        load_header(shard)
    resolved = resolve_shards(all_names, weight_map, load_header)
    index_mismatches = sorted(name for name in all_names if resolved[name] != weight_map[name])

    def fetch(name):
        shard = resolved[name]
        url, header, base, header_hash = headers[shard]
        entry = header[name]
        a, b = entry["data_offsets"]
        local = cache / spec["slug"] / revision / (digest(name.encode()) + ".bin")
        if local.exists():
            raw = local.read_bytes()
            if len(raw) != b - a:
                raise ValueError("Cache size mismatch; move the bad cache file aside and retry")
        else:
            raw = read_range(url, base + a, base + b - 1)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(raw)
        expected_hash = expected_hashes.get((repo, revision, name))
        if expected_hash and digest(raw) != expected_hash:
            raise ValueError(f"Tensor hash differs from the released capture: {name}")
        values = decode_tensor(raw, entry["dtype"], entry["shape"])
        manifest = {"tensor_name": name, "shard": shard, "dtype": entry["dtype"],
                    "shape": entry["shape"], "n_bytes": len(raw), "sha256": digest(raw),
                    "shard_header_sha256": header_hash,
                    "absolute_byte_start": base + a, "absolute_byte_end": base + b - 1}
        if shard != weight_map[name]:
            manifest["index_shard"] = weight_map[name]  # the Hub index named a different shard
        return name, values, manifest

    print(f"Reading {repo}@{revision[:12]}: {len(all_names)} decoder tensors"
          + (f" ({len(index_mismatches)} located outside their indexed shard)" if index_mismatches else ""), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = list(pool.map(fetch, all_names))
    values = {name: array for name, array, _ in fetched}
    shapes = {name: entry["shape"] for name, _, entry in fetched}
    manifests = [entry for _, _, entry in fetched]
    all_rows, all_offsets, summaries = [], [], []
    for suffix, names in groups.items():
        family_shapes = {tuple(shapes[name]) for name in names.values()}
        if len(family_shapes) != 1:
            raise ValueError(f"Tensor shapes differ across layers for {suffix}: {sorted(family_shapes)}")
        rows, spectrum, summary = similarities(
            {i: values[name] for i, name in names.items()}, depth, spec["period"],
            tuple(spec["region"]), spec["min_pairs"], shape=list(family_shapes.pop()),
            segments=spec.get("segments"), pair_groups=spec.get("pair_groups"))
        common = {"model": repo, "model_slug": spec["slug"], "revision": revision, "tensor": suffix}
        all_rows.extend(dict(common, **row) for row in rows)
        all_offsets.extend(dict(common, **row) for row in spectrum)
        summaries.append(dict(common, depth=depth, decoder_prefix=spec["decoder_prefix"],
                              period=spec["period"], region=spec["region"], role=spec.get("role"),
                              min_rank_pairs=spec["min_pairs"], **summary))
        lead = summary["top_offsets"][0] if summary["top_offsets"] else None
        if lead:
            print(f"  {suffix}: {len(names)}/{depth} blocks; peak {lead['offset']} "
                  f"cos={lead['mean_cosine']:.6f}, n={lead['n_pairs']}", flush=True)
        else:
            print(f"  {suffix}: {len(names)}/{depth} blocks; no offset reaches {spec['min_pairs']} pairs", flush=True)
    return all_rows, all_offsets, summaries, {"repo_id": repo, "revision": revision,
                                              "index_shard_mismatches": len(index_mismatches),
                                              "selected_tensors": manifests}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "models.lock.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--provenance", type=Path, default=ROOT / "provenance" / "metadata")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "tensors")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models", nargs="*", help="lock slugs to scan (default: all)")
    args = parser.parse_args()
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    expected_hashes = {}
    released = ROOT / "data" / "provenance.json"
    if released.exists() and released.resolve() != (args.out / "provenance.json").resolve():
        for model in json.loads(released.read_text())["models"]:
            for tensor in model["selected_tensors"]:
                expected_hashes[(model["repo_id"], model["revision"], tensor["tensor_name"])] = tensor["sha256"]
    specs = [s for s in lock["models"] if not args.models or s["slug"] in args.models]
    if not specs:
        raise SystemExit("No lock entries selected")
    rows, offsets, summaries, manifests = [], [], [], []
    started = time.monotonic()
    for spec in specs:
        a, b, c, d = scan_model(spec, args.cache, args.provenance, args.workers, expected_hashes)
        rows.extend(a); offsets.extend(b); summaries.extend(c); manifests.append(d)
    write_csv(args.out / "pairs.csv", rows)
    write_csv(args.out / "offsets.csv", offsets)
    save_json(args.out / "summary.json", {"schema_version": 2, "analyses": summaries})
    save_json(args.out / "provenance.json", {
        "schema_version": 2, "models_lock_sha256": digest(lock_bytes),
        "scanner_sha256": digest(Path(__file__).read_bytes()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "requests", "huggingface_hub")},
        "elapsed_seconds": round(time.monotonic() - started, 2), "models": manifests})
    print(f"Saved {len(rows)} layer pairs to {args.out}")


if __name__ == "__main__":
    main()
