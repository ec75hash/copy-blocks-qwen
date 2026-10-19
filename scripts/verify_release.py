#!/usr/bin/env python3
"""Verify the released files: checksums, lock and scanner hashes, tensor namespaces, block coverage,
and that RESULTS.md is exactly what make_report.py regenerates from data/. No network access."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    failures = []

    checksums = json.loads((root / "checksums.json").read_text())
    for name, expected in checksums.items():
        if not (root / name).is_file():
            failures.append(f"missing file listed in checksums.json: {name}")
        elif sha256(root / name) != expected:
            failures.append(f"checksum mismatch: {name}")

    lock = json.loads((root / "models.lock.json").read_text())
    provenance = json.loads((root / "data/provenance.json").read_text())
    summaries = json.loads((root / "data/summary.json").read_text())["analyses"]
    if provenance["models_lock_sha256"] != sha256(root / "models.lock.json"):
        failures.append("data/provenance.json was produced from a different models.lock.json")
    if provenance["scanner_sha256"] != sha256(root / "scripts/scan_weights.py"):
        failures.append("data/provenance.json was produced by a different scripts/scan_weights.py")

    specs = {m["repo_id"]: m for m in lock["models"]}
    if len(specs) != len(lock["models"]):
        failures.append("duplicate repo_id in models.lock.json")
    scanned_list = [m["repo_id"] for m in provenance["models"]]
    scanned = set(scanned_list)
    if len(scanned) != len(scanned_list):
        failures.append("duplicate model entry in data/provenance.json")
    if scanned != set(specs):
        failures.append(f"provenance roster ({len(scanned)}) differs from lock roster ({len(specs)}): "
                        f"missing {sorted(set(specs) - scanned)}, extra {sorted(scanned - set(specs))}")
    for model in provenance["models"]:
        names = [t["tensor_name"] for t in model["selected_tensors"]]
        if len(names) != len(set(names)):
            failures.append(f"duplicate tensor manifest entries for {model['repo_id']}")
    keys = [(a["model_slug"], a["tensor"]) for a in summaries]
    if len(keys) != len(set(keys)):
        failures.append("duplicate (model, tensor) analysis in data/summary.json")
    for model in provenance["models"]:
        spec = specs[model["repo_id"]]
        pattern = re.compile(r"^" + re.escape(spec["decoder_prefix"]) + r"\d+\.")
        for tensor in model["selected_tensors"]:
            if not pattern.match(tensor["tensor_name"]):
                failures.append(f"non-decoder tensor selected: {tensor['tensor_name']}")
            if not re.fullmatch(r"[0-9a-f]{64}", tensor["sha256"]):
                failures.append(f"missing payload hash: {tensor['tensor_name']}")

    # Block coverage per family must match the pinned configuration.
    by_slug = {}
    for a in summaries:
        by_slug.setdefault(a["model_slug"], {})[a["tensor"]] = a
    for spec in lock["models"]:
        config = json.loads((root / "provenance/metadata" / spec["slug"] / "config.json").read_text())
        text = config.get("text_config", config)
        depth = int(text["num_hidden_layers"])
        layer_types = text.get("layer_types")
        if layer_types is None and text.get("full_attention_interval"):
            k = int(text["full_attention_interval"])
            layer_types = ["full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(depth)]
        for suffix in spec["tensors"]:
            analysis = by_slug.get(spec["slug"], {}).get(suffix)
            if analysis is None:
                failures.append(f"no analysis for {spec['slug']} {suffix}")
                continue
            if analysis["depth"] != depth or depth != spec["depth"]:
                failures.append(f"depth mismatch for {spec['slug']}")
            if suffix.startswith("linear_attn."):
                expected = [i for i, t in enumerate(layer_types or []) if t == "linear_attention"]
            elif suffix.startswith("self_attn.") and layer_types:
                expected = [i for i, t in enumerate(layer_types) if t == "full_attention"]
            else:
                expected = list(range(depth))
            if analysis["tensor_layers"] != expected:
                failures.append(f"block coverage differs from config for {spec['slug']} {suffix}")

    # RESULTS.md and the figures' numbers must be reproducible from data/ with the released script.
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "RESULTS.md"
        subprocess.run([sys.executable, str(root / "scripts/make_report.py"), "--data", str(root / "data"),
                        "--lock", str(root / "models.lock.json"), "--out", tmp, "--report", str(report)],
                       check=True, capture_output=True)
        if report.read_text() != (root / "RESULTS.md").read_text():
            failures.append("RESULTS.md differs from make_report.py output")

    if failures:
        for failure in failures:
            print("FAIL", failure)
        raise SystemExit(1)
    print(f"OK: {len(checksums)} files verified, {len(specs)} checkpoints, {len(summaries)} family analyses, RESULTS.md reproducible")


if __name__ == "__main__":
    main()
