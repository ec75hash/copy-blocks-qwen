"""Verify and plot the released Qwen2.5 cross-checkpoint measurements offline.

This reduces retained matrices/tables; remeasuring model bytes is a separate step.
Generated output stays outside the released study.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
GROUPS = {"0-15 -> 0-15": [(s, s) for s in range(16)],
          "16-31 -> 16-31 (twin A)": [(s, s) for s in range(16, 32)],
          "16-31 -> 32-47 (twin B)": [(s, s + 16) for s in range(16, 32)],
          "32-47 -> 48-63": [(s, s + 16) for s in range(32, 48)]}


def rows(name):
    with (STUDY / "analysis" / name).open() as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    assert abs(float(a) - float(b)) < 1e-10, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reproduction/qwen25_weights_readout")
    args = parser.parse_args()
    out = args.out.resolve()
    if out == STUDY or STUDY in out.parents:
        raise ValueError("Choose an output directory outside the released study")
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((STUDY / "data/cross_qwen25_14b_32b.json").read_text())
    provenance = json.loads((STUDY / "provenance/tensors.json").read_text())
    payloads = {}
    for model in provenance["models"]:
        for tensor in model["selected_tensors"]:
            payloads[model["slug"] + "/" + tensor["tensor_name"]] = tensor["sha256"]
    assert payloads == source["payload_sha256"] and len(payloads) == 672
    assert sum(t["n_bytes"] for m in provenance["models"] for t in m["selected_tensors"]) == 1178304512
    pairs = rows("cross_checkpoint_pairs.csv")
    assert len(pairs) == 18432
    pair_lookup = {(r["family"], int(r["source_layer"]), int(r["target_layer"])): r for r in pairs}
    assert len(pair_lookup) == len(pairs)
    map_rows = {(r["family"], r["map_group"]): r for r in rows("mapping_summary.csv")}
    family_rows = {r["family"]: r for r in rows("family_summary.csv")}
    residual_rows = {(r["family"], int(r["reference_layer"])): r for r in rows("endpoint_residuals.csv")}
    assert len(map_rows) == 24 and len(family_rows) == 6 and len(residual_rows) == 96
    result = {"families": {}, "cross_pair_rows_checked": len(pairs), "mapped_summary_rows_checked": 24,
              "endpoint_residual_rows_checked": 96, "payload_manifest_entries_checked": 672,
              "scope": "Offline matrix/table reduction; no raw tensor remeasurement or inference"}
    lines = ["# Qwen2.5-14B / 32B measured correspondence", "",
             "Generated from the six retained 48×64 cosine matrices and corrected source JSON.", "",
             "All indices are zero-based decoder indices. Norm rows subtract the stored all-ones initialization.", "",
             "| Family | 14B → 32B layer group | Pairs | Mean cosine | Min | Max |",
             "|---|---|---:|---:|---:|---:|"]
    with np.load(STUDY / "data/cross_cosines.npz") as archive:
        assert set(archive.files) == set(source["families"])
        matrices = {family: archive[family] for family in archive.files}
    for family, matrix in matrices.items():
        assert matrix.shape == (48, 64) and np.isfinite(matrix).all()
        assert abs(matrix).max() <= 1 + 1e-10
        expected = source["families"][family]
        best = matrix.argmax(axis=1)
        assert best.tolist() == [r[0] for r in expected["best_match"]]
        mask = np.zeros(matrix.shape, dtype=bool)
        groups = {}
        for label, indices in GROUPS.items():
            values = np.array([matrix[s, t] for s, t in indices])
            for s, t in indices:
                mask[s, t] = True
            mean, low, high = float(values.mean()), float(values.min()), float(values.max())
            close(mean, expected["mapping_means"][label])
            record = map_rows[family, label]
            assert int(record["n_pairs"]) == 16
            for measured, field in [(mean, "mean_cosine"), (low, "min_cosine"), (high, "max_cosine")]:
                close(measured, record[field])
            ranks = [int(1 + np.sum(matrix[s] > matrix[s, t])) for s, t in indices]
            assert (min(ranks), max(ranks)) == (int(record["best_rank"]), int(record["worst_rank"]))
            groups[label] = {"mean": mean, "min": low, "max": high}
            lines.append(f"| `{family}` | {label} | 16 | {mean:.6f} | {low:.6f} | {high:.6f} |")
        assert mask.sum() == 64
        off_max, off_mean = float(abs(matrix[~mask]).max()), float(matrix[~mask].mean())
        close(off_max, expected["off_mapping_max_abs_cosine"])
        close(off_max, family_rows[family]["off_map_max_abs_cosine"])
        close(off_mean, family_rows[family]["off_map_mean_cosine"])
        assert int(family_rows[family]["off_map_pairs"]) == 3008
        for s in range(48):
            for t in range(64):
                r = pair_lookup[family, s, t]
                close(matrix[s, t], r["cosine"])
                assert bool(r["map_group"]) == bool(mask[s, t])
                assert r["source_revision"] == source["small"]["revision"]
                assert r["target_revision"] == source["large"]["revision"]
        for record in expected["drift"]:
            got = residual_rows[family, record["parent_layer"]]
            close(record["rel_drift_a"], got["relative_distance_a"])
            close(record["rel_drift_b"], got["relative_distance_b"])
            close(record["drift_cosine_a_b"], got["endpoint_residual_cosine"])
            close(record["twin_cosine"], got["twin_cosine"])
        result["families"][family] = {"mapping": groups, "off_map_max_abs": off_max, "off_map_mean": off_mean,
                                      "best_32b_layer_for_each_14b_layer": best.tolist()}
    lines += ["", "## Off-map background", "",
              "The 64 specified mapped pairs are excluded explicitly; 3,008 pairs remain per family.", "",
              "| Family | Off-map max absolute cosine | Off-map signed mean |", "|---|---:|---:|"]
    for family, measured in result["families"].items():
        lines.append(f"| `{family}` | {measured['off_map_max_abs']:.6f} | {measured['off_map_mean']:.6f} |")
    lines += ["", "High cross-layer norm similarity is common; the key and layer-specific bias maps locate the correspondence.",
              "Endpoint residuals are in `analysis/endpoint_residuals.csv`; these differences from the 14B reference do not date training updates.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    result["status"] = "passed"
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for ax, family, title, lo, hi in [
            (axes[0], "self_attn.k_proj.weight", "Attention key weights", -0.01, 0.85),
            (axes[1], "self_attn.q_proj.bias", "Query biases", -0.2, 1.0)]:
        im = ax.imshow(matrices[family], origin="upper", interpolation="nearest", cmap="viridis", vmin=lo, vmax=hi, aspect="auto")
        for x, y in [(0, 0), (16, 16), (32, 16), (48, 32)]:
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 16, 16, fill=False, edgecolor="#efad55", linewidth=1.1))
        ax.set(title=title, xlabel="Qwen2.5-32B decoder layer", ylabel="Qwen2.5-14B decoder layer")
        ax.set_xticks([0, 16, 32, 48, 63]); ax.set_yticks([0, 16, 32, 47])
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, label="Flattened cosine")
    fig.suptitle("One 14B block matches two consecutive 32B blocks", fontsize=13)
    fig.savefig(out / "cross_checkpoint.png", dpi=160)
    fig.savefig(out / "cross_checkpoint.svg", metadata={"Date": None})
    plt.close(fig)
    print(f"PASS: 18,432 pair values, 24 map means, 96 residual rows, 672 manifest entries; output {out}")


if __name__ == "__main__":
    main()
