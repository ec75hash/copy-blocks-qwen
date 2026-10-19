#!/usr/bin/env python3
"""Regenerate the family figure, offset spectra, heat maps, and RESULTS.md from the scanned CSV/JSON.

Every number drawn or printed here comes from data/pairs.csv, data/offsets.csv and data/summary.json;
the only inputs that are not measurements are the display order and the block hypothesis
(period and region) recorded for each checkpoint in models.lock.json, and the hypothesis is
drawn solid only where the measurement supports it (see CONFIRM_* below).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Reporting rules (not significance tests): a proposed block is "confirmed" when the top-ranked
# offset for the family is the lock period or twice it, the within-block mean at the period is at
# least CONFIRM_INSIDE, and it exceeds the same-offset off-block mean by at least CONFIRM_EXCESS
# (routers in every Qwen MoE share a 0.25-0.35 baseline between any two layers, so an absolute
# off-block ceiling would be the wrong test for them). A no-block row holds when no offset's mean
# stands more than NO_BLOCK_SPIKE above the mean of its two neighbouring offsets.
CONFIRM_INSIDE = 0.10
CONFIRM_EXCESS = 0.10
NO_BLOCK_SPIKE = 0.05

COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100",
       "track": "#f0efec", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "bar": "#9ec5f4", "bar_hi": "#1c5cab"}
ROLE_LABEL = {"A": "block A", "B": "block B (corresponds to A)", "C": "block C (corresponds to A and B)", "D": "block D"}

# Display order and group headers (presentation only).
DISPLAY = [
    ("Qwen1.5 / Qwen2 MoE", ["qwen15-moe-a2p7b", "qwen2-57b-a14b"]),
    ("Qwen2.5 and its descendants", ["qwen25-7b", "qwen25-math-7b", "qwen25-14b", "qwen25-32b", "r1-distill-qwen-32b", "kat-v1-40b"]),
    ("Qwen3", ["qwen3-0p6b", "qwen3-1p7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-30b-a3b",
               "qwen3-30b-a3b-2507", "qwen3-32b", "qwen3-next-80b-a3b", "qwen3-235b-a22b", "qwen3-coder-480b-a35b"]),
    ("Qwen3.5 / Qwen3.6", ["qwen35-0p8b", "qwen35-2b", "qwen35-4b", "qwen35-9b", "qwen35-27b", "qwen36-27b",
                           "qwen35-35b-a3b-base", "qwen35-35b-a3b", "qwen36-35b-a3b", "qwen35-122b-a10b", "qwen35-397b-a17b"]),
    ("Qwen3.8", ["qwen38-27b", "qwen38-flash-next", "qwen38-2p4t-a95b"]),
    ("Calibration: disclosed depth up-scaling", ["solar-10p7b"]),
]
LABEL = {
    "qwen15-moe-a2p7b": "Qwen1.5-MoE-A2.7B", "qwen2-57b-a14b": "Qwen2-57B-A14B",
    "qwen25-7b": "Qwen2.5-7B", "qwen25-math-7b": "Qwen2.5-Math-7B", "qwen25-14b": "Qwen2.5-14B", "qwen25-32b": "Qwen2.5-32B",
    "qwen35-35b-a3b-base": "Qwen3.5-35B-A3B-Base",
    "r1-distill-qwen-32b": "DeepSeek-R1-Distill-Qwen-32B", "kat-v1-40b": "KAT-V1-40B (grown from Qwen2.5-32B)",
    "qwen3-0p6b": "Qwen3-0.6B", "qwen3-1p7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
    "qwen3-14b": "Qwen3-14B", "qwen3-30b-a3b": "Qwen3-30B-A3B", "qwen3-30b-a3b-2507": "Qwen3-30B-A3B-Instruct-2507",
    "qwen3-32b": "Qwen3-32B", "qwen3-next-80b-a3b": "Qwen3-Next-80B-A3B", "qwen3-235b-a22b": "Qwen3-235B-A22B",
    "qwen3-coder-480b-a35b": "Qwen3-Coder-480B-A35B", "qwen35-0p8b": "Qwen3.5-0.8B", "qwen35-2b": "Qwen3.5-2B",
    "qwen35-4b": "Qwen3.5-4B", "qwen35-9b": "Qwen3.5-9B", "qwen35-27b": "Qwen3.5-27B", "qwen36-27b": "Qwen3.6-27B",
    "qwen35-35b-a3b": "Qwen3.5-35B-A3B", "qwen36-35b-a3b": "Qwen3.6-35B-A3B", "qwen35-122b-a10b": "Qwen3.5-122B-A10B",
    "qwen35-397b-a17b": "Qwen3.5-397B-A17B", "qwen38-27b": "Qwen3.8-27B", "qwen38-flash-next": "Qwen3.8-Flash-Next",
    "qwen38-2p4t-a95b": "Qwen3.8-2.4T-A95B", "solar-10p7b": "SOLAR-10.7B-v1.0",
}
FAMILY_SHORT = {"self_attn.k_proj.weight": "k_proj", "linear_attn.conv1d.weight": "conv1d", "mlp.gate.weight": "router"}


def load(data_dirs):
    analyses, pairs, offsets = [], [], []
    for d in data_dirs:
        analyses += json.loads((d / "summary.json").read_text())["analyses"]
        pairs += list(csv.DictReader((d / "pairs.csv").open()))
        offsets += list(csv.DictReader((d / "offsets.csv").open()))
    return analyses, pairs, offsets


def fnum(value):
    return None if value in (None, "", "None") else float(value)


def proposes_block(spec):
    return spec["role"] not in ("negative_control", "no_block_reported")


def block_segments(spec):
    """(start, end_inclusive, role) for each block in the lock's hypothesis.

    Explicit `segments` + `pair_groups` (KAT: two independent pairs) or a single chain derived from
    (region, period). The role is the block's position within its group: A, B, C, ... The weights
    give no direction, so no block is called the source."""
    roles = ["A", "B", "C", "D"]
    if spec.get("segments"):
        segments = [tuple(s) for s in spec["segments"]]
        groups = spec.get("pair_groups") or [list(range(len(segments)))]
    else:
        start, end = spec["region"]
        period = spec["period"]
        segments, i = [], 0
        while start + (i + 1) * period - 1 <= end and i < len(roles):
            segments.append((start + i * period, start + (i + 1) * period - 1))
            i += 1
        groups = [list(range(len(segments)))]
    out = []
    for group in groups:
        for position, index in enumerate(group):
            s = segments[index]
            out.append((s[0], s[1], roles[position]))
    return out


def family_verdict(spec, analysis, offsets):
    """Per (model, family): peak offset, within/off-block means at the period, spike, verdict."""
    period = spec["period"]
    top = analysis["top_offsets"][0] if analysis["top_offsets"] else None
    at_k = analysis["periods"][str(period)]
    eligible = [o for o in offsets if o["model_slug"] == spec["slug"] and o["tensor"] == analysis["tensor"]
                and o["rank_eligible"] == "True"]
    means = {int(o["offset"]): fnum(o["mean_cosine"]) for o in eligible}
    max_any = max(means.values(), default=None)
    # Spike: how far an offset's mean stands above its two neighbours. Adjacent-layer drift
    # (routers sit at 0.3-0.5 between neighbouring layers in every Qwen MoE) decays smoothly and
    # produces no spike; a copied block produces one at K.
    spikes = {k: v - 0.5 * (means[k - 1] + means[k + 1]) for k, v in means.items()
              if k - 1 in means and k + 1 in means}
    max_spike = max(spikes.values(), default=None)
    inside, outside = at_k.get("inside_mean"), at_k.get("outside_mean")
    excess = None if inside is None else inside - (outside if outside is not None else 0.0)
    if proposes_block(spec):
        confirmed = (top is not None and top["offset"] in (period, 2 * period)
                     and inside is not None and inside >= CONFIRM_INSIDE
                     and excess is not None and excess >= CONFIRM_EXCESS)
        verdict = "confirmed" if confirmed else "not confirmed"
    elif max_any is None:
        verdict = "no eligible offsets"
    elif max_spike is not None:
        verdict = "no block" if max_spike < NO_BLOCK_SPIKE else "unexpected peak"
    else:
        verdict = "no block" if max_any < NO_BLOCK_SPIKE else "unexpected peak"
    return {"period": period, "peak": top["offset"] if top else None,
            "peak_mean": top["mean_cosine"] if top else None, "inside": inside, "outside": outside,
            "excess": excess, "inside_row_mean": at_k.get("inside_row_mean"), "max_any": max_any,
            "max_spike": max_spike, "n_inside": at_k.get("n_inside"), "verdict": verdict}


def model_summary(spec, analyses, offsets):
    fams = {a["tensor"]: family_verdict(spec, a, offsets) for a in analyses if a["model_slug"] == spec["slug"]}
    return fams


def draw_family(specs, summaries, out):
    rows = []
    for header, slugs in DISPLAY:
        present = [s for s in slugs if s in specs]
        if present:
            rows.append(("header", header))
            rows.extend(("model", s) for s in present)
    height = 0.42 * len(rows) + 2.2
    fig, ax = plt.subplots(figsize=(16.5, height))
    max_depth = max(spec["depth"] for spec in specs.values())
    y = 0
    for kind, value in reversed(rows):
        if kind == "header":
            ax.text(-1.0, y, value, ha="right", va="center", fontsize=10, fontweight="bold", color=COL["ink"])
            y += 1
            continue
        spec, fams = specs[value], summaries[value]
        depth = spec["depth"]
        ax.barh(y, depth, left=0, height=0.62, color=COL["track"], edgecolor="none", zorder=1)
        verdicts = {f: v["verdict"] for f, v in fams.items()}
        if proposes_block(spec):
            confirmed = any(v == "confirmed" for v in verdicts.values())
            for start, end, role in block_segments(spec):
                ax.barh(y, end - start + 1 - 0.18, left=start + 0.09, height=0.62,
                        color=COL[role] if confirmed else "none", edgecolor=COL[role] if not confirmed else "none",
                        hatch=None if confirmed else "////", linewidth=1.0, zorder=2)
                ax.text(start + (end - start + 1) / 2, y, f"{start}–{end}", ha="center", va="center",
                        fontsize=7.5, color="#ffffff" if confirmed else COL["ink2"], zorder=3)
        ax.text(-1.0, y, LABEL.get(value, value), ha="right", va="center", fontsize=9.5, color=COL["ink"])
        # Measured annotation: within-block mean at the period for each family, plus the same-offset off-block mean.
        bits = []
        for fam in ("self_attn.k_proj.weight", "linear_attn.conv1d.weight", "mlp.gate.weight"):
            if fam not in fams:
                continue
            v = fams[fam]
            if proposes_block(spec):
                val = v["inside_row_mean"] if (fam == "mlp.gate.weight" and v["inside_row_mean"] is not None) else v["inside"]
                if val is not None:
                    bits.append(f"{FAMILY_SHORT[fam]}{' rows' if fam == 'mlp.gate.weight' else ''} {val:.2f}")
            else:
                if v["max_spike"] is not None:
                    bits.append(f"{FAMILY_SHORT[fam]} spike {v['max_spike']:.3f}")
                elif v["max_any"] is not None:
                    bits.append(f"{FAMILY_SHORT[fam]} max {v['max_any']:.3f}")
        if proposes_block(spec):
            outs = [v["outside"] for v in fams.values() if v["outside"] is not None]
            if outs:
                bits.append(f"off-block at K {(0.0 if abs(max(outs)) < 0.005 else max(outs)):.2f}")
            k = spec["period"]
            lead = f"K={k}" + ("" if all(v == "confirmed" for v in verdicts.values()) else " (not all families confirm)")
            text = f"{depth} layers · {lead} · " + " · ".join(bits)
        else:
            text = f"{depth} layers · " + " · ".join(bits)
        ax.text(depth + 1.0, y, text, ha="left", va="center", fontsize=7.9, color=COL["ink2"])
        y += 1
    ax.set_xlim(0, max_depth + 66)
    ax.set_ylim(-0.8, y - 0.2)
    ax.set_yticks([])
    ax.set_xticks(range(0, max_depth + 1, 8))
    ax.set_xlabel("Decoder block (zero-based)", color=COL["ink2"])
    ax.tick_params(colors=COL["muted"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COL["grid"])
    ax.grid(axis="x", color=COL["grid"], linewidth=0.6, zorder=0)
    ax.legend(handles=[Patch(color=COL["A"], label=ROLE_LABEL["A"]), Patch(color=COL["B"], label=ROLE_LABEL["B"]),
                       Patch(color=COL["C"], label=ROLE_LABEL["C"]),
                       Patch(facecolor="none", edgecolor=COL["ink2"], hatch="////", label="proposed block, not confirmed by this scan"),
                       Patch(color=COL["track"], label="no repeated block found")],
              loc="upper right", frameon=False, fontsize=8.5)
    ax.set_title("Repeated (depth-upscaled) blocks across Qwen-family checkpoints, from pinned weight scans",
                 loc="left", fontsize=12.5, color=COL["ink"])
    fig.text(0.01, 0.005,
             "Segments are the lock's block hypothesis, drawn solid only where this scan's top-ranked offset is K or 2K, the within-block mean at K is ≥ "
             f"{CONFIRM_INSIDE:.2f}, and it exceeds the same-offset off-block mean by ≥ {CONFIRM_EXCESS:.2f}.\n"
             "Numbers: within-block mean cosine at offset K (flattened weights; router rows = mean over aligned expert rows) and the off-block mean at the same offset. "
             f"No-block rows show the largest spike (an offset's mean minus the mean of its two neighbours; < {NO_BLOCK_SPIKE:.2f} means no repetition).\n"
             "Source: data/pairs.csv, data/offsets.csv, data/summary.json; checkpoints pinned in models.lock.json.",
             fontsize=7.6, color=COL["ink2"], va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out / f"family_blocks.{ext}", dpi=170)
    plt.close(fig)


def draw_spectra(specs, analyses, offsets, out):
    items = [(a["model_slug"], a["tensor"]) for a in analyses if proposes_block(specs[a["model_slug"]])]
    cols = 4
    n = len(items)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, 2.6 * rows_n), squeeze=False)
    for ax, (slug, tensor) in zip(axes.flat, items):
        spec = specs[slug]
        period = spec["period"]
        sel = [o for o in offsets if o["model_slug"] == slug and o["tensor"] == tensor and o["rank_eligible"] == "True"]
        xs = [int(o["offset"]) for o in sel]
        ys = [fnum(o["mean_cosine"]) for o in sel]
        if not xs:
            ax.text(0.5, 0.5, "no offset reaches 6 pairs", transform=ax.transAxes, ha="center", color=COL["muted"], fontsize=8)
            ax.set_title(f"{LABEL.get(slug, slug)} · {FAMILY_SHORT.get(tensor, tensor)}", fontsize=8.5, loc="left", color=COL["ink"])
            continue
        colors = [COL["bar_hi"] if x % period == 0 else COL["bar"] for x in xs]
        ax.bar(xs, ys, color=colors, width=0.85, edgecolor="none")
        for x, yv in zip(xs, ys):
            if x % period == 0 and yv >= 0.05:
                ax.text(x, yv, f"{yv:.2f}", ha="center", va="bottom", fontsize=7, color=COL["ink2"])
        ax.set_title(f"{LABEL.get(slug, slug)} · {FAMILY_SHORT.get(tensor, tensor)}", fontsize=8.5, loc="left", color=COL["ink"])
        ax.set_ylim(min(-0.05, min(ys) - 0.02), max(0.2, max(ys) * 1.25))
        ax.set_xlim(0, spec["depth"])
        ax.tick_params(labelsize=7, colors=COL["muted"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(COL["grid"]); ax.spines["bottom"].set_color(COL["grid"])
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Mean cosine by layer offset (offsets with ≥ 6 pairs); dark bars are multiples of the lock period K",
                 x=0.01, ha="left", fontsize=11, color=COL["ink"])
    fig.supxlabel("Layer offset", fontsize=9, color=COL["ink2"]); fig.supylabel("Mean cosine", fontsize=9, color=COL["ink2"])
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    for ext in ("png", "svg"):
        fig.savefig(out / f"offset_spectra.{ext}", dpi=150)
    plt.close(fig)


def draw_heatmaps(specs, analyses, pairs, out, panels):
    lookup = {(a["model_slug"], a["tensor"]): a for a in analyses}
    panels = [(s, t, title) for s, t, title in panels if (s, t) in lookup]
    cols = 3
    rows_n = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, 4.6 * rows_n), squeeze=False, layout="constrained")
    cmap = plt.colormaps["Blues"].copy(); cmap.set_bad("#eeeeee")
    plot = None
    for ax, (slug, tensor, title) in zip(axes.flat, panels):
        a, spec = lookup[(slug, tensor)], specs[slug]
        matrix = np.full((a["depth"], a["depth"]), np.nan)
        for row in pairs:
            if row["model_slug"] == slug and row["tensor"] == tensor:
                i, j = int(row["layer_a"]), int(row["layer_b"])
                matrix[i, j] = matrix[j, i] = float(row["cosine"])
        plot = ax.imshow(matrix, vmin=-0.05, vmax=1, cmap=cmap, origin="lower", interpolation="nearest")
        if proposes_block(spec):
            lo, hi = spec["region"]
            for boundary in range(lo, hi + 2, spec["period"]):
                ax.axhline(boundary - 0.5, color="#e87926", linewidth=0.7, alpha=0.85)
                ax.axvline(boundary - 0.5, color="#e87926", linewidth=0.7, alpha=0.85)
        ax.set(title=title, xlabel="Decoder block", ylabel="Decoder block")
        ax.set_xticks(range(0, a["depth"], 8)); ax.set_yticks(range(0, a["depth"], 8))
        ax.tick_params(labelsize=7, colors=COL["muted"])
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    if plot is not None:
        fig.colorbar(plot, ax=axes, shrink=0.6, label="Flattened weight cosine")
    fig.suptitle("Layer-pair weight similarity; orange lines are the lock's block boundaries", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(out / f"weight_similarity.{ext}", dpi=150)
    plt.close(fig)


def write_results(specs, summaries, analyses, report):
    lines = ["# Numerical results", "",
             "Regenerated from `data/summary.json`, `data/offsets.csv` and `data/pairs.csv` by `scripts/make_report.py`.", "",
             "Cosines use flattened decoder-only tensors; the row-mean column is the mean over aligned rows (experts, for a router). "
             "Within-block means use the aligned pairs between corresponding blocks of the lock's hypothesis; off-block means use the remaining pairs at the same offset. "
             f"A proposed block counts as confirmed when the top-ranked offset is K or 2K, the within-block mean at K is ≥ {CONFIRM_INSIDE:.2f} "
             f"and exceeds the off-block mean at K by ≥ {CONFIRM_EXCESS:.2f}. A no-block row holds when no offset's mean stands more than {NO_BLOCK_SPIKE:.2f} "
             "above the mean of its two neighbouring offsets (the spike column); routers in every Qwen MoE share a smooth 0.3–0.5 similarity between neighbouring layers, "
             "which is drift, not repetition. These are reporting rules, not significance tests.", "",
             "| Checkpoint | Role | Tensor | Blocks with tensor | K | Top offset (mean, pairs) | Within-block mean at K (pairs) | Off-block mean at K | Row-mean at K | Max spike | Verdict |",
             "|---|---|---|---:|---:|---|---|---:|---:|---:|---|"]
    for header, slugs in DISPLAY:
        for slug in slugs:
            if slug not in specs:
                continue
            spec = specs[slug]
            for a in [x for x in analyses if x["model_slug"] == slug]:
                v = summaries[slug][a["tensor"]]
                def f(x, nd=3):
                    return "—" if x is None else f"{x:.{nd}f}"
                top = "—" if v["peak"] is None else f"{v['peak']} ({v['peak_mean']:.3f}, n={a['top_offsets'][0]['n_pairs']})"
                lines.append(f"| {LABEL.get(slug, slug)} | {spec['role']} | `{a['tensor']}` | {len(a['tensor_layers'])}/{a['depth']} | {v['period']} | {top} | "
                             f"{f(v['inside'])} ({v['n_inside'] or 0}) | {f(v['outside'])} | {f(v['inside_row_mean'])} | {f(v['max_spike'])} | **{v['verdict']}** |")
    lines += ["", "## Strongest offsets per family", ""]
    for header, slugs in DISPLAY:
        for slug in slugs:
            for a in [x for x in analyses if x["model_slug"] == slug]:
                tops = "; ".join(f"k={t['offset']}: {t['mean_cosine']:.4f} (n={t['n_pairs']})" for t in a["top_offsets"])
                lines.append(f"- **{LABEL.get(slug, slug)}**, `{a['tensor']}`: {tops}")
    report.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=[ROOT / "data"])
    parser.add_argument("--lock", type=Path, default=ROOT / "models.lock.json")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    parser.add_argument("--report", type=Path, default=ROOT / "RESULTS.md")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "savefig.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "hatch.linewidth": 0.8})
    analyses, pairs, offsets = load(args.data)
    lock = json.loads(args.lock.read_text())
    scanned = {a["model_slug"] for a in analyses}
    specs = {m["slug"]: m for m in lock["models"] if m["slug"] in scanned}
    summaries = {slug: model_summary(spec, analyses, offsets) for slug, spec in specs.items()}
    draw_family(specs, summaries, args.out)
    draw_spectra(specs, analyses, offsets, args.out)
    draw_heatmaps(specs, analyses, pairs, args.out, [
        ("qwen3-32b", "self_attn.k_proj.weight", "Qwen3-32B · attention key projection"),
        ("qwen36-35b-a3b", "mlp.gate.weight", "Qwen3.6-35B-A3B · MoE router"),
        ("qwen25-32b", "self_attn.k_proj.weight", "Qwen2.5-32B · attention key projection"),
        ("solar-10p7b", "self_attn.k_proj.weight", "SOLAR-10.7B · disclosed duplication (calibration)"),
        ("qwen25-7b", "self_attn.k_proj.weight", "Qwen2.5-7B · no block (control)"),
        ("qwen35-9b", "linear_attn.conv1d.weight", "Qwen3.5-9B · no block (control)"),
    ])
    write_results(specs, summaries, analyses, args.report)
    print(f"Wrote figures to {args.out} and {args.report} for {len(specs)} checkpoints")


if __name__ == "__main__":
    main()
