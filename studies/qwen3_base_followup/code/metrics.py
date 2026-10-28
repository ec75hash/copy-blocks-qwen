"""Small, explicit weight-comparison statistics; no model inference."""
import numpy as np


def sampled_rows(seed=20261027):
    rng = np.random.default_rng(seed)
    chunks = np.r_[rng.choice(256, 8, replace=False),
                   256 + rng.choice(288, 9, replace=False)]
    return np.sort((chunks[:, None] * 32 + np.arange(32)).ravel())


def mapped_pairs():
    return [(s, s, "same_index") for s in range(32)] + [
        (s, s + 24, "plus24") for s in range(8, 40)]


def row_stats(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Compare equal two-dimensional row sets")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite weights")
    aa = np.einsum("ij,ij->i", a, a)
    bb = np.einsum("ij,ij->i", b, b)
    valid = (aa > 0) & (bb > 0)
    cos = np.full(len(a), np.nan)
    ua, ub = a[valid] / np.sqrt(aa[valid, None]), b[valid] / np.sqrt(bb[valid, None])
    for u in [ua, ub]:
        if len(u) and np.max(np.abs(np.einsum("ij,ij->i", u, u) - 1)) > 1e-9:
            raise ValueError("Normalized row diagonal off one")
    cos[valid] = np.clip(np.einsum("ij,ij->i", ua, ub), -1, 1)
    denominator = np.sqrt(aa.sum()) * np.sqrt(bb.sum())
    flat = float(np.einsum("ij,ij->", a, b) / denominator) if denominator else None
    return {"flat_cosine": flat, "row_mean": float(np.mean(cos[valid])) if valid.any() else None,
            "row_median": float(np.median(cos[valid])) if valid.any() else None,
            "valid_rows": int(valid.sum()), "total_rows": len(a)}, cos


def cross_cosines(a, b, normalized=False):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Equal vector dimensions required")
    units = []
    for x in [a, b]:
        norms = np.sqrt(np.einsum("ij,ij->i", x, x))
        if not np.isfinite(x).all() or (norms == 0).any():
            raise ValueError("Zero or nonfinite tensor")
        u = x if normalized else x / norms[:, None]
        if np.max(np.abs(np.einsum("ij,ij->i", u, u) - 1)) > 1e-9:
            raise ValueError("Normalized diagonal off one")
        units.append(u)
    return np.clip(units[0] @ units[1].T, -1, 1)
