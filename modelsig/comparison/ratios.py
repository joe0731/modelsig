"""Shape ratio uniformity analysis (grok)."""
from __future__ import annotations
from typing import List
from ..signature.static import param_count


def _is_uniform(ratios: List[float], tol: float = 0.05) -> bool:
    if not ratios:
        return True
    mx = max(abs(r) for r in ratios) or 1.0
    return (max(ratios) - min(ratios)) <= tol * mx


def analyze_shape_ratios(sig_a: dict, sig_b: dict) -> dict:
    result: dict = {}
    for key in sorted(set(sig_a) & set(sig_b)):
        sa = sig_a[key].get("representative_shape", [])
        sb = sig_b[key].get("representative_shape", [])
        if not sa or not sb or len(sa) != len(sb):
            continue
        if any(d == 0 for d in sa) or any(d == 0 for d in sb):
            continue
        vol_a = param_count(sa)
        vol_b = param_count(sb)
        small, large = (sa, sb) if vol_a <= vol_b else (sb, sa)
        ratios = [large[i] / small[i] if small[i] else float("inf") for i in range(len(small))]
        result[key] = {
            "small_shape": small, "large_shape": large,
            "ratios": [round(r, 4) for r in ratios],
            "uniform": _is_uniform(ratios),
        }
    return result
