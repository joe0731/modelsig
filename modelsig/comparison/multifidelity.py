"""Multi-fidelity test plan (gemini)."""
from __future__ import annotations
from typing import Dict
from ..signature.fingerprint import ModelFingerprint


def _size_score(fp: ModelFingerprint) -> int:
    layers = fp.arch_fingerprint.get("num_hidden_layers") or 0
    hidden = fp.arch_fingerprint.get("hidden_size") or 0
    score = layers * hidden
    if fp.arch_fingerprint.get("is_moe"):
        score *= 4
    return score


def build_multi_fidelity_plan(fingerprints: Dict[str, ModelFingerprint],
                               coverage_matrix: dict) -> dict:
    scored = sorted(fingerprints.keys(), key=lambda m: _size_score(fingerprints[m]))
    iso_groups: dict = {}
    seen: set = set()
    for mid in scored:
        if mid in seen:
            continue
        group = [mid]
        seen.add(mid)
        for other in scored:
            if other in seen:
                continue
            k1 = f"{mid}|{other}"
            k2 = f"{other}|{mid}"
            cov = coverage_matrix.get(k1) or coverage_matrix.get(k2) or {}
            if cov.get("isomorphism") in ("ISOMORPHIC", "SCALE_ONLY"):
                group.append(other)
                seen.add(other)
        iso_groups[mid] = group

    level1, level2, level3, level4 = [], [], [], []
    for head, members in iso_groups.items():
        gs = sorted(members, key=lambda m: _size_score(fingerprints[m]))
        smallest, largest = gs[0], gs[-1]
        mid_m = gs[len(gs) // 2]
        large_fp = fingerprints[largest]
        level1.append({"model": smallest, "reason": "Smallest in isomorphism group.",
                        "test_focus": ["model loading", "tensor shape validation", "dtype check"]})
        if len(members) > 1:
            level2.append({"model": mid_m, "reason": "Mid-size for numerical validation.",
                            "test_focus": ["layerwise cosine similarity", "perplexity on calibration set"]})
        level3.append({"model": mid_m if len(members) > 1 else smallest,
                        "reason": "Runtime: prefill / decode / KV cache modes.",
                        "test_focus": ["prefill latency", "decode throughput", "KV cache eviction"]})
        level4.append({"model": largest,
                        "reason": "Large/MoE canary: memory, TP/PP, routing correctness.",
                        "test_focus": (["MoE routing correctness", "peak memory"] if large_fp.arch_fingerprint.get("is_moe")
                                       else ["multi-GPU tensor parallelism", "peak memory"])})
    return {
        "level1_structure": level1,
        "level2_numerical": level2,
        "level3_runtime": level3,
        "level4_canary": level4,
    }
