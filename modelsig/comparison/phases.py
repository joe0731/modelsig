"""3-phase isomorphism comparison (gemini)."""
from __future__ import annotations
from typing import Dict, List, Tuple

from ..signature.fingerprint import ModelFingerprint


def phase1_match(sig_a: dict, sig_b: dict, threshold: float = 0.80) -> Tuple[bool, List[str], List[str]]:
    ka, kb = set(sig_a), set(sig_b)
    only_a = sorted(ka - kb)
    only_b = sorted(kb - ka)
    common = ka & kb
    if not ka or not kb:
        return False, only_a, only_b
    coverage = len(common) / max(len(ka), len(kb))
    return coverage >= threshold, only_a, only_b


def phase2_substructure_match(tmpl_a: dict, tmpl_b: dict) -> bool:
    def suffixes(tmpl):
        return {s.split(".")[-2] if "." in s else s for s in tmpl}
    sa, sb = suffixes(tmpl_a), suffixes(tmpl_b)
    attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
    ffn  = {"gate_proj", "up_proj", "down_proj"}
    norm = {"input_layernorm", "post_attention_layernorm"}
    def has(s, g): return bool(s & g)
    return (has(sa, attn) == has(sb, attn) and
            has(sa, ffn)  == has(sb, ffn)  and
            has(sa, norm) == has(sb, norm))


def phase3_algebraic_check(fp_a: ModelFingerprint, fp_b: ModelFingerprint) -> Tuple[bool, dict]:
    cfa = fp_a.arch_fingerprint
    cfb = fp_b.arch_fingerprint
    scaling: dict = {}
    for dim in ["hidden_size", "intermediate_size", "head_dim"]:
        va, vb = cfa.get(dim), cfb.get(dim)
        if va and vb:
            scaling[dim] = {"a": va, "b": vb, "ratio": round(vb / va, 6)}
    consistent = True
    if len(scaling) >= 2:
        ratios = [v["ratio"] for v in scaling.values()]
        mean = sum(ratios) / len(ratios)
        if mean > 0 and any(abs(r - mean) / mean > 0.20 for r in ratios):
            consistent = False
    dr_a = fp_a.dimension_ratios
    dr_b = fp_b.dimension_ratios
    gqa_a = dr_a.get("gqa_ratio")
    gqa_b = dr_b.get("gqa_ratio")
    if gqa_a is not None and gqa_b is not None:
        if abs(gqa_a - gqa_b) > 0.5:
            consistent = False
    return consistent, scaling


def determine_isomorphism(fp_a: ModelFingerprint, fp_b: ModelFingerprint,
                           p1: bool, p2: bool, p3: bool) -> Tuple[str, str]:
    afa = fp_a.arch_fingerprint
    afb = fp_b.arch_fingerprint
    if afa.get("is_moe") != afb.get("is_moe"):
        return "DIFFERENT_ARCH", "One is MoE, the other Dense — incompatible for proxy testing."
    mt_a = afa.get("model_type")
    mt_b = afb.get("model_type")
    if mt_a and mt_b and mt_a != mt_b:
        return "DIFFERENT_ARCH", f"Different model types ({mt_a} vs {mt_b})."
    if not p1:
        return "DIFFERENT_ARCH", "Phase 1: key sets diverge significantly — likely different architectures."
    if not p2:
        return "DIFFERENT_ARCH", "Phase 2: core attention/FFN substructure mismatch."
    if p3:
        return "ISOMORPHIC", ("All 3 phases pass: same family, same operators, uniform dimension scaling. "
                               "Smaller model is a valid proxy.")
    return "SCALE_ONLY", ("Phase 1+2 pass but dimension scaling is non-uniform. "
                           "Same architectural family with irregular scaling.")
