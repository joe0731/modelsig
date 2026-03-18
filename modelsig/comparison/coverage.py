"""Unified coverage analysis combining structural phases and quant transferability."""
from __future__ import annotations
from typing import Dict
from ..signature.fingerprint import ModelFingerprint
from .phases import phase1_match, phase2_substructure_match, phase3_algebraic_check, determine_isomorphism
from .ratios import analyze_shape_ratios
from .quant_transfer import estimate_quant_transferability


def _test_strategy(ltc: float, sc: float, can_sub: bool) -> dict:
    if can_sub or (ltc >= 0.95 and sc >= 0.90):
        return {
            "level": "RECOMMENDED",
            "description": f"Layer type coverage {ltc:.1%}, shape compatibility {sc:.1%} — both above threshold.",
            "test_scope": "Full functional test suite on the smaller model.",
        }
    elif ltc >= 0.70 and sc >= 0.70:
        return {
            "level": "PARTIAL",
            "description": f"Layer type coverage {ltc:.1%}, shape compatibility {sc:.1%} — acceptable but below ideal.",
            "test_scope": "Core functional tests on smaller model + regression on target size.",
        }
    else:
        return {
            "level": "NOT_RECOMMENDED",
            "description": f"Layer type coverage {ltc:.1%}, shape compatibility {sc:.1%} — too low.",
            "test_scope": "Must use the target model size for all tests.",
        }


def compute_coverage(fp_a: ModelFingerprint, fp_b: ModelFingerprint) -> dict:
    sig_a = fp_a.static_weight_signature
    sig_b = fp_b.static_weight_signature

    p1, only_in_a, only_in_b = phase1_match(sig_a, sig_b)
    p2 = phase2_substructure_match(fp_a.template_signature, fp_b.template_signature)
    p3, scaling = phase3_algebraic_check(fp_a, fp_b)
    isomorphism, iso_rec = determine_isomorphism(fp_a, fp_b, p1, p2, p3)

    ka, kb = set(sig_a), set(sig_b)
    common = ka & kb
    coverage_rate = len(common) / max(len(ka), len(kb)) if (ka or kb) else 0.0
    a_covers_b = kb.issubset(ka)
    b_covers_a = ka.issubset(kb)

    ops_a = set(fp_a.op_types)
    ops_b = set(fp_b.op_types)
    op_union = ops_a | ops_b
    operator_coverage_rate = len(ops_a & ops_b) / len(op_union) if op_union else 1.0

    hl_a = fp_a.unique_ops_highlevel
    hl_b = fp_b.unique_ops_highlevel
    missing_hl_ops = sorted(hl_b - hl_a)

    lt_a = set(fp_a.layer_types)
    lt_b = set(fp_b.layer_types)
    lt_all = lt_a | lt_b
    layer_type_coverage = len(lt_a & lt_b) / len(lt_all) if lt_all else 1.0

    all_keys = ka | kb
    compatible = sum(
        1 for k in (ka & kb)
        if len(sig_a[k].get("representative_shape", [])) ==
           len(sig_b[k].get("representative_shape", []))
    )
    shape_compatibility = compatible / len(all_keys) if all_keys else 1.0

    shape_ratios = analyze_shape_ratios(sig_a, sig_b)
    all_uniform = all(v["uniform"] for v in shape_ratios.values()) if shape_ratios else True

    struct_keys = ["model_type", "num_attention_heads", "num_key_value_heads", "is_moe"]
    structural_coverage = all(
        fp_a.arch_fingerprint.get(k) == fp_b.arch_fingerprint.get(k)
        for k in struct_keys
        if k in fp_a.arch_fingerprint or k in fp_b.arch_fingerprint
    )

    can_substitute = layer_type_coverage >= 0.95 and shape_compatibility >= 0.90
    if isomorphism == "DIFFERENT_ARCH":
        verdict = "NO_SUBSTITUTE"
    elif isomorphism == "ISOMORPHIC" and all_uniform and can_substitute:
        verdict = "FULL_SUBSTITUTE"
    elif isomorphism in ("ISOMORPHIC", "SCALE_ONLY") or (operator_coverage_rate >= 0.80 and structural_coverage):
        verdict = "PARTIAL_SUBSTITUTE"
    else:
        verdict = "NO_SUBSTITUTE"

    strategy = _test_strategy(layer_type_coverage, shape_compatibility, can_substitute)

    quant_transfer = estimate_quant_transferability(
        fp_a, fp_b,
        isomorphism=isomorphism,
        layer_type_coverage=layer_type_coverage,
        shape_all_uniform=all_uniform,
    )

    return {
        "isomorphism": isomorphism,
        "substitution_verdict": verdict,
        "a_covers_b": a_covers_b,
        "b_covers_a": b_covers_a,
        "coverage_rate": round(coverage_rate, 6),
        "only_in_a": only_in_a[:20],
        "only_in_b": only_in_b[:20],
        "operator_coverage_rate": round(operator_coverage_rate, 4),
        "missing_highlevel_ops": missing_hl_ops,
        "layer_type_coverage": round(layer_type_coverage, 4),
        "shape_compatibility": round(shape_compatibility, 4),
        "can_substitute": can_substitute,
        "structural_coverage": structural_coverage,
        "phase1_normalized_match": p1,
        "phase2_substructure_match": p2,
        "phase3_algebraic_consistent": p3,
        "scaling_analysis": scaling,
        "shape_ratios_all_uniform": all_uniform,
        "non_uniform_shape_keys": [k for k, v in shape_ratios.items() if not v["uniform"]][:10],
        "test_strategy": strategy,
        "recommendation": iso_rec,
        "quant_transfer": quant_transfer,
    }
