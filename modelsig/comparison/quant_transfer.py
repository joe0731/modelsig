"""Quantization transferability estimator.

Based on the framework:
  SensCorr > RepAlign > Outlier/Curv > StructSim

Without loading weights, this module estimates transferability from structural
features only, following the theory that same-family LLMs share:
  - Operator histogram similarity (OpHistSim)
  - Layer type distribution and sensitivity pattern
  - Dimension scaling regularity (shape ratio uniformity)
  - MoE routing characteristics (~5% additional outlier risk)

Reference: LLM.int8, AWQ, SmoothQuant, GPTQ/HAWQ, SpinQuant/QuaRot
"""
from __future__ import annotations
import math
from typing import List
from ..signature.fingerprint import ModelFingerprint


def _op_hist_sim(fp_a: ModelFingerprint, fp_b: ModelFingerprint) -> float:
    """Cosine similarity of operator frequency vectors."""
    all_ops = sorted(set(fp_a.op_types) | set(fp_b.op_types))
    if not all_ops:
        return 1.0
    freq_a = [fp_a.op_types.count(op) for op in all_ops]
    freq_b = [fp_b.op_types.count(op) for op in all_ops]
    dot = sum(a * b for a, b in zip(freq_a, freq_b))
    norm_a = math.sqrt(sum(x * x for x in freq_a))
    norm_b = math.sqrt(sum(x * x for x in freq_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _layer_type_hist_sim(fp_a: ModelFingerprint, fp_b: ModelFingerprint) -> float:
    """Jaccard similarity of layer type sets."""
    a, b = set(fp_a.layer_types), set(fp_b.layer_types)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _arch_risk_factors(fp_a: ModelFingerprint, fp_b: ModelFingerprint) -> List[str]:
    """Identify architecture-level quantization risk factors."""
    risks: List[str] = []
    afa, afb = fp_a.arch_fingerprint, fp_b.arch_fingerprint

    h_a = afa.get("hidden_size") or 0
    h_b = afb.get("hidden_size") or 0
    if h_a and h_b and h_b / h_a >= 4:
        risks.append(
            f"Large hidden_size ratio ({h_b}/{h_a}={h_b/h_a:.1f}x): "
            "activation outlier magnitude likely higher in target model — "
            "activation-aware methods (AWQ/SmoothQuant) may need recalibration."
        )

    gqa_a = fp_a.dimension_ratios.get("gqa_ratio")
    gqa_b = fp_b.dimension_ratios.get("gqa_ratio")
    if gqa_a is not None and gqa_b is not None and abs(gqa_a - gqa_b) > 0.5:
        risks.append(
            f"GQA ratio mismatch ({gqa_a} vs {gqa_b}): "
            "KV cache quantization strategies may not transfer directly."
        )

    ffn_a = fp_a.dimension_ratios.get("ffn_expansion")
    ffn_b = fp_b.dimension_ratios.get("ffn_expansion")
    if ffn_a and ffn_b and abs(ffn_a - ffn_b) > 0.3:
        risks.append(
            f"FFN expansion ratio differs ({ffn_a:.2f} vs {ffn_b:.2f}): "
            "FFN outlier distribution may vary; re-check SmoothQuant scaling."
        )

    if afa.get("is_moe") and not afb.get("is_moe"):
        risks.append(
            "Source is MoE, target is Dense: MoE routing sparsity analysis "
            "does not apply — transferability limited to shared backbone operators."
        )
    elif not afa.get("is_moe") and afb.get("is_moe"):
        risks.append(
            "Source is Dense, target is MoE: expert routing introduces ~5%% "
            "additional quantization uncertainty beyond backbone operators."
        )

    rope_a = afa.get("rope_theta")
    rope_b = afb.get("rope_theta")
    if rope_a and rope_b and rope_a != rope_b:
        risks.append(
            f"RoPE theta differs ({rope_a} vs {rope_b}): "
            "position embedding quantization sensitivity may differ."
        )

    return risks


def estimate_quant_transferability(
    fp_a: ModelFingerprint,
    fp_b: ModelFingerprint,
    isomorphism: str,
    layer_type_coverage: float,
    shape_all_uniform: bool,
) -> dict:
    """Estimate quantization method transferability from fp_a to fp_b.

    Structural-only estimate (no weight download).  Follows the priority:
      SensCorr > RepAlign > Outlier/Curv > StructSim

    Since SensCorr and RepAlign require running actual calibration data, this
    function uses the available structural proxies and reports confidence
    accordingly.  Consumers should treat scores as a filter for which methods
    *may* transfer, not as a guarantee.

    Parameters
    ----------
    fp_a : ModelFingerprint
        Smaller / source model (proxy).
    fp_b : ModelFingerprint
        Larger / target model.
    isomorphism : str
        Result from ``determine_isomorphism`` — ISOMORPHIC / SCALE_ONLY /
        DIFFERENT_ARCH.
    layer_type_coverage : float
        Fraction of layer types shared between the two models (0-1).
    shape_all_uniform : bool
        Whether all common weight shapes scale uniformly.

    Returns
    -------
    dict with keys:
        struct_sim_score, op_hist_sim, layer_type_hist_sim,
        moe_correction, arch_risk_factors,
        estimated_transferability, confidence,
        recommended_methods, caveats
    """
    struct_score = {
        "ISOMORPHIC": 1.0,
        "SCALE_ONLY": 0.80,
        "DIFFERENT_ARCH": 0.20,
    }.get(isomorphism, 0.20)

    op_hist = _op_hist_sim(fp_a, fp_b)
    lt_hist = _layer_type_hist_sim(fp_a, fp_b)

    shape_score = 1.0 if shape_all_uniform else 0.85

    moe_correction = 1.0
    if fp_a.arch_fingerprint.get("is_moe") != fp_b.arch_fingerprint.get("is_moe"):
        moe_correction = 0.90
    elif fp_b.arch_fingerprint.get("is_moe"):
        moe_correction = 0.95

    risk_factors = _arch_risk_factors(fp_a, fp_b)
    risk_penalty = max(0.0, 1.0 - 0.05 * len(risk_factors))

    # Weighted composite: StructSim 25%, OpHist 20%, LayerTypeHist 20%,
    # ShapeUniform 15%, MoE 10%, risk 10%
    score = (
        0.25 * struct_score
        + 0.20 * op_hist
        + 0.20 * lt_hist
        + 0.15 * shape_score
        + 0.10 * moe_correction
        + 0.10 * risk_penalty
    )

    if isomorphism == "ISOMORPHIC" and not risk_factors:
        confidence = "HIGH"
    elif isomorphism == "DIFFERENT_ARCH":
        confidence = "LOW"
    else:
        confidence = "MEDIUM"

    # Recommend methods based on structural compatibility
    recommended: List[str] = []
    caveats: List[str] = []

    if score >= 0.80:
        recommended += ["GPTQ (W4A16)", "AWQ (W4A16)"]
        caveats.append(
            "Weight-only methods (GPTQ/AWQ) transfer well for same-family models. "
            "Verify calibration perplexity on target before deployment."
        )
    if score >= 0.70 and not risk_factors:
        recommended.append("Mixed-precision (HAWQ-style bit allocation)")
        caveats.append(
            "Mixed-precision layer sensitivity ordering likely preserved; "
            "confirm with single-layer ablation on target model."
        )
    if fp_b.arch_fingerprint.get("is_moe"):
        recommended.append("Expert-aware quantization (expert-group calibration)")
        caveats.append(
            "For MoE target: quantize expert routing path separately; "
            "shared backbone methods transfer, routing path needs own calibration."
        )
    if risk_factors:
        caveats.append(
            "Activation-aware methods (SmoothQuant, rotation-based QuaRot/SpinQuant) "
            "require per-model activation profiling — cannot transfer static configs."
        )
    if not recommended:
        recommended.append("Run full per-model calibration")
        caveats.append(
            "Structural differences are too large for reliable transfer. "
            "Treat target model as independent quantization task."
        )

    caveats.append(
        "IMPORTANT: This is a structural estimate only. "
        "SensCorr (sensitivity correlation) and RepAlign (CKA layer alignment) "
        "require actual calibration data and are the strongest transfer predictors. "
        "Use this score as a pre-filter, not a final verdict."
    )

    return {
        "struct_sim_score": round(struct_score, 4),
        "op_hist_sim": round(op_hist, 4),
        "layer_type_hist_sim": round(lt_hist, 4),
        "shape_uniform_score": round(shape_score, 4),
        "moe_correction": round(moe_correction, 4),
        "arch_risk_factors": risk_factors,
        "estimated_transferability": round(score, 4),
        "confidence": confidence,
        "recommended_methods": recommended,
        "caveats": caveats,
    }
