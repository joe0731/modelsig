"""QuantPathSignature builder (chagpt)."""
from __future__ import annotations


def build_quant_path_signature(arch_fp: dict, config: dict) -> dict:
    is_moe = arch_fp.get("is_moe", False)
    has_gqa = arch_fp.get("num_key_value_heads") not in (None, arch_fp.get("num_attention_heads"))
    arch_template = "moe_decoder" if is_moe else ("gqa_decoder" if has_gqa else "dense_decoder")
    qc = config.get("quantization_config", {}) or {}
    return {
        "arch_template": arch_template,
        "quant_algo": str(qc.get("quant_type") or qc.get("bits") or "none"),
        "weight_dtype": str(qc.get("weight_dtype") or qc.get("compute_dtype") or "bf16"),
        "act_dtype": str(qc.get("act_dtype") or "bf16"),
        "scale_scheme": str(qc.get("scale_scheme") or "per-channel"),
        "group_size": qc.get("group_size") or qc.get("q_group_size") or 128,
        "placement_policy": "all-linear",
        "kv_cache_dtype": str(qc.get("kv_cache_dtype") or "fp16"),
        "backend": "unknown",
        "hardware_class": "unknown",
        "workload_regime": "prefill_short|decode_bs1",
        "calibration_or_qat": "none",
    }
