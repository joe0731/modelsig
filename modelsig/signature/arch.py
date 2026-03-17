"""Layer 2: Architecture fingerprint builder."""
from __future__ import annotations
import re
from typing import Any, Dict
from ..constants import _ARCH_FIELDS, _MOE_PATTERNS


def build_arch_fingerprint(config: dict, tensor_meta: dict) -> dict:
    fp: dict = {}
    for f in _ARCH_FIELDS:
        if f in config:
            fp[f] = config[f]

    if "head_dim" not in fp:
        hs = fp.get("hidden_size")
        nah = fp.get("num_attention_heads")
        # num_attention_heads can be a list (per-layer) in some hierarchical models
        if isinstance(nah, list):
            nah = max(nah) if nah else None
        if isinstance(hs, int) and isinstance(nah, int) and nah > 0:
            fp["head_dim"] = hs // nah

    is_moe = bool(
        config.get("num_local_experts") or config.get("num_experts") or
        config.get("moe_num_experts") or "moe" in config.get("model_type", "").lower()
    )
    if not is_moe:
        for key in tensor_meta:
            if any(re.search(p, key) for p in _MOE_PATTERNS):
                is_moe = True
                break
    fp["is_moe"] = is_moe
    fp["num_experts"] = (
        config.get("num_local_experts") or config.get("num_experts") or config.get("moe_num_experts")
    )
    fp["num_experts_per_tok"] = (
        config.get("num_experts_per_tok") or config.get("num_selected_experts")
    )
    fp.setdefault("layer_types", [])
    return fp


def build_kv_cache_shape_pattern(arch_fp: dict) -> str:
    num_kv = arch_fp.get("num_key_value_heads", arch_fp.get("num_attention_heads", "?"))
    head_dim = arch_fp.get("head_dim", "?")
    return f"[batch, {num_kv}, seq_len, {head_dim}]"


def compute_dimension_ratios(config: dict) -> dict:
    ratios: dict = {}
    h = config.get("hidden_size")
    ffn = config.get("intermediate_size")
    nah = config.get("num_attention_heads")
    nkv = config.get("num_key_value_heads")
    # Guard against per-layer list values
    if isinstance(nah, list): nah = max(nah) if nah else None
    if isinstance(nkv, list): nkv = max(nkv) if nkv else None
    if isinstance(h, int) and isinstance(ffn, int) and h > 0:
        ratios["ffn_expansion"] = round(ffn / h, 6)
    if isinstance(nah, int) and isinstance(nkv, int) and nkv > 0:
        ratios["gqa_ratio"] = round(nah / nkv, 6)
    return ratios
