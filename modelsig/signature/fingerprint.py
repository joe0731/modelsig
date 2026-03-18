"""ModelFingerprint dataclass and builder."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..constants import _OP_RULES, _LAYER_TYPE_RULES
from ..hf.client import hf_model_files
from ..parsers.safetensors import collect_raw_tensors
from ..parsers.config import load_config
from ..onnx.collector import collect_raw_tensors_onnx
from ..onnx.ops import onnx_op_types_to_canonical
from ..onnx.selector import is_onnx_model
from ..torch.layer_sig import collect_layer_signatures
from .static import build_static_weight_signature, norm_dtype, param_count
from .arch import build_arch_fingerprint, build_kv_cache_shape_pattern, compute_dimension_ratios
from .template import build_template_signature


@dataclass
class ModelFingerprint:
    model_id: str
    static_weight_signature: Dict[str, dict] = field(default_factory=dict)
    arch_fingerprint: Dict[str, Any] = field(default_factory=dict)
    op_types: List[str] = field(default_factory=list)
    kv_cache_shape_pattern: str = ""
    unique_ops_highlevel: Set[str] = field(default_factory=set)
    layer_types: List[str] = field(default_factory=list)
    dimension_ratios: Dict[str, Any] = field(default_factory=dict)
    template_signature: Dict[str, dict] = field(default_factory=dict)
    layer_signatures: Dict[str, dict] = field(default_factory=dict)
    source: str = "safetensors"


def _infer_op_types(tensor_meta: dict) -> List[str]:
    ops: Set[str] = {"aten/mm", "scaled_dot_product_attention"}
    all_keys = " ".join(k for k in tensor_meta if k != "__metadata__")
    for pat, label in _OP_RULES:
        if re.search(pat, all_keys):
            ops.add(label)
    if re.search(r"\.(input_layernorm|post_attention_layernorm|norm)\.", all_keys):
        ops.add("rms_norm")
    if re.search(r"\.rotary_emb\.|\.rope\.", all_keys):
        ops.add("rope")
    if re.search(r"\.(gate_proj|up_proj)\.", all_keys):
        ops.add("silu")
    if re.search(r"\.experts\.|\.router\.|\.gate\.", all_keys):
        ops.add("topk/router")
    return sorted(ops)


def _infer_unique_ops_highlevel(tensor_meta: dict) -> Set[str]:
    ops: Set[str] = set()
    for key in tensor_meta:
        k = key.lower()
        if any(x in k for x in ("q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "attention")):
            ops.add("Attention")
        if any(x in k for x in ("gate_proj", "up_proj", "down_proj", "swiglu", "ffn", "mlp")):
            ops.add("SwiGLU/FFN")
        if any(x in k for x in ("experts", "moe", "router")):
            ops.add("MoE")
        if any(x in k for x in ("norm", "layernorm", "rmsnorm", "layer_norm")):
            ops.add("Norm")
        if "embed" in k:
            ops.add("Embedding")
        if any(x in k for x in ("lm_head", "head")):
            ops.add("LMHead")
        if any(x in k for x in ("rotary", "rope", "pos_emb")):
            ops.add("RotaryEmb")
    return ops or {"Linear"}


def _minimal_arch_config(config: dict) -> dict:
    is_moe = bool(
        config.get("num_local_experts") or config.get("num_experts") or
        config.get("moe_num_experts") or "moe" in config.get("model_type", "").lower()
    )
    return {
        "model_type": config.get("model_type", "unknown"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "intermediate_size": config.get("intermediate_size"),
        "vocab_size": config.get("vocab_size"),
        "is_moe": is_moe,
    }


def _scalar_int(v) -> int:
    """Return v as int, handling per-layer list values (take max)."""
    if isinstance(v, list):
        return max(v) if v else 0
    return int(v) if v else 0


def _synthetic_sig_from_config(cfg: dict) -> Tuple[dict, List[str]]:
    h = _scalar_int(cfg.get("hidden_size"))
    i = _scalar_int(cfg.get("intermediate_size"))
    v = _scalar_int(cfg.get("vocab_size"))
    nah = _scalar_int(cfg.get("num_attention_heads"))
    nkv = _scalar_int(cfg.get("num_key_value_heads")) or nah
    head_dim = (h // nah) if nah else 0
    is_moe = cfg.get("is_moe", False)
    sig: dict = {}
    ltypes: set = set()

    def add(key, shape, dtype, lt):
        sig[key] = {"representative_shape": shape, "dtype": dtype, "layer_type": lt,
                    "count": 1, "param_count": param_count(shape)}
        ltypes.add(lt)

    if v and h:
        add("model.embed_tokens.weight", [v, h], "BF16", "EmbeddingLayer")
        add("lm_head.weight", [v, h], "BF16", "LMHead")
    if h:
        add("model.layers.N.input_layernorm.weight", [h], "FP32", "RMSNorm")
        add("model.layers.N.post_attention_layernorm.weight", [h], "FP32", "RMSNorm")
    if h and nah:
        add("model.layers.N.self_attn.q_proj.weight", [nah * head_dim, h], "BF16", "AttentionLayer")
        add("model.layers.N.self_attn.k_proj.weight", [nkv * head_dim, h], "BF16", "AttentionLayer")
        add("model.layers.N.self_attn.v_proj.weight", [nkv * head_dim, h], "BF16", "AttentionLayer")
        add("model.layers.N.self_attn.o_proj.weight", [h, nah * head_dim], "BF16", "AttentionLayer")
    if h and i:
        if is_moe:
            add("model.layers.N.mlp.experts.N.gate_proj.weight", [i, h], "BF16", "MoELayer")
            add("model.layers.N.mlp.experts.N.up_proj.weight",   [i, h], "BF16", "MoELayer")
            add("model.layers.N.mlp.experts.N.down_proj.weight", [h, i], "BF16", "MoELayer")
        else:
            add("model.layers.N.mlp.gate_proj.weight", [i, h], "BF16", "FFN_SwiGLU")
            add("model.layers.N.mlp.up_proj.weight",   [i, h], "BF16", "FFN_SwiGLU")
            add("model.layers.N.mlp.down_proj.weight", [h, i], "BF16", "FFN_SwiGLU")
    return sig, sorted(ltypes)


def build_fingerprint(
    model_id: str,
    local_path: Optional[str] = None,
    fast: bool = False,
    timeout: int = 30,
    trust_remote_code: bool = False,
    layer_sig: bool = False,
) -> ModelFingerprint:
    print(f"  Analyzing: {model_id}", file=sys.stderr)
    config = load_config(model_id, local_path)
    onnx_raw_ops: List[str] = []

    if fast:
        arch_fp_raw = _minimal_arch_config(config)
        static_sig, layer_types = _synthetic_sig_from_config(arch_fp_raw)
        tensor_meta: dict = {}
        source = "config_only"
    else:
        fmt = "safetensors"
        if not local_path:
            files = hf_model_files(model_id)
            fnames = [f.get("rfilename", "") for f in files]
            has_st = any(f.endswith(".safetensors") for f in fnames)
            has_onnx = any(f.endswith(".onnx") for f in fnames)
            if has_onnx and not has_st:
                fmt = "onnx"

        if fmt == "onnx":
            try:
                tensor_meta, onnx_raw_ops = collect_raw_tensors_onnx(model_id)
                static_sig = build_static_weight_signature(tensor_meta)
                layer_types = sorted(set(v["layer_type"] for v in static_sig.values()))
                source = "onnx"
            except Exception as exc:
                print(f"  [warn] ONNX parse failed ({exc}), falling back to config", file=sys.stderr)
                arch_fp_raw = _minimal_arch_config(config)
                static_sig, layer_types = _synthetic_sig_from_config(arch_fp_raw)
                tensor_meta = {}
                onnx_raw_ops = []
                source = "config_only"
        else:
            try:
                tensor_meta = collect_raw_tensors(model_id, local_path)
                static_sig = build_static_weight_signature(tensor_meta)
                layer_types = sorted(set(v["layer_type"] for v in static_sig.values()))
                source = "safetensors"
            except Exception as exc:
                print(f"  [warn] safetensors parse failed ({exc}), falling back to config", file=sys.stderr)
                arch_fp_raw = _minimal_arch_config(config)
                static_sig, layer_types = _synthetic_sig_from_config(arch_fp_raw)
                tensor_meta = {}
                source = "config_only"

    arch_fp = build_arch_fingerprint(config, tensor_meta)

    if source == "onnx" and onnx_raw_ops:
        op_types = onnx_op_types_to_canonical(onnx_raw_ops)
        arch_fp["onnx_op_types"] = onnx_raw_ops
    else:
        op_types = _infer_op_types(tensor_meta)

    kv_pattern = build_kv_cache_shape_pattern(arch_fp)
    dim_ratios = compute_dimension_ratios(config)
    template_sig = build_template_signature(tensor_meta)
    unique_ops_hl = _infer_unique_ops_highlevel(tensor_meta) if tensor_meta else set()

    layer_sigs: dict = {}
    if layer_sig:
        layer_sigs = collect_layer_signatures(model_id, local_path, trust_remote_code=trust_remote_code)

    return ModelFingerprint(
        model_id=model_id,
        static_weight_signature=static_sig,
        arch_fingerprint=arch_fp,
        op_types=op_types,
        kv_cache_shape_pattern=kv_pattern,
        unique_ops_highlevel=unique_ops_hl,
        layer_types=layer_types,
        dimension_ratios=dim_ratios,
        template_signature=template_sig,
        layer_signatures=layer_sigs,
        source=source,
    )
