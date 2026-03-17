"""JSON output formatter."""
from __future__ import annotations
import json
from ..signature.fingerprint import ModelFingerprint


def fp_to_dict(fp: ModelFingerprint) -> dict:
    d: dict = {
        "arch_fingerprint": fp.arch_fingerprint,
        "static_weight_signature": fp.static_weight_signature,
        "op_types": fp.op_types,
        "unique_ops_highlevel": sorted(fp.unique_ops_highlevel),
        "layer_types": fp.layer_types,
        "kv_cache_shape_pattern": fp.kv_cache_shape_pattern,
        "dimension_ratios": fp.dimension_ratios,
        "fx_trace_available": fp.fx_trace_available,
        "source": fp.source,
    }
    if fp.arch_fingerprint.get("onnx_op_types"):
        d["onnx_op_types"] = fp.arch_fingerprint.pop("onnx_op_types")
    if fp.quant_path_signature:
        d["quant_path_signature"] = fp.quant_path_signature
    if fp.hook_shapes:
        d["hook_shapes_count"] = len(fp.hook_shapes)
    return d


def format_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=str, ensure_ascii=False)
