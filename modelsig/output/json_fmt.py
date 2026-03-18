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
        "source": fp.source,
    }
    if fp.arch_fingerprint.get("onnx_op_types"):
        d["onnx_op_types"] = fp.arch_fingerprint.pop("onnx_op_types")
    if fp.layer_signatures:
        d["layer_signatures"] = fp.layer_signatures
    return d


def format_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=str, ensure_ascii=False)
