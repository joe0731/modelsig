"""Layer 1: Static weight signature builder."""
from __future__ import annotations
import re
from typing import List
from ..constants import _DTYPE_MAP, _LAYER_TYPE_RULES

_LAYER_IDX_RE = re.compile(r"\.\d+\.")


def norm_dtype(raw: str) -> str:
    return _DTYPE_MAP.get(raw, _DTYPE_MAP.get(raw.upper(), raw.upper() or "UNKNOWN"))


def norm_key(key: str) -> str:
    return _LAYER_IDX_RE.sub(".N.", key)


def infer_layer_type(name: str) -> str:
    lower = name.lower()
    for keywords, label in _LAYER_TYPE_RULES:
        if any(k in lower for k in keywords):
            return label
    return "LinearLayer"


def param_count(shape: list) -> int:
    r = 1
    for d in shape:
        r *= d
    return r


def build_static_weight_signature(tensor_meta: dict) -> dict:
    sig: dict = {}
    for raw_key, meta in tensor_meta.items():
        if not isinstance(meta, dict):
            continue
        shape = meta.get("shape", [])
        dtype = norm_dtype(meta.get("dtype", ""))
        layer_type = infer_layer_type(raw_key)
        abstract = norm_key(raw_key)
        if abstract not in sig:
            sig[abstract] = {
                "representative_shape": shape,
                "dtype": dtype,
                "layer_type": layer_type,
                "count": 0,
                "param_count": 0,
            }
        sig[abstract]["count"] += 1
        sig[abstract]["param_count"] += param_count(shape)
    return sig
