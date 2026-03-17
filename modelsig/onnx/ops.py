"""ONNX dtype/op type maps and canonical op mapping."""
from __future__ import annotations
from typing import Dict, List, Set

_ONNX_DTYPE: Dict[int, str] = {
    1: "F32", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16",
    6: "INT32", 7: "INT64", 8: "STRING", 9: "BOOL", 10: "FP16",
    11: "F64", 12: "UINT32", 13: "UINT64", 16: "BF16",
}

_ONNX_OP_MAP: Dict[str, str] = {
    "MatMul": "aten/mm",
    "Gemm": "aten/mm",
    "GroupQueryAttention": "attention",
    "MultiHeadAttention": "attention",
    "Attention": "attention",
    "SimplifiedLayerNormalization": "rms_norm",
    "SkipSimplifiedLayerNormalization": "rms_norm",
    "LayerNormalization": "layer_norm",
    "RotaryEmbedding": "rope",
    "Sigmoid": "silu",
    "Softmax": "softmax",
    "Gather": "embedding",
    "MoE": "moe_expert",
    "QOrderedMatMul": "aten/mm",
    "MatMulNBits": "aten/mm",
    "MatMulFpQ4": "aten/mm",
}


def onnx_op_types_to_canonical(onnx_ops: List[str]) -> List[str]:
    canonical: Set[str] = {"aten/mm", "scaled_dot_product_attention"}
    for op in onnx_ops:
        mapped = _ONNX_OP_MAP.get(op)
        if mapped:
            canonical.add(mapped)
        else:
            canonical.add(op.lower())
    return sorted(canonical)
