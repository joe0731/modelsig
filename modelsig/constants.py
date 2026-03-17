"""Shared constants and inference rules for modelsig."""
from __future__ import annotations
from typing import List, Tuple

TOOL_NAME = "modelsig"
TOOL_VERSION = "2.0"
HF_BASE = "https://huggingface.co"

_OP_RULES: List[Tuple[str, str]] = [
    (r"\.(q_proj|k_proj|v_proj|o_proj)\.", "attention"),
    (r"\.(query_key_value|qkv_proj|query|key|value)\.", "attention"),
    (r"\.(gate_proj|up_proj|down_proj)\.", "swiglu"),
    (r"\.(fc1|fc2|ffn)\.", "ffn"),
    (r"\.experts\.", "moe_expert"),
    (r"\.(router|gate)\.", "moe_router"),
    (r"\.shared_experts\.", "moe_shared_expert"),
    (r"\.(embed_tokens|wte|word_embeddings)\.", "embedding"),
    (r"\.(lm_head|output)\.", "lm_head"),
    (r"\.(input_layernorm|post_attention_layernorm|norm|ln_f|layernorm)\.", "rms_norm"),
    (r"\.(rotary_emb|rope)\.", "rope"),
]

_LAYER_TYPE_RULES: List[Tuple[List[str], str]] = [
    (["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "query_key_value", "self_attn", "attention"], "AttentionLayer"),
    (["gate_proj", "up_proj", "down_proj", "fc1", "fc2", "ffn"], "FFN_SwiGLU"),
    (["experts", "moe", "router"], "MoELayer"),
    (["layernorm", "layer_norm", "rmsnorm", "norm"], "RMSNorm"),
    (["embed_tokens", "embedding", "wte", "wpe"], "EmbeddingLayer"),
    (["lm_head"], "LMHead"),
    (["rotary_emb", "rope"], "RotaryEmb"),
]

_ARCH_FIELDS = [
    "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "intermediate_size", "head_dim",
    "max_position_embeddings", "rope_theta", "rope_scaling", "sliding_window",
    "layer_types", "num_experts_per_tok", "num_local_experts",
    "vocab_size", "architectures",
]

_MOE_PATTERNS = [r"mlp\.experts\.", r"block_sparse_moe", r"moe\.experts\."]

_SUBMODULE_KEYS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "input_layernorm", "post_attention_layernorm",
]

_DTYPE_MAP = {
    "BF16": "BF16", "F16": "FP16", "F32": "FP32", "F64": "FP64",
    "F8_E4M3": "FP8_E4M3", "F8_E5M2": "FP8_E5M2",
    "I8": "INT8", "I4": "INT4", "U8": "UINT8", "BOOL": "BOOL",
    "torch.bfloat16": "BF16", "torch.float16": "FP16",
    "torch.float32": "FP32", "torch.float8_e4m3fn": "FP8_E4M3",
    "torch.int8": "INT8",
}
