"""Config loading with AutoConfig normalization and field alias fallback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..hf.client import hf_load_json_file, get_token

_AUTOCONFIG_OK: Optional[bool] = None  # None = not yet checked


def _flatten_config(config: dict) -> dict:
    result = dict(config)
    for sub_key in ("text_config", "llm_config", "language_config", "language_model_config"):
        sub = config.get(sub_key)
        if isinstance(sub, dict):
            for k, v in sub.items():
                if k not in result:
                    result[k] = v
    _aliases: List[Tuple[str, str]] = [
        ("n_embd",   "hidden_size"),
        ("n_head",   "num_attention_heads"),
        ("n_layer",  "num_hidden_layers"),
        ("n_inner",  "intermediate_size"),
        ("dim",      "hidden_size"),
        ("n_heads",  "num_attention_heads"),
        ("n_layers", "num_hidden_layers"),
        ("hidden_dim", "intermediate_size"),
        ("d_model",  "hidden_size"),
        ("num_layers", "num_hidden_layers"),
        ("d_ff",     "intermediate_size"),
        ("num_heads", "num_attention_heads"),
        ("d_kv",     "head_dim"),
        ("encoder_attention_heads", "num_attention_heads"),
        ("encoder_layers",          "num_hidden_layers"),
    ]
    for src, dst in _aliases:
        if dst not in result and src in result:
            result[dst] = result[src]
    return result


def _try_autoconfig(model_id_or_path: str, token: Optional[str] = None) -> Optional[dict]:
    global _AUTOCONFIG_OK
    if _AUTOCONFIG_OK is False:
        return None
    try:
        from transformers import AutoConfig
        _AUTOCONFIG_OK = True
        kwargs: Dict[str, Any] = {"trust_remote_code": False}
        if token:
            kwargs["token"] = token
        cfg = AutoConfig.from_pretrained(model_id_or_path, **kwargs)
        canonical_attrs = [
            "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "intermediate_size", "head_dim",
            "max_position_embeddings", "rope_theta", "rope_scaling", "sliding_window",
            "num_experts_per_tok", "num_local_experts", "vocab_size", "architectures",
            "layer_types",
        ]
        result: Dict[str, Any] = {}
        for attr in canonical_attrs:
            val = getattr(cfg, attr, None)
            if val is not None:
                result[attr] = val
        if hasattr(cfg, "to_dict"):
            raw = cfg.to_dict()
            for k, v in raw.items():
                if k not in result:
                    result[k] = v
        return result
    except Exception:
        _AUTOCONFIG_OK = False
        return None


def load_config(model_id: str, local_path: Optional[str] = None) -> dict:
    src = local_path if local_path else model_id
    cfg = _try_autoconfig(src, token=get_token())
    if cfg:
        return cfg
    if local_path:
        cfg_path = Path(local_path) / "config.json"
        try:
            with open(cfg_path) as f:
                return _flatten_config(json.load(f))
        except Exception:
            return {}
    try:
        return _flatten_config(hf_load_json_file(model_id, "config.json"))
    except Exception:
        return {}
