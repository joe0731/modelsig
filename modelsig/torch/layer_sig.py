"""Layer-level I/O signature capture via forward hooks.

Enumerates every named module in the model and records:
  key   — normalized module path  (layer indices collapsed to .N.)
  value — {module_type, input: [{dtype, shape}], output: [{dtype, shape}]}

Runs on meta device so no GPU / real weights are needed.
"""
from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional


def _norm_path(name: str) -> str:
    """Collapse numeric indices: model.layers.3.self_attn → model.layers.N.self_attn"""
    return re.sub(r"\.\d+\.", ".N.", name)


def _tensor_info(obj) -> List[dict]:
    """Recursively extract {dtype, shape} from Tensor / nested sequence."""
    import torch
    results: List[dict] = []
    if isinstance(obj, torch.Tensor):
        results.append({
            "dtype": str(obj.dtype).replace("torch.", ""),
            "shape": list(obj.shape),
        })
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            results.extend(_tensor_info(item))
    return results


def collect_layer_signatures(
    model_id: str,
    local_path: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Dict[str, dict]:
    """Return per-module I/O dtype+shape signatures.

    Returns
    -------
    dict keyed by *raw* module name, e.g.:
        {
          "model.embed_tokens": {
              "module_type": "Embedding",
              "norm_path":   "model.embed_tokens",
              "input":  [{"dtype": "int64",   "shape": [1, 8]}],
              "output": [{"dtype": "float32", "shape": [1, 8, 4096]}],
          },
          "model.layers.0.self_attn": { ... },
          ...
        }
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        print(f"  [layer-sig] torch/transformers not available: {exc}", file=sys.stderr)
        return {}

    path = local_path or model_id
    try:
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                cfg, torch_dtype=torch.float32, trust_remote_code=trust_remote_code
            )
            model.eval()
    except Exception as exc:
        print(f"  [layer-sig] model load failed: {exc}", file=sys.stderr)
        return {}

    signatures: Dict[str, dict] = {}
    hooks = []

    def _make_hook(name: str, module_type: str):
        def hook(module, inp, out):
            signatures[name] = {
                "module_type": module_type,
                "norm_path":   _norm_path(name),
                "input":       _tensor_info(inp),
                "output":      _tensor_info(out),
            }
        return hook

    for name, module in model.named_modules():
        if not name:          # skip root module
            continue
        hooks.append(module.register_forward_hook(
            _make_hook(name, type(module).__name__)
        ))

    try:
        dummy = torch.zeros((1, 8), dtype=torch.long)
        with torch.no_grad():
            model(dummy)
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()

    return signatures
