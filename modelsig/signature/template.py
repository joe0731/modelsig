"""Layer template signature for gemini phase-2 comparison."""
from __future__ import annotations
import re
from ..constants import _SUBMODULE_KEYS
from .static import norm_dtype


def build_template_signature(tensor_meta: dict) -> dict:
    layer_keys: dict = {}
    for name, meta in tensor_meta.items():
        m = re.search(r"\.\d+\.(.+)", name)
        if m:
            suffix = m.group(1)
            layer_keys.setdefault(suffix, meta)
    template: dict = {}
    for suffix, meta in layer_keys.items():
        for sub in _SUBMODULE_KEYS:
            if sub in suffix:
                template[suffix] = {
                    "shape": meta.get("shape", []),
                    "dtype": norm_dtype(meta.get("dtype", "")),
                }
                break
    return template
