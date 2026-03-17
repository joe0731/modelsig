"""Optional FX symbolic trace on meta device (kimi)."""
from __future__ import annotations
import sys
from typing import Optional


def run_fx_trace(model_id: str, local_path: Optional[str]) -> bool:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.fx import symbolic_trace
        path = local_path or model_id
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32,
                                                      trust_remote_code=True)
            model.eval()
        sample = torch.empty((1, 16), dtype=torch.long, device="meta")
        symbolic_trace(model, concrete_args={"input_ids": sample})
        return True
    except Exception as exc:
        print(f"  [fx-trace] Not available: {exc}", file=sys.stderr)
        return False
