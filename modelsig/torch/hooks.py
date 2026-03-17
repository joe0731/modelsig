"""Optional forward-hook shape capture (minimax)."""
from __future__ import annotations
import sys
from typing import Optional


def run_hook_capture(model_id: str, local_path: Optional[str]) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig

        path = local_path or model_id
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32,
                                                      trust_remote_code=True)
            model.eval()

        shapes: dict = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, inp, out):
                def _s(obj):
                    if isinstance(obj, torch.Tensor):
                        return [list(obj.shape)]
                    if isinstance(obj, (list, tuple)):
                        r = []
                        for o in obj:
                            r.extend(_s(o))
                        return r
                    return []
                shapes[name] = {"input_shapes": _s(inp), "output_shapes": _s(out)}
            return hook

        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            dummy = torch.zeros((1, 8), dtype=torch.long)
            with torch.no_grad():
                model(dummy)
        except Exception:
            pass
        finally:
            for h in hooks:
                h.remove()

        return shapes
    except Exception as exc:
        print(f"  [hook-capture] Not available: {exc}", file=sys.stderr)
        return {}
