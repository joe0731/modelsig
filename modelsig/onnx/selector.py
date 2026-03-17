"""ONNX file selection heuristics for HF model repos."""
from __future__ import annotations
from typing import List, Optional


def select_primary_onnx(files: List[dict]) -> Optional[str]:
    onnx_files = [f for f in files if f.get("rfilename", "").endswith(".onnx")]
    if not onnx_files:
        return None
    names = {f["rfilename"] for f in onnx_files}
    data_names = {f["rfilename"] for f in files if ".onnx_data" in f.get("rfilename", "")}

    def has_ext_data(fname: str) -> bool:
        return any(d.startswith(fname) for d in data_names)

    candidates_with_ext = [f["rfilename"] for f in onnx_files if has_ext_data(f["rfilename"])]

    def is_base_variant(fname: str) -> bool:
        base = fname.split("/")[-1]
        return not any(q in base for q in ("q4", "q8", "int8", "int4", "quant", "fp16", "bnb4", "uint8"))

    if candidates_with_ext:
        for preferred_name in ["decoder_model_merged.onnx", "model.onnx"]:
            for c in candidates_with_ext:
                if c.endswith(preferred_name) and is_base_variant(c):
                    return c
        for c in candidates_with_ext:
            if is_base_variant(c):
                return c
        return candidates_with_ext[0]

    for preferred in ["onnx/decoder_model_merged.onnx", "onnx/model.onnx", "model.onnx"]:
        if preferred in names:
            return preferred

    onnx_with_size = [(f.get("size") or 10**12, f["rfilename"]) for f in onnx_files
                      if is_base_variant(f["rfilename"])]
    if not onnx_with_size:
        onnx_with_size = [(f.get("size") or 10**12, f["rfilename"]) for f in onnx_files]
    onnx_with_size.sort()
    return onnx_with_size[0][1] if onnx_with_size else None


def is_onnx_model(files: List[dict]) -> bool:
    fnames = [f.get("rfilename", "") for f in files]
    return any(f.endswith(".onnx") for f in fnames) and not any(f.endswith(".safetensors") for f in fnames)
