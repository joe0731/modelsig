"""Orchestrates ONNX model fetching and parsing."""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

from ..hf.client import hf_model_files, hf_resolve_url, hf_hub_download_file, http_get, get_token, _REQUESTS_OK
from .ops import onnx_op_types_to_canonical
from .parser import _ONNX_LIB_OK, parse_model_bytes_lib, parse_model_bytes_fallback
from .selector import select_primary_onnx, is_onnx_model

_ONNX_SIZE_LIMIT = 50 * 1024 * 1024  # 50 MB


def collect_raw_tensors_onnx(model_id: str) -> Tuple[dict, List[str]]:
    files = hf_model_files(model_id)
    onnx_path = select_primary_onnx(files)
    if not onnx_path:
        print(f"  [onnx] No .onnx file found in {model_id}", file=sys.stderr)
        return {}, []

    print(f"  [onnx] Fetching {onnx_path}", file=sys.stderr)
    local_onnx_path: Optional[str] = None

    local_onnx_path = hf_hub_download_file(model_id, onnx_path, token=get_token())
    if local_onnx_path:
        fsize = os.path.getsize(local_onnx_path)
        if fsize > _ONNX_SIZE_LIMIT:
            print(f"  [onnx] File too large ({fsize/1e6:.0f} MB > "
                  f"{_ONNX_SIZE_LIMIT//1024//1024} MB limit); falling back to config-only mode",
                  file=sys.stderr)
            return {}, []

    onnx_bytes: Optional[bytes] = None
    if local_onnx_path is None:
        url = hf_resolve_url(model_id, onnx_path)
        try:
            if _REQUESTS_OK:
                import requests as _requests
                head = _requests.head(url, headers={}, timeout=15, allow_redirects=True)
                cl = int(head.headers.get("Content-Length", 0) or 0)
                if cl > _ONNX_SIZE_LIMIT:
                    print(f"  [onnx] File too large ({cl/1e6:.0f} MB), skipping", file=sys.stderr)
                    return {}, []
            onnx_bytes = http_get(url, timeout=60)
        except Exception as e:
            print(f"  [onnx] Download failed: {e}", file=sys.stderr)
            return {}, []

    try:
        if _ONNX_LIB_OK and local_onnx_path:
            print(f"  [onnx] Parsing via onnx library ({os.path.getsize(local_onnx_path)//1024} KB) ...",
                  file=sys.stderr)
            return parse_model_bytes_lib(local_onnx_path)
        else:
            if onnx_bytes is None:
                with open(local_onnx_path, "rb") as f:  # type: ignore[arg-type]
                    onnx_bytes = f.read()
            print(f"  [onnx] Parsing via protobuf fallback ({len(onnx_bytes)//1024} KB) ...",
                  file=sys.stderr)
            return parse_model_bytes_fallback(onnx_bytes)
    except Exception as e:
        print(f"  [onnx] Parse error: {e}", file=sys.stderr)
        return {}, []
