"""ONNX model graph parsing — onnx library (preferred) or protobuf fallback."""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .ops import _ONNX_DTYPE

try:
    import onnx as _onnx_lib
    _ONNX_LIB_OK = True
except ImportError:
    _ONNX_LIB_OK = False


def parse_model_bytes_lib(local_path: str) -> Tuple[dict, List[str]]:
    """Parse via onnx library (load_external_data=False — no weight data loaded)."""
    model = _onnx_lib.load(local_path, load_external_data=False)
    graph = model.graph
    tensor_meta: dict = {}
    for init in graph.initializer:
        if init.name.startswith("/"):
            continue
        dims = list(init.dims)
        if not dims:
            continue
        dtype_str = _ONNX_DTYPE.get(init.data_type, f"T{init.data_type}")
        tensor_meta[init.name] = {"shape": dims, "dtype": dtype_str}
    raw_ops = sorted({node.op_type for node in graph.node if node.op_type})
    return tensor_meta, raw_ops


def _pb_varint(data: bytes, pos: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _pb_fields(data: bytes, want: Set[int]) -> Dict[int, list]:
    pos = 0; n = len(data); out: Dict[int, list] = {}
    while pos < n:
        try:
            tag, pos = _pb_varint(data, pos)
        except (IndexError, KeyError):
            break
        fn, wt = tag >> 3, tag & 7
        try:
            if wt == 0:
                v, pos = _pb_varint(data, pos)
                if fn in want: out.setdefault(fn, []).append(v)
            elif wt == 1: pos += 8
            elif wt == 2:
                ln, pos = _pb_varint(data, pos)
                v = data[pos: pos + ln]; pos += ln
                if fn in want: out.setdefault(fn, []).append(v)
            elif wt == 5: pos += 4
            else: break
        except Exception: break
    return out


def parse_model_bytes_fallback(data: bytes) -> Tuple[dict, List[str]]:
    """Minimal protobuf ONNX parser — fallback when onnx library is unavailable."""
    mf = _pb_fields(data, {7})
    graphs = mf.get(7, [])
    if not graphs:
        return {}, []
    gf = _pb_fields(graphs[0], {1, 5})
    tensor_meta: dict = {}
    for init_b in gf.get(5, []):
        f = _pb_fields(init_b, {1, 2, 8})
        name_raw = f.get(8, [b""])[0]
        name = name_raw.decode("utf-8", "replace") if isinstance(name_raw, bytes) else str(name_raw)
        dims = [int(d) for d in f.get(1, [])]
        dt = int(f.get(2, [0])[0]) if f.get(2) else 0
        if name and dims and not name.startswith("/"):
            tensor_meta[name] = {"shape": dims, "dtype": _ONNX_DTYPE.get(dt, f"T{dt}")}
    raw_ops: List[str] = []
    for node_b in gf.get(1, []):
        f4 = _pb_fields(node_b, {4})
        raw = f4.get(4, [b""])[0]
        op = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else ""
        if op: raw_ops.append(op)
    return tensor_meta, sorted(set(raw_ops))
