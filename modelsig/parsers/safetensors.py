"""Safetensors header parsing — local and remote (zero weight download)."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import List, Optional

from ..hf.client import http_get, hf_resolve_url, hf_load_json_file


def parse_local_header(path: str) -> dict:
    with open(path, "rb") as f:
        size_bytes = f.read(8)
        if len(size_bytes) < 8:
            return {}
        header_len = struct.unpack_from("<Q", size_bytes, 0)[0]
        raw = json.loads(f.read(header_len).decode("utf-8"))
    raw.pop("__metadata__", None)
    return raw


def parse_remote_header(url: str) -> dict:
    data = http_get(url, headers={"Range": "bytes=0-7"})
    header_len = struct.unpack_from("<Q", data)[0]
    body = http_get(url, headers={"Range": f"bytes=8-{7 + header_len}"})
    raw = json.loads(body.decode("utf-8"))
    return {k: v for k, v in raw.items() if k != "__metadata__"}


def discover_shards_remote(model_id: str) -> List[str]:
    try:
        idx = hf_load_json_file(model_id, "model.safetensors.index.json")
        return sorted(set(idx["weight_map"].values()))
    except Exception:
        return ["model.safetensors"]


def discover_shards_local(directory: str) -> List[str]:
    d = Path(directory)
    index = d / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            import json as _json
            idx = _json.load(f)
        return sorted(set(idx.get("weight_map", {}).values()))
    if (d / "model.safetensors").exists():
        return ["model.safetensors"]
    found = sorted(p.name for p in d.glob("*.safetensors"))
    if found:
        return found
    raise FileNotFoundError(f"No .safetensors files found in {directory}")


def collect_raw_tensors(model_id: str, local_path: Optional[str] = None) -> dict:
    merged: dict = {}
    if local_path:
        d = Path(local_path)
        shards = discover_shards_local(local_path)
        for shard in shards:
            try:
                hdr = parse_local_header(str(d / shard))
                for k, v in hdr.items():
                    merged.setdefault(k, v)
            except Exception as e:
                print(f"  [warn] {shard}: {e}", file=sys.stderr)
    else:
        shards = discover_shards_remote(model_id)
        for shard in shards:
            url = hf_resolve_url(model_id, shard)
            try:
                hdr = parse_remote_header(url)
                for k, v in hdr.items():
                    merged.setdefault(k, v)
            except Exception as e:
                print(f"  [warn] {shard}: {e}", file=sys.stderr)
    return merged
