"""HuggingFace Hub client — auth, HTTP, file listing, downloads."""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional

from ..constants import TOOL_NAME, TOOL_VERSION, HF_BASE

try:
    import huggingface_hub as _hfhub
    from huggingface_hub import (
        HfApi as _HfApi,
        hf_hub_url as _hf_hub_url,
        hf_hub_download as _hf_hub_download,
    )
    from huggingface_hub.utils import build_hf_headers as _build_hf_headers
    _HFHUB_OK = True
except ImportError:
    _HFHUB_OK = False

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# Module-level HF token (set via --token or HF_TOKEN env var)
import os
_HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
_HF_API: Optional[Any] = None


def set_token(token: str) -> None:
    global _HF_TOKEN, _HF_API
    _HF_TOKEN = token
    _HF_API = None


def get_token() -> Optional[str]:
    return _HF_TOKEN


def _get_hf_api() -> Any:
    global _HF_API
    if _HF_API is None and _HFHUB_OK:
        _HF_API = _HfApi(token=_HF_TOKEN)
    return _HF_API


def _hf_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if _HFHUB_OK:
        h = dict(_build_hf_headers(token=_HF_TOKEN))
    else:
        h = {"User-Agent": f"{TOOL_NAME}/{TOOL_VERSION}"}
    if extra:
        h.update(extra)
    return h


def http_get(url: str, headers: Optional[Dict[str, str]] = None,
             timeout: int = 30, max_retries: int = 4) -> bytes:
    import urllib.request
    hdrs = _hf_headers(headers)
    delay = 2.0
    last_exc: Exception = RuntimeError("no attempts")
    for attempt in range(max_retries):
        if attempt:
            print(f"  [retry {attempt}/{max_retries-1} after {delay:.0f}s]", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)
        try:
            if _REQUESTS_OK:
                r = _requests.get(url, headers=hdrs, timeout=timeout, allow_redirects=True)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", delay))
                    time.sleep(max(wait, delay))
                    continue
                r.raise_for_status()
                return r.content
            else:
                req = urllib.request.Request(url, headers=hdrs)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
        except Exception as e:
            last_exc = e
            s = str(e)
            if any(x in s for x in ("429", "Too Many", "timeout", "Connection")):
                continue
            raise
    raise RuntimeError(f"Max retries for {url}: {last_exc}")


def hf_resolve_url(model_id: str, filename: str) -> str:
    if _HFHUB_OK:
        return _hf_hub_url(model_id, filename)
    return f"{HF_BASE}/{model_id}/resolve/main/{filename}"


def hf_load_json_file(model_id: str, filename: str) -> dict:
    if _HFHUB_OK:
        try:
            import json as _json
            local = _hf_hub_download(model_id, filename, token=_HF_TOKEN)
            with open(local) as f:
                return _json.load(f)
        except Exception:
            pass
    url = hf_resolve_url(model_id, filename)
    data = http_get(url)
    return json.loads(data.decode("utf-8"))


def hf_model_files(model_id: str) -> List[dict]:
    if _HFHUB_OK:
        try:
            api = _get_hf_api()
            info = api.model_info(model_id, token=_HF_TOKEN)
            siblings = info.siblings or []
            return [
                {"rfilename": s.rfilename, "size": getattr(s, "size", None)}
                for s in siblings
            ]
        except Exception:
            pass
    try:
        url = f"{HF_BASE}/api/models/{model_id}"
        data = json.loads(http_get(url).decode())
        return data.get("siblings", [])
    except Exception:
        return []


def hf_hub_download_file(model_id: str, filename: str, token: Optional[str] = None) -> Optional[str]:
    if not _HFHUB_OK:
        return None
    try:
        return _hf_hub_download(model_id, filename, token=token or _HF_TOKEN)
    except Exception:
        return None
