#!/usr/bin/env python3
"""
modelsig/analyze.py — CLI entry point.

Synthesizes the best approaches from 5 independent plans:
  chagpt     – 4-layer signature system, QuantPathSignature
  gemini     – 3-phase isomorphism comparison, multi-fidelity test plan
  grok       – shape-ratio uniformity analysis, ANSI color output
  kimi       – OOP architecture (SafetensorAnalyzer, CoverageAnalyzer), optional FX trace
  minimax    – forward-hook shape capture (lazy), test strategy, config-only fast mode

Usage:
  python analyze.py MODEL_ID [MODEL_ID ...] [OPTIONS]

  python analyze.py Qwen/Qwen3-7B Qwen/Qwen3-72B --compare --output table
  python analyze.py Qwen/Qwen3-30B-A3B Qwen/Qwen3-235B-A22B --compare --multi-fidelity
  python analyze.py local:/models/7b local:/models/72b --compare --output markdown
  python analyze.py Qwen/Qwen3-7B --quant-path --output json --save report.json
  python analyze.py org/private-model --token hf_xxx --output table
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from .constants import TOOL_NAME, TOOL_VERSION
from .hf import client as _hf_client
from .signature.fingerprint import ModelFingerprint, build_fingerprint
from .comparison.coverage import compute_coverage
from .comparison.multifidelity import build_multi_fidelity_plan
from .output.json_fmt import fp_to_dict, format_json
from .output.table_fmt import format_table
from .output.markdown_fmt import format_markdown


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modelsig",
        description=(
            "modelsig: compare LLM architectures without downloading weights.\n"
            "Zero weight download — uses HTTP Range for safetensors headers only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  modelsig Qwen/Qwen3-7B Qwen/Qwen3-72B --compare --output table
  modelsig Qwen/Qwen3-7B Qwen/Qwen3-30B-A3B Qwen/Qwen3-235B-A22B \\
      --compare --multi-fidelity --output markdown --save report.md
  modelsig local:/models/7b local:/models/72b --compare
  modelsig Qwen/Qwen3-7B --quant-path --output json
  modelsig Qwen/Qwen3-235B-A22B --fast --output table
  modelsig Qwen/Qwen3-7B --no-fx-trace --no-hook-capture --output json
""",
    )
    p.add_argument("model_ids", nargs="*", metavar="MODEL_ID",
                   help="HF model IDs or local:PATH entries")
    p.add_argument("-m", "--model", dest="model_flag", action="append",
                   metavar="MODEL_ID", default=None,
                   help="Model ID (repeatable; merged with positional MODEL_IDs)")
    p.add_argument("--local", metavar="PATH", default=None,
                   help="Local directory for the first positional MODEL_ID")
    p.add_argument("--output", choices=["json", "table", "markdown"], default="json",
                   help="Output format (default: json)")
    p.add_argument("--compare", action="store_true",
                   help="Compute pairwise coverage for all model pairs")
    p.add_argument("--save", metavar="FILE", default=None,
                   help="Save results to file")
    p.add_argument("--quant-path", action="store_true", dest="quant_path",
                   help="Include QuantPathSignature in output")
    p.add_argument("--multi-fidelity", action="store_true", dest="multi_fidelity",
                   help="Include 4-level multi-fidelity test plan")
    p.add_argument("--fast", action="store_true",
                   help="Config-only mode (no safetensors parsing, fastest)")
    fx_grp = p.add_mutually_exclusive_group()
    fx_grp.add_argument("--fx-trace", action="store_true", dest="fx_trace", default=True)
    fx_grp.add_argument("--no-fx-trace", action="store_false", dest="fx_trace")
    hk_grp = p.add_mutually_exclusive_group()
    hk_grp.add_argument("--hook-capture", action="store_true", dest="hook_capture", default=True)
    hk_grp.add_argument("--no-hook-capture", action="store_false", dest="hook_capture")
    p.add_argument("--timeout", type=int, default=30, metavar="SEC")
    p.add_argument("--token", metavar="TOKEN", default=None,
                   help="HuggingFace Hub token (overrides HF_TOKEN env var)")
    p.add_argument("--no-color", action="store_true", dest="no_color")
    p.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code",
                   help="Pass trust_remote_code=True to AutoConfig/AutoModel (use only for "
                        "verified models — enables arbitrary code execution)")
    p.add_argument("--layer-sig", action="store_true", dest="layer_sig",
                   help="Collect per-module input/output dtype+shape signatures via forward hooks "
                        "(requires torch + transformers)")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "token", None):
        _hf_client.set_token(args.token)

    raw_ids = list(args.model_ids) + (args.model_flag or [])
    if args.local and not raw_ids:
        raw_ids = [os.path.basename(args.local.rstrip("/"))]

    targets: List[Tuple[str, Optional[str]]] = []
    for idx, raw in enumerate(raw_ids):
        if raw.startswith("local:"):
            path = raw[len("local:"):]
            label = os.path.basename(path.rstrip("/"))
            targets.append((label, path))
        else:
            local = args.local if (idx == 0 and args.local) else None
            targets.append((raw, local))

    if not targets:
        parser.print_help()
        return 1

    do_compare = args.compare or len(targets) > 1

    fingerprints: Dict[str, ModelFingerprint] = {}
    for mid, local_dir in targets:
        try:
            fingerprints[mid] = build_fingerprint(
                model_id=mid,
                local_path=local_dir,
                fx_trace=args.fx_trace,
                hook_capture=args.hook_capture,
                quant_path=args.quant_path,
                fast=args.fast,
                timeout=args.timeout,
                trust_remote_code=args.trust_remote_code,
                layer_sig=args.layer_sig,
            )
        except Exception as exc:
            print(f"ERROR analyzing {mid}: {exc}", file=sys.stderr)
            fingerprints[mid] = ModelFingerprint(model_id=mid, source="error")

    coverage_matrix: dict = {}
    if do_compare and len(fingerprints) >= 2:
        valid = {k: v for k, v in fingerprints.items() if v.source != "error"}
        ids = list(valid.keys())
        for id_a, id_b in combinations(ids, 2):
            pair_key = f"{id_a}|{id_b}"
            print(f"  Comparing: {pair_key}", file=sys.stderr)
            coverage_matrix[pair_key] = compute_coverage(valid[id_a], valid[id_b])

    mf_plan: Optional[dict] = None
    if args.multi_fidelity and coverage_matrix:
        valid = {k: v for k, v in fingerprints.items() if v.source != "error"}
        mf_plan = build_multi_fidelity_plan(valid, coverage_matrix)

    strategy_summary = {
        pair: {
            "verdict": c.get("substitution_verdict"),
            "level": c.get("test_strategy", {}).get("level"),
            "scope": c.get("test_strategy", {}).get("test_scope"),
            "isomorphism": c.get("isomorphism"),
            "coverage_rate": c.get("coverage_rate"),
        }
        for pair, c in coverage_matrix.items()
    }

    output_doc: dict = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": {mid: fp_to_dict(fp) for mid, fp in fingerprints.items()},
        "test_strategy_summary": strategy_summary,
        "coverage_matrix": coverage_matrix,
    }
    if mf_plan:
        output_doc["multi_fidelity_plan"] = mf_plan

    color = not args.no_color
    if args.output == "json":
        text = format_json(output_doc)
    elif args.output == "table":
        text = format_table(output_doc, color=color)
    else:
        text = format_markdown(output_doc)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
        print(f"Saved to: {args.save}", file=sys.stderr)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
