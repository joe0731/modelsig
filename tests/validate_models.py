#!/usr/bin/env python3
"""
tests/validate_models.py — Integration validation against real HF models.

Usage:
    python tests/validate_models.py                  # validate all models
    python tests/validate_models.py --group safetensors
    python tests/validate_models.py --group onnx
    python tests/validate_models.py --save results/run.json
    python tests/validate_models.py --fail-fast

Exit code: 0 = all pass, 1 = some failures.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

SAFETENSORS_MODELS = [
    # Small / medium — full safetensors header fetch
    ("Tesslate/OmniCoder-9B",                                "safetensors"),
    ("LocoreMind/LocoTrainer-4B",                            "safetensors"),
    ("Qwen/Qwen3.5-9B",                                      "safetensors"),
    ("Qwen/Qwen3.5-4B",                                      "safetensors"),
    ("Qwen/Qwen3.5-0.8B",                                    "safetensors"),
    ("Qwen/Qwen3.5-27B",                                     "safetensors"),
    ("Qwen/Qwen3.5-35B-A3B",                                 "safetensors"),
    ("RekaAI/reka-edge-2603",                                "safetensors"),
    ("ibm-granite/granite-4.0-1b-speech",                   "safetensors"),
    ("microsoft/bitnet-b1.58-2B-4T",                        "safetensors"),
    ("sentence-transformers/all-MiniLM-L6-v2",              "safetensors"),
    ("google/translategemma-4b-it",                         "safetensors"),
    ("bharatgenai/Param2-17B-A2.4B-Thinking",               "safetensors"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",               "safetensors"),
    ("Qwen/Qwen2.5-7B-Instruct",                            "safetensors"),
    ("meta-llama/Llama-3.1-8B-Instruct",                    "safetensors"),
    ("Nanbeige/Nanbeige4.1-3B",                             "safetensors"),
    ("openai/gpt-oss-20b",                                  "safetensors"),
    ("miromind-ai/MiroThinker-1.7-mini",                    "safetensors"),
    ("sarvamai/sarvam-30b",                                  "safetensors"),
    # Large — config-only fast mode to avoid header fetch cost
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",       "fast"),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",      "fast"),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",        "fast"),
    ("MiniMaxAI/MiniMax-M2.5",                              "fast"),
    ("sarvamai/sarvam-105b",                                 "fast"),
    ("miromind-ai/MiroThinker-1.7",                         "fast"),
    ("zai-org/GLM-5",                                       "fast"),
    ("Qwen/Qwen3.5-397B-A17B",                              "fast"),
    ("deepseek-ai/DeepSeek-V3.2",                           "fast"),
    ("moonshotai/Kimi-K2.5",                                "fast"),
    ("Qwen/Qwen3-Coder-Next",                               "fast"),
]

ONNX_MODELS = [
    ("onnx-community/NVIDIA-Nemotron-3-Nano-4B-BF16-ONNX",               "onnx"),
    ("onnx-community/codet5-base-ONNX",                                    "onnx"),
    ("onnx-community/Jan-code-4b-ONNX",                                    "onnx"),
    ("onnx-community/Josiefied-Qwen3.5-0.8B-gabliterated-v1-ONNX",       "onnx"),
    ("onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX",           "onnx"),
    ("onnx-community/chinese-roberta-wwm-ext-ONNX",                       "onnx"),
    ("onnx-community/bert-base-multilingual-cased-ner-hrl-ONNX",          "onnx"),
    ("onnx-community/Qwen3-Reranker-0.6B-ONNX",                          "onnx"),
    ("onnx-community/Qwen2.5-0.5B-Instruct-abliterated-v3-ONNX",         "onnx"),
    ("onnx-community/news_title_classification-indobert-base-p1-ONNX",   "onnx"),
    ("onnx-community/Olmo-Hybrid-Instruct-SFT-7B-ONNX",                  "onnx"),
    ("onnx-community/Olmo-Hybrid-Instruct-DPO-7B-ONNX",                  "onnx"),
    ("onnx-community/Olmo-Hybrid-Think-SFT-7B-ONNX",                     "onnx"),
    ("onnx-community/ai-image-detection-ONNX",                            "onnx"),
    ("onnx-community/ai-image-detect-distilled-ONNX",                     "onnx"),
    ("onnx-community/ai-source-detector-ONNX",                            "onnx"),
    ("onnx-community/SMOGY-Ai-images-detector-ONNX",                     "onnx"),
    ("onnx-community/Qwen3.5-4B-ONNX",                                    "onnx"),
    ("onnx-community/Qwen3.5-2B-ONNX",                                    "onnx"),
    ("onnx-community/Qwen3.5-0.8B-ONNX",                                  "onnx"),
    ("onnx-community/tmr-ai-text-detector-ONNX",                         "onnx"),
    ("onnx-community/autotrain-vehicle-classification-test-ONNX",        "onnx"),
    ("onnx-community/LFM2-24B-A2B-ONNX",                                 "onnx"),
    ("onnx-community/Qwen3-4B-VL-ONNX",                                  "onnx"),
    ("onnx-community/Qwen3-VL-2B-Instruct-ONNX",                        "onnx"),
    ("onnx-community/Qwen2.5-VL-3B-Instruct-ONNX",                      "onnx"),
    ("onnx-community/granite-4.0-1b-speech-ONNX",                       "onnx"),
    ("onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX",               "onnx"),
]

ALL_MODELS = SAFETENSORS_MODELS + ONNX_MODELS


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

def _run_one(model_id: str, mode: str, timeout: int = 120) -> dict:
    """Run modelsig.analyze on one model and parse the JSON result."""
    cmd = [
        sys.executable, "-m", "modelsig.analyze",
        model_id,
        "--output", "json",
        "--no-fx-trace", "--no-hook-capture",
    ]
    if mode == "fast":
        cmd.append("--fast")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = round(time.time() - t0, 1)

        if proc.returncode != 0:
            return {
                "model_id": model_id,
                "mode": mode,
                "status": "FAIL",
                "error": f"exit code {proc.returncode}",
                "stderr": proc.stderr[-2000:],
                "elapsed_s": elapsed,
            }

        try:
            doc = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            return {
                "model_id": model_id,
                "mode": mode,
                "status": "FAIL",
                "error": f"JSON parse error: {e}",
                "stderr": proc.stderr[-2000:],
                "stdout_tail": proc.stdout[-500:],
                "elapsed_s": elapsed,
            }

        model_data = doc.get("models", {}).get(model_id, {})
        arch = model_data.get("arch_fingerprint", {})
        source = model_data.get("source", "unknown")
        tensors = len(model_data.get("static_weight_signature", {}))

        # PASS criteria:
        #   source is not "error"  AND
        #   at least one of: model_type, hidden_size, tensors > 0, or onnx_op_types present
        has_arch_data = (
            arch.get("model_type") or
            arch.get("hidden_size") or
            tensors > 0 or
            bool(model_data.get("onnx_op_types"))
        )
        if source == "error" or not has_arch_data:
            return {
                "model_id": model_id,
                "mode": mode,
                "status": "FAIL",
                "error": "source=error or no arch data",
                "stderr": proc.stderr[-2000:],
                "elapsed_s": elapsed,
            }

        return {
            "model_id": model_id,
            "mode": mode,
            "status": "PASS",
            "source": source,
            "model_type": arch.get("model_type", "?"),
            "hidden_size": arch.get("hidden_size", "?"),
            "num_hidden_layers": arch.get("num_hidden_layers", "?"),
            "num_attention_heads": arch.get("num_attention_heads", "?"),
            "num_key_value_heads": arch.get("num_key_value_heads", "?"),
            "is_moe": arch.get("is_moe", False),
            "tensors": tensors,
            "elapsed_s": elapsed,
            "warnings": [line for line in proc.stderr.splitlines()
                         if "[warn]" in line or "[onnx]" in line],
        }

    except subprocess.TimeoutExpired:
        return {
            "model_id": model_id,
            "mode": mode,
            "status": "FAIL",
            "error": f"timeout after {timeout}s",
            "elapsed_s": timeout,
        }
    except Exception as exc:
        return {
            "model_id": model_id,
            "mode": mode,
            "status": "FAIL",
            "error": str(exc),
            "elapsed_s": round(time.time() - t0, 1),
        }


def run_validation(models, workers=4, fail_fast=False):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, mid, mode): (mid, mode) for mid, mode in models}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            status = res["status"]
            mid = res["model_id"]
            if status == "PASS":
                src = res.get("source", "?")
                mt = res.get("model_type", "?")
                hs = res.get("hidden_size", "?")
                t = res.get("tensors", "?")
                print(f"  [PASS] {mid:<60}  {src:<14}  {mt:<18}  hidden={hs}  tensors={t}")
            else:
                err = res.get("error", "")
                print(f"  [FAIL] {mid:<60}  {err}", file=sys.stderr)
                if fail_fast:
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
    return results


def print_summary(results):
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    total = len(results)
    print(f"\n{'='*70}")
    print(f"  Results: {len(passed)}/{total} PASS  |  {len(failed)} FAIL")
    print(f"{'='*70}")
    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  - {r['model_id']}")
            print(f"      error  : {r.get('error', 'unknown')}")
            if r.get("stderr"):
                # show last 3 lines of stderr
                lines = r["stderr"].strip().splitlines()[-3:]
                for line in lines:
                    print(f"      stderr : {line}")
    return len(failed)


def main():
    p = argparse.ArgumentParser(description="Validate modelsig against real HF models")
    p.add_argument("--group", choices=["safetensors", "onnx", "all"], default="all")
    p.add_argument("--save", metavar="FILE", default=None, help="Save JSON results to file")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    p.add_argument("--fail-fast", action="store_true", dest="fail_fast",
                   help="Stop after first failure")
    p.add_argument("--timeout", type=int, default=120, help="Per-model timeout in seconds")
    args = p.parse_args()

    if args.group == "safetensors":
        models = SAFETENSORS_MODELS
    elif args.group == "onnx":
        models = ONNX_MODELS
    else:
        models = ALL_MODELS

    print(f"  Running validation: {len(models)} models  (workers={args.workers})\n")
    t0 = time.time()
    results = run_validation(models, workers=args.workers, fail_fast=args.fail_fast)
    elapsed = round(time.time() - t0, 1)

    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
        "elapsed_s": elapsed,
        "results": sorted(results, key=lambda r: r["status"]),
    }

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to: {args.save}")

    num_failed = print_summary(results)
    return 1 if num_failed else 0


if __name__ == "__main__":
    sys.exit(main())
