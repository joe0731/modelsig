# modelsig

**Model Signature Analyzer** — compare LLM architectures without downloading weights.

`modelsig` extracts a multi-layer structural fingerprint from any HuggingFace model and
tells you whether two models are architecturally isomorphic — meaning the smaller one
is a valid proxy for testing the larger one.

---

## Why modelsig?

Testing inference engines (vLLM, TensorRT-LLM, SGLang, etc.) against every large model
is prohibitively expensive. `modelsig` answers:

> *"Can I test correctness of Qwen3-72B using Qwen3-7B instead?"*

It does this by comparing structural fingerprints — shape ratios, operator sets, KV cache
patterns, and layer topology — without ever downloading a single weight tensor.

---

## How It Works

### Zero-Weight-Download

For **safetensors** models, only the file header is fetched via HTTP Range requests
(2 × ~20 bytes). No weights are transferred.

For **ONNX** models, only the `.onnx` graph file is downloaded (typically < 5 MB).
The paired `.onnx_data` weight file (which can be GBs) is never touched.

### 5-Layer Signature System

| Layer | What it captures | Source |
|-------|-----------------|--------|
| **L1** Static weight signature | Per-tensor `{abstract_key → shape, dtype, layer_type}` — layer indices normalized to `.N.` | safetensors header / ONNX initializers |
| **L2** Architecture fingerprint | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `head_dim`, MoE config | `config.json` via AutoConfig |
| **L3** Op type set | Canonical operator vocabulary: `aten/mm`, `attention`, `rms_norm`, `rope`, `silu`, `topk/router` … | tensor key patterns / ONNX opset |
| **L4** KV cache shape pattern | `[batch, num_kv_heads, seq_len, head_dim]` | derived from L2 |
| **L5** Hook shapes *(optional)* | Per-module I/O shapes from a forward pass on meta device | torch forward hooks |

### 3-Phase Isomorphism Comparison

```
Phase 1 — Key coverage    : normalized key set overlap ≥ 80%
Phase 2 — Substructure    : attention / FFN / norm submodules match
Phase 3 — Algebraic scale : hidden_size / intermediate_size / head_dim ratios uniform within 20%
```

Result: `ISOMORPHIC` / `SCALE_ONLY` / `DIFFERENT_ARCH`

### Substitution Verdict

| Verdict | Meaning |
|---------|---------|
| `FULL_SUBSTITUTE` | All 3 phases pass + shape ratios uniform + layer_type_coverage ≥ 95% |
| `PARTIAL_SUBSTITUTE` | Phase 1+2 pass or op coverage ≥ 80% |
| `NO_SUBSTITUTE` | Different arch, MoE vs Dense mismatch, or key divergence |

### Multi-Fidelity Test Plan (4 levels)

```
L1 Structure    — cheapest: model loading, tensor shapes, dtype validation
L2 Numerical    — cosine similarity, perplexity on calibration set
L3 Runtime      — prefill latency, decode throughput, KV cache eviction
L4 Canary       — large/MoE model: peak memory, TP/PP correctness
```

---

## Installation

```bash
pip install huggingface_hub onnx requests transformers
# clone and use directly (no package install needed)
git clone https://github.com/joe0731/modelsig
cd modelsig
```

**Required:**
- `huggingface_hub` — model file listing, downloads, auth
- `requests` — HTTP Range fetching

**Optional (richer analysis):**
- `onnx` — ONNX graph parsing (falls back to built-in protobuf parser)
- `transformers` — AutoConfig canonical config normalization + FX trace + hook capture
- `torch` — FX symbolic trace and forward hook capture

---

## Usage

### Basic — analyze a single model

```bash
python -m modelsig.analyze Qwen/Qwen3-7B --output table
```

```
==============================================================================
  modelsig v2.0  |  2026-03-17T10:00:00Z
==============================================================================

   Model: Qwen/Qwen3-7B
  type                   qwen3
  hidden_size            3584
  num_hidden_layers      28
  num_attention_heads    28  (kv: 8)
  intermediate_size      18944
  head_dim               128
  is_moe                 False
  ffn_expansion          5.285714
  gqa_ratio              3.5
  kv_cache_pattern       [batch, 8, seq_len, 128]
  op_types               aten/mm, attention, embedding, rms_norm, rope, silu, swiglu
  layer_types            AttentionLayer, EmbeddingLayer, FFN_SwiGLU, LMHead, RMSNorm
  abstract_keys          14
  source                 safetensors
```

### Compare models (proxy-testing decision)

```bash
python -m modelsig.analyze Qwen/Qwen3-7B Qwen/Qwen3-72B --compare --output table
```

### Full analysis with multi-fidelity plan

```bash
python -m modelsig.analyze \
    Qwen/Qwen3-7B Qwen/Qwen3-30B-A3B Qwen/Qwen3-235B-A22B \
    --compare --multi-fidelity --output markdown --save report.md
```

### ONNX model

```bash
python -m modelsig.analyze onnx-community/Qwen3-4B-ONNX --output json
```

### Config-only fast mode (no safetensors/ONNX fetch, instantaneous)

```bash
python -m modelsig.analyze Qwen/Qwen3-235B-A22B --fast --output table
```

### Local model directory

```bash
python -m modelsig.analyze local:/path/to/model --output json
python -m modelsig.analyze local:/path/to/7b local:/path/to/72b --compare
```

### Private / gated models

```bash
python -m modelsig.analyze org/private-model --token hf_xxx
# or: export HF_TOKEN=hf_xxx
```

### Save report

```bash
python -m modelsig.analyze Qwen/Qwen3-7B Qwen/Qwen3-72B \
    --compare --output markdown --save report.md
```

---

## Scenario Examples

### Scenario 1 — Inference Engine Regression Testing

**Problem:** You want to validate a new vLLM kernel for Qwen3-72B but CI is limited to A10G GPUs (24 GB VRAM).

```bash
python -m modelsig.analyze Qwen/Qwen3-7B Qwen/Qwen3-72B --compare --output table
```

**Expected result:** `ISOMORPHIC / FULL_SUBSTITUTE` — same GQA pattern, same op set, uniform scaling. You can run full functional tests on 7B and gate the 72B behind a nightly canary run.

---

### Scenario 2 — MoE vs Dense Compatibility Check

**Problem:** Does Qwen3-30B-A3B (MoE) behave like a drop-in proxy for Qwen3-235B-A22B?

```bash
python -m modelsig.analyze Qwen/Qwen3-30B-A3B Qwen/Qwen3-235B-A22B \
    --compare --multi-fidelity --output markdown
```

Both are MoE models from the same family → `ISOMORPHIC`. The multi-fidelity plan shows:
- L1: use 30B-A3B for structure/conversion tests
- L2: numerical validation on 30B
- L4: 235B-A22B as canary for routing correctness and peak memory

---

### Scenario 3 — Cross-Family Sanity Check

**Problem:** Can Llama-3.1-8B proxy-test a Mistral-7B?

```bash
python -m modelsig.analyze meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-v0.1 \
    --compare --output json
```

Both are dense GQA decoders with the same op set → `ISOMORPHIC / FULL_SUBSTITUTE`. Despite different model_type labels, the structural fingerprint matches.

---

### Scenario 4 — ONNX Runtime Compatibility

**Problem:** You converted GPT-2 to ONNX and want to verify the ONNX version matches the torch version structurally.

```bash
python -m modelsig.analyze openai-community/gpt2 onnx-community/gpt2 --compare --output table
```

The ONNX version (from `onnx-community/gpt2`) is parsed from the `.onnx` graph file.
The safetensors version is parsed from the header.
Both share the same abstract key set → `ISOMORPHIC`.

---

### Scenario 5 — Quantized Model Compatibility

**Problem:** Will `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (quantized to FP4)
behave the same as the BF16 variant?

```bash
python -m modelsig.analyze \
    nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --compare --fast --output table
```

Both share the same architecture (120B MoE). `--fast` uses config-only mode to avoid
downloading large safetensors headers. Result: `ISOMORPHIC` — same layer topology,
only dtype differs.

---

### Scenario 6 — Batch Analysis with QuantPathSignature

**Problem:** Prepare a quantization validation plan for a fleet of Qwen models.

```bash
python -m modelsig.analyze \
    Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-7B \
    --compare --quant-path --output json --save qwen3_fleet.json
```

The `quant_path_signature` block for each model documents:
`arch_template` (gqa_decoder), `kv_cache_dtype`, `group_size`, `scale_scheme` —
feeding directly into a quantization config generator.

---

## CLI Reference

```
python -m modelsig.analyze MODEL_ID [MODEL_ID ...] [OPTIONS]

Arguments:
  MODEL_ID          HF model ID (e.g. Qwen/Qwen3-7B) or local:PATH

Options:
  --output          json | table | markdown  (default: json)
  --compare         Compute pairwise coverage for all model pairs
  --save FILE       Save output to file
  --fast            Config-only mode — no safetensors/ONNX download
  --quant-path      Include QuantPathSignature block
  --multi-fidelity  Include 4-level multi-fidelity test plan
  --no-fx-trace     Disable FX symbolic trace (on by default)
  --no-hook-capture Disable forward-hook capture (on by default)
  --token TOKEN     HF Hub token for private/gated models
  --timeout SEC     HTTP timeout (default: 30)
  --no-color        Disable ANSI colors in table output
```

---

## Module Structure

```
modelsig/
├── analyze.py              CLI entry point (~190 lines)
├── constants.py            Shared constants: TOOL_NAME, _OP_RULES, _LAYER_TYPE_RULES, …
│
├── hf/
│   └── client.py           HF Hub client: token management, HTTP GET + backoff,
│                           model_info().siblings, hf_hub_download
│
├── parsers/
│   ├── safetensors.py      HTTP Range header fetch + local shard discovery
│   └── config.py           AutoConfig.from_pretrained() + _flatten_config() aliases
│
├── onnx/
│   ├── ops.py              _ONNX_DTYPE map, _ONNX_OP_MAP, canonical op mapping
│   ├── parser.py           onnx.load(load_external_data=False) + protobuf fallback
│   ├── selector.py         Primary .onnx file selection heuristics
│   └── collector.py        Orchestrates HF download → parse pipeline
│
├── torch/
│   ├── fx_trace.py         FX symbolic trace on meta device (lazy torch import)
│   └── hooks.py            Forward-hook I/O shape capture (lazy torch import)
│
├── signature/
│   ├── static.py           L1: build_static_weight_signature, norm_key, norm_dtype
│   ├── arch.py             L2: build_arch_fingerprint, KV cache pattern, dim ratios
│   ├── quant.py            QuantPathSignature builder
│   ├── template.py         Per-layer canonical submodule template (for phase-2)
│   └── fingerprint.py      ModelFingerprint dataclass + build_fingerprint orchestrator
│
├── comparison/
│   ├── phases.py           Phase 1/2/3 isomorphism tests
│   ├── ratios.py           Shape ratio uniformity analysis (grok)
│   ├── coverage.py         Unified compute_coverage + test strategy verdict
│   └── multifidelity.py    4-level multi-fidelity test plan builder
│
└── output/
    ├── colors.py           ANSI color helpers
    ├── json_fmt.py         JSON formatter + fp_to_dict
    ├── table_fmt.py        ANSI table formatter
    └── markdown_fmt.py     Markdown report formatter
```

---

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Zero weight download** | HTTP Range (safetensors), graph-only .onnx, config-only fast path |
| **Framework-driven parsing** | `AutoConfig.from_pretrained()` for config normalization; `onnx.load()` for graph parsing |
| **Graceful degradation** | Every heavy dependency is optional — falls back to built-in parsers |
| **Architecture-agnostic** | Works on dense decoders, GQA models, MoE, vision-language, speech, classification |
| **Single CLI, composable API** | Import any module independently or use the unified CLI |

---

## Contributing

All logic is in the `modelsig/` package. Each subdirectory has a single responsibility.
Tests live in `tests/` and cover unit + integration scenarios.

```bash
pip install pytest
pytest tests/ -v
```

Weekly validation against the full model zoo runs via GitHub Actions (`.github/workflows/weekly-validation.yml`).
