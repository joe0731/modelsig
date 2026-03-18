┌─────────────────────────────────────────────────────────────────────────────┐                  
│                         modelsig CLI Entry Point                              │            
│                    analyze.py :: main()                                      │                 
│                                                                              │                 
│  args: -m MODEL_ID [...]  --compare  --multi-fidelity  --fast               │
│         --layer-sig (default)  --no-layer-sig  --trust-remote-code          │
│         --output [json|table|markdown]  --save FILE  --token  --no-color    │                  
└───────────────────────────────┬─────────────────────────────────────────────┘                  
                                │  for each MODEL_ID                                             
                                ▼                                                                
┌───────────────────────────────────────────────────────────────────────────────────────────┐    
│                     signature/fingerprint.py :: build_fingerprint()                        │   
│                                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐ │    
│  │  STEP 1  CONFIG LOAD   parsers/config.py :: load_config()                            │ │
│  │                                                                                      │ │    
│  │   ┌─ Try ──────────────────────────────────────────────────────────────────────────┐ │ │    
│  │   │  transformers.AutoConfig.from_pretrained()   ← preferred, handles custom archs │ │ │                                                                                                                                                                                                                                                                                                               
│  │   └────────────────────────────────────────────────────────────────────────────────┘ │ │   
│  │   ┌─ Fallback ─────────────────────────────────────────────────────────────────────┐ │ │
│  │   │  hf/client.py :: hf_load_json_file()  →  raw config.json download             │ │ │     
│  │   │    + _flatten_config()  alias normalization (n_embd→hidden_size, etc.)         │ │ │
│  │   │    + sub-config merge (text_config / llm_config / language_config)             │ │ │
│  │   └────────────────────────────────────────────────────────────────────────────────┘ │ │ 
│  └──────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  STEP 2  FORMAT DETECTION                                                            │ │ 
│  │                                                                                      │ │
│  │   --fast ──────────────────────────────────────────────────────► config_only        │ │  
│  │                                                                                      │ │
│  │   else → hf/client.py :: hf_model_files()                                           │ │   
│  │            ├── huggingface_hub.HfApi.model_info().siblings                          │ │  
│  │            └── fallback: GET /api/models/{id}                                       │ │ 
│  │                                                                                      │ │ 
│  │          .onnx exists AND no .safetensors ──────────────────────► ONNX              │ │ 
│  │          .safetensors exists (default) ────────────────────────► SAFETENSORS        │ │  
│  └──────────────────────────────────────────────────────────────────────────────────────┘ │ 
│                                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐ │ 
│  │  STEP 3  TENSOR METADATA FETCH                                                       │ │ 
│  │                                                                                      │ │ 
│  │  ┌── SAFETENSORS ─────────────────────────────────────────────────────────────────┐ │ │  
│  │  │  parsers/safetensors.py :: collect_raw_tensors()                               │ │ │  
│  │  │                                                                                 │ │ │
│  │  │   discover_shards_remote()                                                      │ │ │
│  │  │     └─ fetch model.safetensors.index.json  (weight_map → shard filenames)      │ │ │  
│  │  │                                                                                 │ │ │
│  │  │   for each shard:                                                               │ │ │ 
│  │  │     parse_remote_header(url)                                                    │ │ │
│  │  │       ├─ HTTP GET Range: bytes=0-7  →  read header_len (uint64 LE)             │ │ │  
│  │  │       └─ HTTP GET Range: bytes=8-(7+header_len)  →  parse JSON                 │ │ │ 
│  │  │            { "tensor_name": {"dtype":"BF16","shape":[4096,1024]}, ... }         │ │ │ 
│  │  │                                                                                 │ │ │
│  │  │   ZERO FULL WEIGHT DOWNLOAD  ──  only header bytes per shard                   │ │ │ 
│  │  └─────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                      │ │
│  │  ┌── ONNX ──────────────────────────────────────────────────────────────────────── ┐ │ │
│  │  │  onnx/collector.py :: collect_raw_tensors_onnx()                                │ │ │
│  │  │                                                                                  │ │ │
│  │  │   onnx/selector.py :: select_primary_onnx()                                     │ │ │
│  │  │     └─ pick smallest model_quantized.onnx / model.onnx / decoder*.onnx          │ │ │ 
│  │  │                                                                                  │ │ │
│  │  │   size check (HEAD request)  >50MB → fallback to config_only                    │ │ │
│  │  │   download .onnx file                                                            │ │ │
│  │  │                                                                                  │ │ │   
│  │  │   ┌─ if onnx lib available ──────────────────────────────────────────────────┐  │ │ │
│  │  │   │  onnx/parser.py :: parse_model_bytes_lib()                               │  │ │ │    
│  │  │   │    onnx.load(load_external_data=False)  →  graph.initializer + graph.node│  │ │ │
│  │  │   └──────────────────────────────────────────────────────────────────────────┘  │ │ │ 
│  │  │   ┌─ fallback ────────────────────────────────────────────────────────────────┐ │ │ │ 
│  │  │   │  onnx/parser.py :: parse_model_bytes_fallback()                           │ │ │ │
│  │  │   │    hand-rolled protobuf varint parser (no dependencies)                   │ │ │ │ 
│  │  │   └───────────────────────────────────────────────────────────────────────────┘ │ │ │ 
│  │  │                                                                                  │ │ │
│  │  │   onnx/ops.py :: onnx_op_types_to_canonical()                                   │ │ │
│  │  │     ONNX opset names → modelsig canonical names                                 │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────── ┘ │ │
│  │                                                                                      │ │
│  │  ┌── CONFIG_ONLY (--fast) ────────────────────────────────────────────────────────┐ │ │     
│  │  │  _synthetic_sig_from_config()  ← reconstruct tensor shapes from config dims    │ │ │     
│  │  │    embed_tokens, q/k/v/o_proj, gate/up/down_proj, layernorm, lm_head           │ │ │ 
│  │  └────────────────────────────────────────────────────────────────────────────────┘ │ │  
│  └──────────────────────────────────────────────────────────────────────────────────────┘ │ 
│                                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  STEP 4  SIGNATURE CONSTRUCTION   (from tensor_meta + config)                        │ │
│  │                                                                                      │ │ 
│  │  signature/static.py :: build_static_weight_signature()                             │ │  
│  │    ├─ norm_key()         "layers.0.q_proj" → "layers.N.q_proj"  (collapse indices) │ │   
│  │    ├─ infer_layer_type() rule-based: AttentionLayer/FFN_SwiGLU/MoELayer/RMSNorm    │ │  
│  │    └─ param_count()      ∏(shape dims) per abstract key                             │ │
│  │                                                                                      │ │    
│  │  signature/arch.py :: build_arch_fingerprint()                                      │ │     
│  │    ├─ extract _ARCH_FIELDS from config                                              │ │
│  │    ├─ derive head_dim = hidden_size / num_attention_heads                           │ │     
│  │    └─ detect is_moe  (config keys + tensor name regex _MOE_PATTERNS)               │ │
│  │                                                                                      │ │    
│  │  signature/arch.py :: build_kv_cache_shape_pattern()                                │ │
│  │    → "[batch, {num_kv_heads}, seq_len, {head_dim}]"                                 │ │
│  │                                                                                      │ │
│  │  signature/arch.py :: compute_dimension_ratios()                                    │ │     
│  │    → ffn_expansion = intermediate_size / hidden_size                                │ │
│  │    → gqa_ratio     = num_attention_heads / num_key_value_heads                     │ │
│  │                                                                                      │ │
│  │  signature/template.py :: build_template_signature()                                │ │
│  │    → extract per-layer submodule key patterns for phase-2 matching                  │ │
│  │                                                                                      │ │    
│  │  _infer_op_types()          regex over tensor names → canonical op list             │ │
│  │  _infer_unique_ops_highlevel() → {Attention, SwiGLU/FFN, MoE, Norm, Embedding, …}  │ │      
│  └──────────────────────────────────────────────────────────────────────────────────────┘ │    
│                                                                                            │   
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐ │    
│  │  STEP 5  (optional --layer-sig)                                                      │ │    
│  │  torch/layer_sig.py :: collect_layer_signatures()                                   │ │     
│  │    ├─ AutoConfig.from_pretrained()                                                  │ │     
│  │    ├─ AutoModelForCausalLM.from_config()  on torch.device("meta")  ← no weights    │ │      
│  │    ├─ register forward hooks on all named_modules                                   │ │     
│  │    ├─ run dummy forward (zeros [1,8])                                               │ │
│  │    └─ collect {module_type, norm_path, input:[{dtype,shape}], output:[{dtype,shape}]}│ │    
│  └──────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                            │   
│  → returns ModelFingerprint dataclass                                                      │   
└───────────────────────────────────────────────────────────────────────────────────────────┘                                                                                                                                                                                           
                                │                                                                
                                │  (if --compare or len(models) >= 2)

┌───────────────────────────────────────────────────────────────────────────────────────────┐
│            comparison/coverage.py :: compute_coverage(fp_a, fp_b)                         │
│                                                                                            │
│   ┌─ Phase 1 ─────────────────────────────────────────────────────────────────────────┐   │
│   │  phases.py :: phase1_match()                                                       │   │
│   │    key set overlap: len(A∩B) / max(|A|,|B|)  ≥ 0.80  ?                           │   │
│   └────────────────────────────────────────────────────────────────────────────────────┘   │
│   ┌─ Phase 2 ─────────────────────────────────────────────────────────────────────────┐   │
│   │  phases.py :: phase2_substructure_match()                                          │   │
│   │    check q_proj/k_proj/v_proj/o_proj  gate_proj/up_proj/down_proj  layernorm      │   │
│   │    presence must match between models                                               │   │
│   └────────────────────────────────────────────────────────────────────────────────────┘   │
│   ┌─ Phase 3 ─────────────────────────────────────────────────────────────────────────┐   │
│   │  phases.py :: phase3_algebraic_check()                                             │   │
│   │    dimension scaling ratios (hidden/intermediate/head_dim) uniform ≤ 20% spread?  │   │
│   │    GQA ratio difference ≤ 0.5 ?                                                    │   │
│   └────────────────────────────────────────────────────────────────────────────────────┘   │
│   ┌─ Isomorphism verdict ─────────────────────────────────────────────────────────────┐   │
│   │  phases.py :: determine_isomorphism()                                              │   │
│   │    is_moe mismatch          → DIFFERENT_ARCH                                       │   │
│   │    model_type mismatch      → DIFFERENT_ARCH                                       │   │
│   │    phase1 fail              → DIFFERENT_ARCH                                       │   │
│   │    phase2 fail              → DIFFERENT_ARCH                                       │   │
│   │    phase1+2 pass, phase3 ✓  → ISOMORPHIC                                          │   │
│   │    phase1+2 pass, phase3 ✗  → SCALE_ONLY                                          │   │
│   └────────────────────────────────────────────────────────────────────────────────────┘   │
│   ┌─ Shape ratio analysis ────────────────────────────────────────────────────────────┐   │
│   │  ratios.py :: analyze_shape_ratios()                                               │   │
│   │    per common key: large[i]/small[i] ratios  → uniform if spread ≤ 5%             │   │
│   └────────────────────────────────────────────────────────────────────────────────────┘   │
│   ┌─ Quant transferability ───────────────────────────────────────────────────────────┐   │
│   │  quant_transfer.py :: estimate_quant_transferability()                             │   │
│   │    struct_sim_score   ISOMORPHIC=1.0 / SCALE_ONLY=0.80 / DIFFERENT_ARCH=0.20     │   │
│   │    op_hist_sim        cosine similarity of op frequency vectors                   │   │
│   │    layer_type_hist_sim Jaccard of layer type sets                                 │   │
│   │    moe_correction     1.0 / 0.95(same MoE) / 0.90(cross-type)                   │   │
│   │    arch_risk_factors  hidden ratio ≥4x, GQA mismatch, FFN expansion drift, RoPE  │   │
│   │    → estimated_transferability (weighted composite, 0-1)                          │   │
│   │    → confidence: HIGH / MEDIUM / LOW                                               │   │
│   │    → recommended_methods: GPTQ/AWQ/mixed-precision/expert-aware                  │   │
│   └────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                            │
│   → coverage_matrix { pair_key → coverage_result_dict }                                   │
└───────────────────────────────────────────────────────────────────────────────────────────┘
                                │
                                │  (if --multi-fidelity)
                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│         comparison/multifidelity.py :: build_multi_fidelity_plan()                         │
│                                                                                            │
│   size_score = layers × hidden_size  (×4 if MoE)                                          │
│   group by ISOMORPHIC / SCALE_ONLY pairs                                                   │
│                                                                                            │
│   L1  smallest in group  → structure / conversion / dtype check                           │
│   L2  mid-size           → layerwise cosine similarity, perplexity on calibration set     │
│   L3  mid-size           → prefill latency, decode throughput, KV cache eviction          │
│   L4  largest in group   → memory peak, MoE routing correctness / TP/PP                   │
└───────────────────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                               OUTPUT FORMATTERS                                            │
│                                                                                            │
│   --output json  ──►  output/json_fmt.py :: fp_to_dict() + format_json()                  │
│   --output table ──►  output/table_fmt.py :: format_table()   (ANSI color)                │
│   --output markdown ► output/markdown_fmt.py :: format_markdown()                         │
│                                                                                            │
│   --save FILE  →  write to disk                                                            │
│   (default)    →  stdout                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────┘

---
Key Path Summary

User Input
  └─► load_config           Network: Download config.json (~5KB)
        └─► hf_model_files  Network: List file metadata
              ├─ safetensors: parse_remote_header × N shards   Network: Only read header bytes (few KB per shard)
              ├─ onnx:        Download .onnx file (≤50MB)      Network: Full file download
              └─ fast:        _synthetic_sig_from_config        In-memory only calculation
                    │
              build_static_weight_signature   ← Layer type/shape/dtype indexing
              build_arch_fingerprint          ← Architecture params + MoE detection
              compute_dimension_ratios        ← ffn_expansion / gqa_ratio
              build_template_signature        ← Submodule key patterns
              [torch meta device forward]     ← default on; --no-layer-sig to skip
                    │
              ModelFingerprint  ◄─────────────── Data carrier for all analyses
                    │
              compute_coverage (pairwise)
                ├─ 3-phase isomorphism check
                ├─ shape ratio uniformity analysis
                └─ quantization transferability estimation
                    │
              build_multi_fidelity_plan  (Optional)
                    │
              format & output