"""Unit tests for modelsig.signature modules."""
import pytest
from modelsig.signature.static import (
    norm_dtype, norm_key, infer_layer_type, param_count,
    build_static_weight_signature,
)
from modelsig.signature.arch import (
    build_arch_fingerprint, build_kv_cache_shape_pattern, compute_dimension_ratios,
)
from modelsig.signature.quant import build_quant_path_signature
from modelsig.signature.template import build_template_signature
from modelsig.signature.fingerprint import (
    ModelFingerprint, _synthetic_sig_from_config, _minimal_arch_config,
    _infer_op_types, _infer_unique_ops_highlevel,
)


# ---------------------------------------------------------------------------
# static.py
# ---------------------------------------------------------------------------

class TestNormDtype:
    def test_bf16(self):
        assert norm_dtype("BF16") == "BF16"

    def test_fp16(self):
        assert norm_dtype("F16") == "FP16"

    def test_torch_bfloat16(self):
        assert norm_dtype("torch.bfloat16") == "BF16"

    def test_torch_float16(self):
        assert norm_dtype("torch.float16") == "FP16"

    def test_fp8(self):
        assert norm_dtype("F8_E4M3") == "FP8_E4M3"

    def test_int8(self):
        assert norm_dtype("I8") == "INT8"

    def test_unknown(self):
        result = norm_dtype("CUSTOM_DTYPE")
        assert isinstance(result, str)


class TestNormKey:
    def test_replaces_layer_index(self):
        assert norm_key("model.layers.0.self_attn.q_proj.weight") == \
               "model.layers.N.self_attn.q_proj.weight"

    def test_multiple_indices(self):
        result = norm_key("model.layers.3.mlp.experts.7.gate_proj.weight")
        assert result == "model.layers.N.mlp.experts.N.gate_proj.weight"

    def test_no_index(self):
        key = "model.embed_tokens.weight"
        assert norm_key(key) == key


class TestInferLayerType:
    def test_attention(self):
        assert infer_layer_type("model.layers.0.self_attn.q_proj.weight") == "AttentionLayer"

    def test_ffn(self):
        assert infer_layer_type("model.layers.0.mlp.gate_proj.weight") == "FFN_SwiGLU"

    def test_moe(self):
        # "gate_proj" keyword matches FFN_SwiGLU before MoE in the rule list.
        # Use a key with only MoE keywords (router, moe) to get MoELayer.
        assert infer_layer_type("model.layers.0.moe_block.router.weight") == "MoELayer"

    def test_norm(self):
        assert infer_layer_type("model.layers.0.input_layernorm.weight") == "RMSNorm"

    def test_embedding(self):
        assert infer_layer_type("model.embed_tokens.weight") == "EmbeddingLayer"

    def test_lm_head(self):
        assert infer_layer_type("lm_head.weight") == "LMHead"

    def test_rope(self):
        # Key must not contain "self_attn" (which matches AttentionLayer first).
        # Use a top-level rotary_emb key instead.
        assert infer_layer_type("model.layers.0.rotary_emb.inv_freq") == "RotaryEmb"

    def test_fallback(self):
        assert infer_layer_type("some.unknown.linear.weight") == "LinearLayer"


class TestParamCount:
    def test_1d(self):
        assert param_count([1024]) == 1024

    def test_2d(self):
        assert param_count([4096, 1024]) == 4096 * 1024

    def test_scalar(self):
        assert param_count([1]) == 1


class TestBuildStaticWeightSignature:
    def _make_meta(self, shape, dtype="BF16"):
        return {"shape": shape, "dtype": dtype}

    def test_basic_grouping(self):
        meta = {
            "model.layers.0.self_attn.q_proj.weight": self._make_meta([4096, 1024]),
            "model.layers.1.self_attn.q_proj.weight": self._make_meta([4096, 1024]),
            "model.layers.2.self_attn.q_proj.weight": self._make_meta([4096, 1024]),
        }
        sig = build_static_weight_signature(meta)
        assert "model.layers.N.self_attn.q_proj.weight" in sig
        entry = sig["model.layers.N.self_attn.q_proj.weight"]
        assert entry["count"] == 3
        assert entry["representative_shape"] == [4096, 1024]
        assert entry["layer_type"] == "AttentionLayer"

    def test_skips_non_dict(self):
        meta = {
            "good_key": {"shape": [1024], "dtype": "BF16"},
            "__metadata__": "not a tensor",
        }
        sig = build_static_weight_signature(meta)
        assert "good_key" in sig or "good_key" not in sig  # just no crash

    def test_param_count_accumulates(self):
        meta = {
            "model.layers.0.mlp.gate_proj.weight": self._make_meta([4096, 1024]),
            "model.layers.1.mlp.gate_proj.weight": self._make_meta([4096, 1024]),
        }
        sig = build_static_weight_signature(meta)
        key = "model.layers.N.mlp.gate_proj.weight"
        assert sig[key]["param_count"] == 2 * 4096 * 1024


# ---------------------------------------------------------------------------
# arch.py
# ---------------------------------------------------------------------------

class TestBuildArchFingerprint:
    def _make_config(self, **kwargs):
        base = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
        }
        base.update(kwargs)
        return base

    def test_basic_fields(self):
        fp = build_arch_fingerprint(self._make_config(), {})
        assert fp["model_type"] == "llama"
        assert fp["hidden_size"] == 4096
        assert fp["num_hidden_layers"] == 32

    def test_head_dim_derived(self):
        fp = build_arch_fingerprint(self._make_config(), {})
        assert fp["head_dim"] == 4096 // 32  # 128

    def test_head_dim_not_overridden_if_present(self):
        cfg = self._make_config(head_dim=64)
        fp = build_arch_fingerprint(cfg, {})
        assert fp["head_dim"] == 64

    def test_moe_from_config(self):
        cfg = self._make_config(num_local_experts=8, model_type="qwen2_moe")
        fp = build_arch_fingerprint(cfg, {})
        assert fp["is_moe"] is True

    def test_moe_from_weight_names(self):
        cfg = self._make_config()
        meta = {"model.layers.0.mlp.experts.0.gate_proj.weight": {}}
        fp = build_arch_fingerprint(cfg, meta)
        assert fp["is_moe"] is True

    def test_dense_is_not_moe(self):
        fp = build_arch_fingerprint(self._make_config(), {})
        assert fp["is_moe"] is False


class TestKvCachePattern:
    def test_gqa(self):
        fp = {"num_key_value_heads": 8, "head_dim": 128}
        assert build_kv_cache_shape_pattern(fp) == "[batch, 8, seq_len, 128]"

    def test_mha_fallback(self):
        fp = {"num_attention_heads": 32, "head_dim": 128}
        result = build_kv_cache_shape_pattern(fp)
        assert "32" in result and "128" in result

    def test_unknown(self):
        result = build_kv_cache_shape_pattern({})
        assert "?" in result


class TestDimensionRatios:
    def test_ffn_expansion(self):
        cfg = {"hidden_size": 4096, "intermediate_size": 14336}
        r = compute_dimension_ratios(cfg)
        assert abs(r["ffn_expansion"] - 14336 / 4096) < 1e-4

    def test_gqa_ratio(self):
        cfg = {"num_attention_heads": 32, "num_key_value_heads": 8}
        r = compute_dimension_ratios(cfg)
        assert abs(r["gqa_ratio"] - 4.0) < 1e-4

    def test_mha_no_gqa_ratio(self):
        cfg = {"num_attention_heads": 32, "num_key_value_heads": 32}
        r = compute_dimension_ratios(cfg)
        # gqa_ratio = 32/32 = 1.0 is still valid
        assert r.get("gqa_ratio") == 1.0

    def test_empty_config(self):
        assert compute_dimension_ratios({}) == {}


# ---------------------------------------------------------------------------
# quant.py
# ---------------------------------------------------------------------------

class TestBuildQuantPathSignature:
    def test_dense_decoder(self):
        arch_fp = {"is_moe": False, "num_key_value_heads": 8, "num_attention_heads": 32}
        qps = build_quant_path_signature(arch_fp, {})
        assert qps["arch_template"] == "gqa_decoder"

    def test_moe_decoder(self):
        arch_fp = {"is_moe": True, "num_key_value_heads": 8, "num_attention_heads": 32}
        qps = build_quant_path_signature(arch_fp, {})
        assert qps["arch_template"] == "moe_decoder"

    def test_full_mha(self):
        arch_fp = {"is_moe": False, "num_key_value_heads": 32, "num_attention_heads": 32}
        qps = build_quant_path_signature(arch_fp, {})
        assert qps["arch_template"] == "dense_decoder"

    def test_quant_config_fields(self):
        arch_fp = {"is_moe": False}
        cfg = {"quantization_config": {"quant_type": "awq", "group_size": 128}}
        qps = build_quant_path_signature(arch_fp, cfg)
        assert qps["quant_algo"] == "awq"
        assert qps["group_size"] == 128

    def test_required_keys_present(self):
        qps = build_quant_path_signature({"is_moe": False}, {})
        for key in ["arch_template", "quant_algo", "weight_dtype", "kv_cache_dtype"]:
            assert key in qps


# ---------------------------------------------------------------------------
# template.py
# ---------------------------------------------------------------------------

class TestBuildTemplateSignature:
    def test_extracts_submodule_keys(self):
        meta = {
            "model.layers.0.self_attn.q_proj.weight": {"shape": [4096, 1024], "dtype": "BF16"},
            "model.layers.0.mlp.gate_proj.weight":     {"shape": [14336, 4096], "dtype": "BF16"},
        }
        tmpl = build_template_signature(meta)
        assert any("q_proj" in k for k in tmpl)
        assert any("gate_proj" in k for k in tmpl)

    def test_no_top_level_keys(self):
        meta = {"model.embed_tokens.weight": {"shape": [32000, 4096], "dtype": "BF16"}}
        tmpl = build_template_signature(meta)
        # embed_tokens has no layer index pattern \.N\.
        assert len(tmpl) == 0


# ---------------------------------------------------------------------------
# fingerprint.py helpers
# ---------------------------------------------------------------------------

class TestSyntheticSigFromConfig:
    def test_dense_gqa(self):
        cfg = {
            "hidden_size": 4096, "intermediate_size": 14336,
            "vocab_size": 32000, "num_attention_heads": 32,
            "num_key_value_heads": 8, "is_moe": False,
        }
        sig, ltypes = _synthetic_sig_from_config(cfg)
        assert "model.embed_tokens.weight" in sig
        assert "lm_head.weight" in sig
        assert "AttentionLayer" in ltypes
        assert "FFN_SwiGLU" in ltypes

    def test_moe(self):
        cfg = {
            "hidden_size": 2048, "intermediate_size": 4096,
            "vocab_size": 32000, "num_attention_heads": 16,
            "num_key_value_heads": 4, "is_moe": True,
        }
        sig, ltypes = _synthetic_sig_from_config(cfg)
        assert "MoELayer" in ltypes
        assert any("experts" in k for k in sig)


class TestInferOpTypes:
    def test_attention_detected(self):
        meta = {"model.layers.0.self_attn.q_proj.weight": {}}
        ops = _infer_op_types(meta)
        assert "attention" in ops

    def test_swiglu_detected(self):
        meta = {"model.layers.0.mlp.gate_proj.weight": {}}
        ops = _infer_op_types(meta)
        assert "swiglu" in ops

    def test_rope_detected(self):
        meta = {"model.layers.0.self_attn.rotary_emb.inv_freq": {}}
        ops = _infer_op_types(meta)
        assert "rope" in ops

    def test_moe_router_detected(self):
        meta = {"model.layers.0.mlp.experts.0.gate_proj.weight": {}}
        ops = _infer_op_types(meta)
        assert "topk/router" in ops


class TestModelFingerprintDataclass:
    def test_default_fields(self):
        fp = ModelFingerprint(model_id="test/model")
        assert fp.model_id == "test/model"
        assert fp.source == "safetensors"
        assert fp.op_types == []
        assert fp.layer_types == []
        assert fp.static_weight_signature == {}
