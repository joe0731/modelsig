"""Unit tests for modelsig.comparison modules."""
import pytest
from modelsig.signature.fingerprint import ModelFingerprint
from modelsig.comparison.phases import (
    phase1_match, phase2_substructure_match,
    phase3_algebraic_check, determine_isomorphism,
)
from modelsig.comparison.ratios import _is_uniform, analyze_shape_ratios
from modelsig.comparison.coverage import compute_coverage, _test_strategy
from modelsig.comparison.multifidelity import build_multi_fidelity_plan, _size_score
from modelsig.comparison.quant_transfer import estimate_quant_transferability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fp(model_id, hidden=4096, layers=32, heads=32, kv=8, inter=14336,
             is_moe=False, source="safetensors", model_type="llama"):
    """Create a ModelFingerprint with a realistic Qwen/Llama-style structure."""
    head_dim = hidden // heads
    # Build a minimal but realistic static_weight_signature
    sig = {}
    for k in [
        f"model.layers.N.self_attn.q_proj.weight",
        f"model.layers.N.self_attn.k_proj.weight",
        f"model.layers.N.self_attn.v_proj.weight",
        f"model.layers.N.self_attn.o_proj.weight",
        f"model.layers.N.mlp.gate_proj.weight",
        f"model.layers.N.mlp.up_proj.weight",
        f"model.layers.N.mlp.down_proj.weight",
        f"model.layers.N.input_layernorm.weight",
        f"model.layers.N.post_attention_layernorm.weight",
        f"model.embed_tokens.weight",
        f"lm_head.weight",
    ]:
        if "q_proj" in k:
            shape = [heads * head_dim, hidden]
        elif "k_proj" in k or "v_proj" in k:
            shape = [kv * head_dim, hidden]
        elif "o_proj" in k:
            shape = [hidden, heads * head_dim]
        elif "gate_proj" in k or "up_proj" in k:
            shape = [inter, hidden]
        elif "down_proj" in k:
            shape = [hidden, inter]
        elif "layernorm" in k:
            shape = [hidden]
        else:
            shape = [32000, hidden]
        sig[k] = {"representative_shape": shape, "dtype": "BF16",
                   "layer_type": "AttentionLayer", "count": layers, "param_count": 1}

    # Build template_signature
    tmpl = {}
    for k, v in sig.items():
        if "self_attn" in k or "mlp" in k or "layernorm" in k:
            tmpl[k] = {"shape": v["representative_shape"], "dtype": "BF16"}

    arch = {
        "model_type": model_type,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "intermediate_size": inter,
        "head_dim": head_dim,
        "is_moe": is_moe,
    }
    ltypes = ["AttentionLayer", "FFN_SwiGLU", "RMSNorm", "EmbeddingLayer", "LMHead"]
    dr = {
        "ffn_expansion": round(inter / hidden, 6),
        "gqa_ratio": round(heads / kv, 6),
    }
    return ModelFingerprint(
        model_id=model_id,
        static_weight_signature=sig,
        arch_fingerprint=arch,
        op_types=["aten/mm", "attention", "rms_norm", "rope", "silu", "swiglu",
                  "scaled_dot_product_attention"],
        kv_cache_shape_pattern=f"[batch, {kv}, seq_len, {head_dim}]",
        unique_ops_highlevel={"Attention", "SwiGLU/FFN", "Norm", "Embedding", "LMHead"},
        layer_types=ltypes,
        dimension_ratios=dr,
        template_signature=tmpl,
        source=source,
    )


# ---------------------------------------------------------------------------
# phases.py
# ---------------------------------------------------------------------------

class TestPhase1Match:
    def test_identical_keys_pass(self):
        sig = {"a": {}, "b": {}, "c": {}}
        ok, only_a, only_b = phase1_match(sig, sig)
        assert ok is True
        assert only_a == []
        assert only_b == []

    def test_high_overlap_passes(self):
        a = {k: {} for k in range(10)}
        b = {k: {} for k in range(9)}  # 9/10 = 90% > 80%
        ok, _, _ = phase1_match(a, b)
        assert ok is True

    def test_low_overlap_fails(self):
        a = {k: {} for k in range(10)}
        b = {k + 100: {} for k in range(10)}  # 0% overlap
        ok, _, _ = phase1_match(a, b)
        assert ok is False

    def test_empty_sigs_fail(self):
        ok, _, _ = phase1_match({}, {"a": {}})
        assert ok is False


class TestPhase2SubstructureMatch:
    def test_matching_templates(self):
        tmpl = {
            "self_attn.q_proj": {}, "self_attn.k_proj": {},
            "mlp.gate_proj": {}, "input_layernorm": {},
        }
        assert phase2_substructure_match(tmpl, tmpl) is True

    def test_attn_mismatch(self):
        # Keys need 3 parts (e.g. parent.subkey.weight) so split[-2] yields the subkey
        a = {"self_attn.q_proj.weight": {}, "self_attn.k_proj.weight": {},
             "mlp.gate_proj.weight": {}, "input_layernorm.weight": {}}
        b = {"mlp.gate_proj.weight": {}, "input_layernorm.weight": {}}  # no attention
        assert phase2_substructure_match(a, b) is False

    def test_empty_templates(self):
        # Both empty → all checks trivially equal (True)
        assert phase2_substructure_match({}, {}) is True


class TestPhase3AlgebraicCheck:
    def test_uniform_scaling_passes(self):
        fp_a = _make_fp("a", hidden=2048, inter=8192)
        fp_b = _make_fp("b", hidden=4096, inter=16384)  # exact 2x
        ok, scaling = phase3_algebraic_check(fp_a, fp_b)
        assert ok is True
        assert "hidden_size" in scaling

    def test_non_uniform_scaling_fails(self):
        fp_a = _make_fp("a", hidden=2048, inter=8192)
        fp_b = _make_fp("b", hidden=8192, inter=8300)  # hidden 4x but inter ~1x
        ok, _ = phase3_algebraic_check(fp_a, fp_b)
        assert ok is False

    def test_gqa_ratio_mismatch_fails(self):
        fp_a = _make_fp("a", heads=32, kv=8)   # gqa_ratio=4
        fp_b = _make_fp("b", heads=32, kv=32)  # gqa_ratio=1
        ok, _ = phase3_algebraic_check(fp_a, fp_b)
        assert ok is False


class TestDetermineIsomorphism:
    def test_moe_vs_dense(self):
        fp_a = _make_fp("a", is_moe=True)
        fp_b = _make_fp("b", is_moe=False)
        iso, _ = determine_isomorphism(fp_a, fp_b, p1=True, p2=True, p3=True)
        assert iso == "DIFFERENT_ARCH"

    def test_different_model_type(self):
        fp_a = _make_fp("a", model_type="llama")
        fp_b = _make_fp("b", model_type="mistral")
        iso, _ = determine_isomorphism(fp_a, fp_b, p1=True, p2=True, p3=True)
        assert iso == "DIFFERENT_ARCH"

    def test_all_phases_pass(self):
        fp_a = _make_fp("a")
        fp_b = _make_fp("b")
        iso, msg = determine_isomorphism(fp_a, fp_b, p1=True, p2=True, p3=True)
        assert iso == "ISOMORPHIC"
        assert "proxy" in msg.lower()

    def test_phase1_fails(self):
        fp_a = _make_fp("a")
        fp_b = _make_fp("b")
        iso, _ = determine_isomorphism(fp_a, fp_b, p1=False, p2=True, p3=True)
        assert iso == "DIFFERENT_ARCH"

    def test_scale_only(self):
        fp_a = _make_fp("a")
        fp_b = _make_fp("b")
        iso, _ = determine_isomorphism(fp_a, fp_b, p1=True, p2=True, p3=False)
        assert iso == "SCALE_ONLY"


# ---------------------------------------------------------------------------
# ratios.py
# ---------------------------------------------------------------------------

class TestIsUniform:
    def test_identical_ratios(self):
        assert _is_uniform([2.0, 2.0, 2.0]) is True

    def test_near_uniform(self):
        assert _is_uniform([2.0, 2.02, 1.98]) is True  # within 5% tolerance

    def test_non_uniform(self):
        assert _is_uniform([2.0, 4.0]) is False

    def test_empty(self):
        assert _is_uniform([]) is True


class TestAnalyzeShapeRatios:
    def test_uniform_scaling(self):
        sig_a = {"model.layers.N.q_proj.weight": {"representative_shape": [1024, 512]}}
        sig_b = {"model.layers.N.q_proj.weight": {"representative_shape": [2048, 1024]}}
        result = analyze_shape_ratios(sig_a, sig_b)
        assert "model.layers.N.q_proj.weight" in result
        entry = result["model.layers.N.q_proj.weight"]
        assert entry["uniform"] is True
        assert entry["ratios"] == [2.0, 2.0]

    def test_non_uniform(self):
        sig_a = {"k": {"representative_shape": [1024, 512]}}
        sig_b = {"k": {"representative_shape": [4096, 512]}}  # 4x vs 1x
        result = analyze_shape_ratios(sig_a, sig_b)
        assert result["k"]["uniform"] is False

    def test_rank_mismatch_skipped(self):
        sig_a = {"k": {"representative_shape": [1024]}}
        sig_b = {"k": {"representative_shape": [1024, 512]}}
        result = analyze_shape_ratios(sig_a, sig_b)
        assert "k" not in result

    def test_no_common_keys(self):
        sig_a = {"a": {"representative_shape": [512]}}
        sig_b = {"b": {"representative_shape": [512]}}
        assert analyze_shape_ratios(sig_a, sig_b) == {}


# ---------------------------------------------------------------------------
# coverage.py
# ---------------------------------------------------------------------------

class TestTestStrategy:
    def test_recommended(self):
        s = _test_strategy(1.0, 1.0, True)
        assert s["level"] == "RECOMMENDED"

    def test_partial(self):
        s = _test_strategy(0.80, 0.75, False)
        assert s["level"] == "PARTIAL"

    def test_not_recommended(self):
        s = _test_strategy(0.50, 0.50, False)
        assert s["level"] == "NOT_RECOMMENDED"


class TestComputeCoverage:
    def test_isomorphic_models(self):
        fp_a = _make_fp("a/small", hidden=2048, layers=24, inter=8192)
        fp_b = _make_fp("b/large", hidden=4096, layers=32, inter=16384)
        cov = compute_coverage(fp_a, fp_b)
        # Same model_type ("llama"), same op set → should be high coverage
        assert cov["isomorphism"] in ("ISOMORPHIC", "SCALE_ONLY")
        assert cov["layer_type_coverage"] >= 0.9
        assert cov["coverage_rate"] > 0.8

    def test_different_arch(self):
        fp_a = _make_fp("a", is_moe=True)
        fp_b = _make_fp("b", is_moe=False)
        cov = compute_coverage(fp_a, fp_b)
        assert cov["isomorphism"] == "DIFFERENT_ARCH"
        assert cov["substitution_verdict"] == "NO_SUBSTITUTE"

    def test_output_keys_present(self):
        fp_a = _make_fp("a")
        fp_b = _make_fp("b")
        cov = compute_coverage(fp_a, fp_b)
        for key in [
            "isomorphism", "substitution_verdict", "coverage_rate",
            "operator_coverage_rate", "layer_type_coverage", "shape_compatibility",
            "test_strategy", "recommendation",
        ]:
            assert key in cov, f"Missing key: {key}"

    def test_same_model_full_coverage(self):
        fp = _make_fp("a")
        cov = compute_coverage(fp, fp)
        assert cov["coverage_rate"] == 1.0
        assert cov["layer_type_coverage"] == 1.0
        assert cov["operator_coverage_rate"] == 1.0


# ---------------------------------------------------------------------------
# multifidelity.py
# ---------------------------------------------------------------------------

class TestBuildMultiFidelityPlan:
    def test_single_model(self):
        fp = _make_fp("a")
        plan = build_multi_fidelity_plan({"a": fp}, {})
        assert "level1_structure" in plan
        assert len(plan["level1_structure"]) == 1

    def test_two_isomorphic_models(self):
        fp_a = _make_fp("a/small", hidden=2048)
        fp_b = _make_fp("b/large", hidden=4096)
        cov = compute_coverage(fp_a, fp_b)
        cm = {"a/small|b/large": cov}
        plan = build_multi_fidelity_plan({"a/small": fp_a, "b/large": fp_b}, cm)
        # Level 1 uses smallest; level 4 uses largest
        l1_models = [x["model"] for x in plan["level1_structure"]]
        l4_models = [x["model"] for x in plan["level4_canary"]]
        assert len(l1_models) > 0
        assert len(l4_models) > 0

    def test_size_score(self):
        fp_small = _make_fp("small", hidden=1024, layers=12)
        fp_large = _make_fp("large", hidden=4096, layers=32)
        assert _size_score(fp_small) < _size_score(fp_large)

    def test_moe_size_score_higher(self):
        fp_dense = _make_fp("dense", hidden=4096, layers=32, is_moe=False)
        fp_moe   = _make_fp("moe",   hidden=4096, layers=32, is_moe=True)
        assert _size_score(fp_moe) > _size_score(fp_dense)


# ---------------------------------------------------------------------------
# quant_transfer.py
# ---------------------------------------------------------------------------

class TestEstimateQuantTransferability:
    def test_isomorphic_same_family_high_score(self):
        fp_a = _make_fp("small", hidden=2048, layers=24, inter=8192)
        fp_b = _make_fp("large", hidden=4096, layers=32, inter=16384)
        result = estimate_quant_transferability(
            fp_a, fp_b,
            isomorphism="ISOMORPHIC",
            layer_type_coverage=1.0,
            shape_all_uniform=True,
        )
        assert result["estimated_transferability"] >= 0.80
        assert result["confidence"] == "HIGH"

    def test_different_arch_low_confidence(self):
        fp_a = _make_fp("a", is_moe=False)
        fp_b = _make_fp("b", is_moe=True)
        result = estimate_quant_transferability(
            fp_a, fp_b,
            isomorphism="DIFFERENT_ARCH",
            layer_type_coverage=0.5,
            shape_all_uniform=False,
        )
        # DIFFERENT_ARCH gets struct_score=0.20; confidence is always LOW
        assert result["struct_sim_score"] == 0.20
        assert result["confidence"] == "LOW"
        # score is still dragged down vs ISOMORPHIC same-family
        iso_result = estimate_quant_transferability(
            fp_a, fp_b,
            isomorphism="ISOMORPHIC",
            layer_type_coverage=1.0,
            shape_all_uniform=True,
        )
        assert result["estimated_transferability"] < iso_result["estimated_transferability"]

    def test_moe_correction_applied(self):
        fp_dense = _make_fp("dense", is_moe=False)
        fp_moe = _make_fp("moe", is_moe=True)
        result = estimate_quant_transferability(
            fp_dense, fp_moe,
            isomorphism="SCALE_ONLY",
            layer_type_coverage=0.9,
            shape_all_uniform=True,
        )
        assert result["moe_correction"] == 0.90

    def test_same_moe_applies_095_correction(self):
        fp_a = _make_fp("moe_small", is_moe=True)
        fp_b = _make_fp("moe_large", is_moe=True)
        result = estimate_quant_transferability(
            fp_a, fp_b,
            isomorphism="ISOMORPHIC",
            layer_type_coverage=1.0,
            shape_all_uniform=True,
        )
        assert result["moe_correction"] == 0.95

    def test_large_hidden_size_ratio_adds_risk(self):
        fp_a = _make_fp("small", hidden=1024)
        fp_b = _make_fp("large", hidden=8192)  # 8x ratio
        result = estimate_quant_transferability(
            fp_a, fp_b,
            isomorphism="ISOMORPHIC",
            layer_type_coverage=1.0,
            shape_all_uniform=True,
        )
        assert len(result["arch_risk_factors"]) >= 1
        assert any("hidden_size" in r for r in result["arch_risk_factors"])

    def test_output_keys_present(self):
        fp_a = _make_fp("a")
        fp_b = _make_fp("b")
        result = estimate_quant_transferability(
            fp_a, fp_b,
            isomorphism="ISOMORPHIC",
            layer_type_coverage=1.0,
            shape_all_uniform=True,
        )
        for key in [
            "struct_sim_score", "op_hist_sim", "layer_type_hist_sim",
            "moe_correction", "arch_risk_factors",
            "estimated_transferability", "confidence",
            "recommended_methods", "caveats",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_coverage_matrix_includes_quant_transfer(self):
        fp_a = _make_fp("a/small", hidden=2048)
        fp_b = _make_fp("b/large", hidden=4096)
        cov = compute_coverage(fp_a, fp_b)
        assert "quant_transfer" in cov
        qt = cov["quant_transfer"]
        assert "estimated_transferability" in qt
        assert 0.0 <= qt["estimated_transferability"] <= 1.0
