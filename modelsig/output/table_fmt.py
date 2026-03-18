"""Table output formatter."""
from __future__ import annotations
from .colors import c as _c


def format_table(result: dict, color: bool = True) -> str:
    SEP = "=" * 78
    sep = "-" * 78
    lines: list = []
    lines.append(SEP)
    lines.append(_c(f"  {result['tool']} v{result['version']}", "bold", color) +
                 f"  |  {result['timestamp']}")
    lines.append(SEP)

    for mid, mdata in result["models"].items():
        lines.append(f"\n{'Model':>8}: {_c(mid, 'cyan', color)}")
        fp = mdata.get("arch_fingerprint", {})
        dr = mdata.get("dimension_ratios", {})
        lines.append(f"  {'type':<22} {fp.get('model_type','?')}")
        lines.append(f"  {'hidden_size':<22} {fp.get('hidden_size','?')}")
        lines.append(f"  {'num_hidden_layers':<22} {fp.get('num_hidden_layers','?')}")
        lines.append(f"  {'num_attention_heads':<22} {fp.get('num_attention_heads','?')}  "
                     f"(kv: {fp.get('num_key_value_heads','?')})")
        lines.append(f"  {'intermediate_size':<22} {fp.get('intermediate_size','?')}")
        lines.append(f"  {'head_dim':<22} {fp.get('head_dim','?')}")
        lines.append(f"  {'is_moe':<22} {fp.get('is_moe','?')}")
        if fp.get("is_moe"):
            lines.append(f"  {'num_experts':<22} {fp.get('num_experts','?')}  "
                         f"per_tok={fp.get('num_experts_per_tok','?')}")
        lines.append(f"  {'ffn_expansion':<22} {dr.get('ffn_expansion','?')}")
        lines.append(f"  {'gqa_ratio':<22} {dr.get('gqa_ratio','?')}")
        lines.append(f"  {'kv_cache_pattern':<22} {mdata.get('kv_cache_shape_pattern','?')}")
        lines.append(f"  {'op_types':<22} {', '.join(mdata.get('op_types',[]))}")
        lines.append(f"  {'layer_types':<22} {', '.join(mdata.get('layer_types',[]))}")
        lines.append(f"  {'abstract_keys':<22} {len(mdata.get('static_weight_signature',{}))}")
        lines.append(f"  {'source':<22} {mdata.get('source','?')}")

    cov = result.get("coverage_matrix", {})
    if cov:
        lines.append(f"\n{SEP}")
        lines.append("  Coverage Matrix")
        lines.append(SEP)
        header = f"  {'Pair':<44} {'Isomorphism':<14} {'Coverage':>9}  Verdict"
        lines.append(header)
        lines.append("  " + sep)
        _VERDICT_COLOR = {
            "FULL_SUBSTITUTE": "green", "PARTIAL_SUBSTITUTE": "yellow", "NO_SUBSTITUTE": "red"
        }
        for pair, cv in cov.items():
            iso = cv.get("isomorphism", "?")
            verdict = cv.get("substitution_verdict", "?")
            rate = f"{cv.get('coverage_rate', 0):.1%}"
            phases = ("P1" if cv.get("phase1_normalized_match") else "--") + " " + \
                     ("P2" if cv.get("phase2_substructure_match") else "--") + " " + \
                     ("P3" if cv.get("phase3_algebraic_consistent") else "--")
            verdict_str = _c(verdict, _VERDICT_COLOR.get(verdict, ""), color)
            lines.append(f"  {pair:<44} {iso:<14} {rate:>9}  {verdict_str}  [{phases}]")
        lines.append("")

    if cov:
        _STRAT_COLOR = {"RECOMMENDED": "green", "PARTIAL": "yellow", "NOT_RECOMMENDED": "red"}
        lines.append(f"{SEP}")
        lines.append(_c("  Test Strategy Summary", "bold", color))
        lines.append(SEP)
        for pair, cv in cov.items():
            strat = cv.get("test_strategy", {})
            level = strat.get("level", "?")
            level_str = _c(f"[{level}]", _STRAT_COLOR.get(level, ""), color)
            lines.append(f"  {pair}")
            lines.append(f"    {level_str}  {strat.get('scope', strat.get('test_scope',''))}")
            lines.append(f"    {strat.get('description','')}")
            lines.append(f"    Recommendation : {cv.get('recommendation','')}")
            qt = cv.get("quant_transfer", {})
            if qt:
                score = qt.get("estimated_transferability", 0)
                conf = qt.get("confidence", "?")
                methods = ", ".join(qt.get("recommended_methods", []))
                lines.append(f"    Quant transfer : {score:.2f} [{conf}]  →  {methods}")
            if cv.get("non_uniform_shape_keys"):
                lines.append(f"    Non-uniform keys: {cv['non_uniform_shape_keys']}")
            lines.append("")

    mfp = result.get("multi_fidelity_plan", {})
    if mfp:
        lines.append(SEP)
        lines.append("  Multi-Fidelity Test Plan")
        lines.append(SEP)
        for key, label in [
            ("level1_structure", "L1 Structure/Conversion (cheapest)"),
            ("level2_numerical", "L2 Numerical (cosine/perplexity)"),
            ("level3_runtime",   "L3 Runtime (prefill/decode/KV)"),
            ("level4_canary",    "L4 Large Model Canary (memory/TP/PP)"),
        ]:
            items = mfp.get(key, [])
            lines.append(f"\n  {_c(label, 'bold', color)}")
            for item in items:
                lines.append(f"    - {item['model']}")
                lines.append(f"      {item['reason']}")
                lines.append(f"      Focus: {', '.join(item['test_focus'])}")
        lines.append("")

    return "\n".join(lines)
