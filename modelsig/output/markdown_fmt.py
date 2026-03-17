"""Markdown output formatter."""
from __future__ import annotations


def format_markdown(result: dict) -> str:
    lines: list = []
    lines.append(f"# {result['tool']} Analysis Report")
    lines.append(f"\n**Version:** {result['version']}  **Timestamp:** {result['timestamp']}\n")

    lines.append("## Models\n")
    for mid, mdata in result["models"].items():
        fp = mdata.get("arch_fingerprint", {})
        dr = mdata.get("dimension_ratios", {})
        lines.append(f"### `{mid}`\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in fp.items():
            if v is not None:
                lines.append(f"| {k} | {v} |")
        for k, v in dr.items():
            lines.append(f"| {k} | {v} |")
        lines.append(f"| kv_cache_shape_pattern | {mdata.get('kv_cache_shape_pattern','')} |")
        lines.append(f"| abstract_tensor_keys | {len(mdata.get('static_weight_signature',{}))} |")
        lines.append(f"| source | {mdata.get('source','')} |")
        lines.append("")
        lines.append(f"**Op types:** {', '.join(f'`{x}`' for x in mdata.get('op_types',[]))}")
        lines.append(f"\n**Layer types:** {', '.join(f'`{x}`' for x in mdata.get('layer_types',[]))}\n")

        if mdata.get("quant_path_signature"):
            lines.append("**QuantPathSignature:**\n")
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            for k, v in mdata["quant_path_signature"].items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

    cov = result.get("coverage_matrix", {})
    if cov:
        lines.append("## Coverage Matrix\n")
        lines.append("| Pair | Isomorphism | Coverage | P1 | P2 | P3 | Verdict |")
        lines.append("|------|-------------|----------|----|----|----|---------|")
        for pair, cv in cov.items():
            p1 = "✓" if cv.get("phase1_normalized_match") else "✗"
            p2 = "✓" if cv.get("phase2_substructure_match") else "✗"
            p3 = "✓" if cv.get("phase3_algebraic_consistent") else "✗"
            cov_rate = f"{cv.get('coverage_rate', 0):.2%}"
            iso = cv.get("isomorphism", "?")
            verdict = cv.get("substitution_verdict", "?")
            lines.append(f"| `{pair}` | **{iso}** | {cov_rate} | {p1} | {p2} | {p3} | **{verdict}** |")
        lines.append("")

    if cov:
        _BADGE = {"RECOMMENDED": "🟢", "PARTIAL": "🟡", "NOT_RECOMMENDED": "🔴"}
        lines.append("## Test Strategy Summary\n")
        lines.append("| Pair | Verdict | Strategy | Coverage | Scope |")
        lines.append("|------|---------|----------|----------|-------|")
        for pair, cv in cov.items():
            strat = cv.get("test_strategy", {})
            level = strat.get("level", "?")
            badge = _BADGE.get(level, "")
            scope = strat.get("test_scope", strat.get("scope", ""))
            verdict = cv.get("substitution_verdict", "?")
            rate = f"{cv.get('coverage_rate', 0):.1%}"
            lines.append(f"| `{pair}` | **{verdict}** | {badge} **{level}** | {rate} | {scope} |")
        lines.append("")
        for pair, cv in cov.items():
            strat = cv.get("test_strategy", {})
            lines.append(f"**`{pair}`** — {strat.get('description','')}")
            lines.append(f"\n> {cv.get('recommendation','')}\n")

        lines.append("### Pair Details\n")
        for pair, cv in cov.items():
            lines.append(f"#### `{pair}`\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k in ["isomorphism", "substitution_verdict", "a_covers_b", "b_covers_a",
                      "coverage_rate", "operator_coverage_rate", "layer_type_coverage",
                      "shape_compatibility", "can_substitute", "shape_ratios_all_uniform",
                      "structural_coverage"]:
                v = cv.get(k)
                if v is not None:
                    lines.append(f"| {k} | {v} |")
            lines.append("")
            if cv.get("non_uniform_shape_keys"):
                lines.append(f"**Non-uniform shape keys:** {cv['non_uniform_shape_keys']}\n")
            if cv.get("missing_highlevel_ops"):
                lines.append(f"**Missing ops in A:** {cv['missing_highlevel_ops']}\n")

    mfp = result.get("multi_fidelity_plan", {})
    if mfp:
        lines.append("## Multi-Fidelity Test Plan\n")
        for key, label in [
            ("level1_structure", "Level 1 — Structure / Conversion (Cheapest)"),
            ("level2_numerical", "Level 2 — Numerical Validation"),
            ("level3_runtime",   "Level 3 — Runtime Characterisation"),
            ("level4_canary",    "Level 4 — Large Model Canary"),
        ]:
            lines.append(f"### {label}\n")
            for item in mfp.get(key, []):
                lines.append(f"- **`{item['model']}`**: {item['reason']}")
                lines.append(f"  - Focus: {', '.join(item['test_focus'])}")
            lines.append("")

    return "\n".join(lines)
