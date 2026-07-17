"""Human-readable rendering of a RoutingDecision for logs / CLI / debugging."""

from app.models.schemas import RoutingDecision


def format_decision_summary(decision: RoutingDecision) -> str:
    lines = []
    rec = decision.decision_record
    lines.append("=== Neural Gateway Decision ===")
    lines.append(f"Decision ID: {rec.get('decision_id', 'N/A')}")
    lines.append(f"Status: {str(rec.get('status', 'unknown')).upper()}")
    lines.append(f"Timestamp: {rec.get('timestamp_utc', 'N/A')}")
    lines.append(f"Routing latency: {float(rec.get('elapsed_ms', 0)):.1f} ms")
    lines.append("")

    ts = rec.get("task_summary", {})
    lines.append("-- Task Analysis --")
    lines.append(f"  Primary family: {ts.get('primary_family', 'N/A')}")
    lines.append(f"  Domain: {ts.get('domain', 'N/A')}")
    lines.append(f"  Risk tier: {ts.get('risk_tier', 'N/A')}")
    lines.append(f"  Complexity: {ts.get('complexity', 'N/A')}")
    lines.append(f"  Workflow profile: {ts.get('workflow_profile', 'N/A')}")
    lines.append(f"  Stages: {ts.get('required_stages', [])}")
    lines.append(f"  Parser confidence: {ts.get('parser_confidence', 0):.2f}")
    if ts.get("conflict_flags"):
        lines.append(f" \u26a0 Conflicts: {ts['conflict_flags']}")
    if ts.get("inferred_topics"):
        lines.append(f" Inferred topics: {ts['inferred_topics']}")
    if ts.get("detected_languages"):
        lines.append(f" Languages: {ts['detected_languages']}")
    lines.append("")

    if decision.abstain:
        lines.append("!! DECISION: ABSTAIN")
        if rec.get("deny_reason"):
            lines.append(f"   Deny reason: {rec['deny_reason']}")
        if decision.escalate_to_human:
            lines.append("!! ESCALATION: Human review required")
        lines.append("")
    elif decision.selected_plan:
        plan = decision.selected_plan
        lines.append("-- Selected Plan --")
        lines.append(f"  Plan ID: {plan.plan_id}")
        lines.append(f"  Plan type: {plan.plan_type}")
        if plan.plan_type == "single_model":
            lines.append(f"  Primary model: {plan.selected_model}")
        else:
            for sr in plan.stage_routes:
                lines.append(f"    Stage {sr.stage_id} [{sr.stage_name}]: {sr.selected_model}")
        lines.append(f"  Fallbacks: {plan.fallback_models}")
        lines.append(f"  Verifiers: {plan.verifier_models}")
        lines.append(f"  Profile used: {plan.profile_used}")
        lines.append(f"  Expected quality: {plan.expected_quality:.3f}")
        lines.append(f"  Expected latency: {plan.expected_latency_ms:.0f} ms")
        lines.append(f"  Expected cost: ${plan.expected_cost_usd:.5f}")
        lines.append(f"  Confidence: {float(plan.confidence or 0):.3f}")
        lines.append(f"  Confidence margin: {float(plan.confidence_margin or 0):.3f}")
        lines.append("")
        if decision.escalate_to_human:
            lines.append("!! ESCALATION: Human review recommended")
            lines.append("")
        qb = plan.explanation.get("quality_breakdown", {})
        if qb:
            lines.append("-- Quality Breakdown --")
            for k, v in qb.items():
                try:
                    lines.append(f"  {k}: {float(v):.3f}")
                except (TypeError, ValueError):
                    lines.append(f"  {k}: {v}")
            lines.append("")
        sla = plan.explanation.get("sla_check", {})
        if sla:
            lines.append("-- SLA Check --")
            lines.append(f"  Within latency SLA: {'PASS' if sla.get('within_latency_sla') else 'FAIL'}")
            lines.append(f"  Within cost SLA: {'PASS' if sla.get('within_cost_sla') else 'FAIL'}")
            lines.append("")

    pt = rec.get("pipeline_trace", {})
    if pt:
        lines.append("-- Pipeline Trace --")
        lines.append(f"  Registry models: {pt.get('registry_models', 0)}")
        lines.append(f"  After feasibility: {pt.get('feasible_after_filter', 0)}")
        lines.append(f"  After policy: {pt.get('after_policy', 0)}")
        lines.append(f"  After Pareto: {pt.get('after_pareto', 0)}")
        lines.append(f"  Policy notes: {pt.get('policy_notes', [])}")
        lines.append("")

    conf = rec.get("confidence", {})
    if conf:
        lines.append("-- Confidence --")
        lines.append(f"  Top confidence: {conf.get('top_confidence', 0):.3f}")
        lines.append(f"  Margin: {conf.get('margin', 0):.3f}")
        lines.append("")

    lines.append(f"Reproducibility hash: {rec.get('reproducibility_hash', 'N/A')}")
    lines.append("=================================")
    return "\n".join(lines)
