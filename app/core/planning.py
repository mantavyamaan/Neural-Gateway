"""
Execution Plan Generation.

Builds the complete execution strategy for a routing decision: either a
single-model plan or, for multi-stage workflows (OCR -> document QA,
audio -> summarization, etc.), a per-stage specialized plan. Also selects
verifier models independent of the primary model selection.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import ROUTER_VERSION, PARSER_VERSION, POLICY_VERSION, SCORING_VERSION, REGISTRY_VERSION
from app.core.scoring import (
    estimate_request_cost_usd,
    estimate_request_latency_ms,
    runtime_health_score,
    stage_fit,
)
from app.models.schemas import ExecutionPlan, PolicyDecision, StageRoute, TaskFeatures


def choose_verifier_models(
    all_models: List[Dict[str, Any]],
    verifier_types: List[str],
    selected_model_name: Optional[str],
    max_verifiers: int = 2,
) -> List[Dict[str, Any]]:
    """Select best verifier models for each required type, excluding the primary model."""
    if not verifier_types:
        return []
    candidates = [m for m in all_models if m["name"] != selected_model_name]
    if not candidates:
        return []
    scored = []
    for model in candidates:
        vf = model.get("verifier_fit", {})
        avg_fit = float(np.mean([vf.get(vt, 0.5) for vt in verifier_types]))
        scored.append((model, avg_fit))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:max_verifiers]]


def select_best_for_stage(
    models: List[Dict[str, Any]],
    stage_name: str,
    task: TaskFeatures,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not models:
        return None, []
    scored = []
    for m in models:
        sf = stage_fit(m, stage_name, task)
        blended = 0.80 * sf + 0.20 * runtime_health_score(m)
        scored.append((m, blended))
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]
    fallbacks = [s[0] for s in scored[1:3]]
    return best, fallbacks


def generate_multi_stage_plan(
    models: List[Dict[str, Any]],
    task: TaskFeatures,
    confidence_data: Dict[str, Any],
    verifier_models: List[Dict[str, Any]],
    policy: PolicyDecision,
    profile_name: str,
) -> Optional[ExecutionPlan]:
    if len(task.required_stages) < 2:
        return None

    n_stages = len(task.required_stages)
    in_per_stage = max(task.estimated_tokens // n_stages, 1)
    out_per_stage = max(task.estimated_output_tokens // n_stages, 1)

    stage_routes: List[StageRoute] = []
    total_latency = 0.0
    total_cost = 0.0
    stage_quality_scores: List[float] = []
    stage_utilities: List[float] = []

    for i, stage_name in enumerate(task.required_stages):
        best, fallbacks = select_best_for_stage(models, stage_name, task)
        if best is None:
            return None
        # Each stage has its own full TTFT — do NOT divide by n_stages
        s_latency = estimate_request_latency_ms(best, task)
        s_cost = estimate_request_cost_usd(best, in_per_stage, out_per_stage)
        total_latency += s_latency
        total_cost += s_cost
        stage_quality_scores.append(best["q"]["runtime_adjusted_mean"])
        stage_utilities.append(best["u"]["expected_utility"])
        stage_routes.append(StageRoute(
            stage_id=i + 1,
            stage_name=stage_name,
            selected_model=best["name"],
            fallback_models=[fb["name"] for fb in fallbacks],
            verifier_models=[v["name"] for v in verifier_models] if i == n_stages - 1 else [],
            stage_confidence=confidence_data.get("selected_confidence", confidence_data["top_confidence"]),
            expected_latency_ms=s_latency,
            expected_cost_usd=s_cost,
            explanation=f"Stage '{stage_name}' assigned to {best['name']} (best stage fit).",
        ))

    rc = task.request_constraints
    max_cost = rc.max_cost_usd if rc else None
    
    final_verifiers = []
    for v in verifier_models:
        v_latency = (v.get("ops_dynamic") or {}).get("recent_latency_ms", 200) * 0.5
        v_cost = estimate_request_cost_usd(v, out_per_stage, 300)
        
        if max_cost is not None and (total_cost + v_cost) > max_cost:
            continue
            
        total_latency += v_latency
        total_cost += v_cost
        final_verifiers.append(v)

    expected_quality = float(np.mean(stage_quality_scores)) if stage_quality_scores else 0.0
    utility = float(np.mean(stage_utilities)) if stage_utilities else 0.0

    return ExecutionPlan(
        plan_id=str(uuid.uuid4())[:12],
        plan_type="multi_stage",
        selected_model=None,
        stage_routes=stage_routes,
        fallback_models=[],
        verifier_models=[v["name"] for v in final_verifiers],
        expected_latency_ms=total_latency,
        expected_cost_usd=total_cost,
        expected_quality=expected_quality,
        confidence=confidence_data.get("selected_confidence", confidence_data.get("top_confidence", 0.0)),
        utility=utility,
        confidence_margin=confidence_data.get("selected_margin", confidence_data.get("margin", 0.0)),
        profile_used=profile_name,
        explanation={
            "plan_type": "multi_stage",
            "stage_assignments": [
                {"stage": sr.stage_name, "model": sr.selected_model} for sr in stage_routes
            ],
            "task_family": task.primary_family,
            "domain": task.domain,
            "risk_tier": task.risk_tier,
            "workflow_profile": task.workflow_profile,
            "verifiers_attached": [v["name"] for v in final_verifiers],
            "policy_notes": policy.notes,
            "note": "Per-stage specialization; sequential latency is additive.",
        },
        trace={
            "router_version": ROUTER_VERSION,
            "scoring_version": SCORING_VERSION,
            "parser_version": PARSER_VERSION,
            "policy_version": POLICY_VERSION,
            "registry_version": REGISTRY_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def generate_single_model_plan(
    model: Dict[str, Any],
    task: TaskFeatures,
    confidence_data: Dict[str, Any],
    verifier_models: List[Dict[str, Any]],
    policy: PolicyDecision,
    fallback_models: List[Dict[str, Any]],
    profile_name: str,
    cascade_strategy: Optional[str] = None,
) -> ExecutionPlan:
    from app.core.scoring import predict_output_tokens
    est_latency = estimate_request_latency_ms(model, task)
    pred_output = predict_output_tokens(task)
    est_cost = estimate_request_cost_usd(model, task.estimated_tokens, pred_output)
    stage_routes = []
    n_stages = max(len(task.required_stages), 1)

    for i, stage_name in enumerate(task.required_stages):
        stage_routes.append(StageRoute(
            stage_id=i + 1,
            stage_name=stage_name,
            selected_model=model["name"],
            fallback_models=[fb["name"] for fb in fallback_models[:2]],
            verifier_models=[v["name"] for v in verifier_models],
            stage_confidence=confidence_data.get("selected_confidence", confidence_data["top_confidence"]),
            expected_latency_ms=est_latency / n_stages,
            expected_cost_usd=est_cost / n_stages,
            explanation=f"Stage '{stage_name}' handled by {model['name']}.",
        ))

    rc = task.request_constraints
    max_cost = rc.max_cost_usd if rc else None
    
    final_verifiers = []
    for v in verifier_models:
        v_latency = (v.get("ops_dynamic") or {}).get("recent_latency_ms", 200) * 0.5
        v_cost = estimate_request_cost_usd(v, task.estimated_output_tokens, 500)
        
        if max_cost is not None and (est_cost + v_cost) > max_cost:
            continue
            
        est_latency += v_latency
        est_cost += v_cost
        final_verifiers.append(v)

    return ExecutionPlan(
        plan_id=str(uuid.uuid4())[:12],
        plan_type="cascade" if cascade_strategy else "single_model",
        selected_model=model["name"],
        stage_routes=stage_routes,
        fallback_models=[fb["name"] for fb in fallback_models[:3]],
        verifier_models=[v["name"] for v in final_verifiers],
        expected_latency_ms=est_latency,
        expected_cost_usd=est_cost,
        expected_quality=model["q"]["runtime_adjusted_mean"],
        confidence=confidence_data.get("selected_confidence", confidence_data.get("top_confidence", 0.0)),
        utility=model["u"]["expected_utility"],
        confidence_margin=confidence_data.get("selected_margin", confidence_data.get("margin", 0.0)),
        profile_used=profile_name,
        explanation={
            "primary_model": model["name"],
            "provider": model["provider"],
            "tier": model["tier"],
            "task_family": task.primary_family,
            "domain": task.domain,
            "risk_tier": task.risk_tier,
            "workflow_profile": task.workflow_profile,
            "stages": task.required_stages,
            "verifiers_attached": [v["name"] for v in final_verifiers],
            "policy_notes": policy.notes,
            "quality_breakdown": {
                "family_fit": model["q"]["family_fit"],
                "workflow_fit": model["q"]["workflow_fit"],
                "domain_fit": model["q"]["domain_fit"],
                "runtime_fit": model["q"]["runtime_fit"],
                "contextual_mean": model["q"]["contextual_mean"],
                "runtime_adjusted_mean": model["q"]["runtime_adjusted_mean"],
            },
            "sla_check": {
                "estimated_latency_ms": est_latency,
                "estimated_cost_usd": est_cost,
                "max_latency_budget_ms": task.request_constraints.max_latency_ms,
                "max_cost_budget_usd": task.request_constraints.max_cost_usd,
                "within_latency_sla": (
                    task.request_constraints.max_latency_ms is None
                    or est_latency <= task.request_constraints.max_latency_ms
                ),
                "within_cost_sla": (
                    task.request_constraints.max_cost_usd is None
                    or est_cost <= task.request_constraints.max_cost_usd
                ),
            },
        },
        trace={
            "router_version": ROUTER_VERSION,
            "parser_version": PARSER_VERSION,
            "policy_version": POLICY_VERSION,
            "scoring_version": SCORING_VERSION,
            "registry_version": REGISTRY_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "confidence_simulations": confidence_data["n_simulations"],
            "win_probabilities": confidence_data["win_probabilities"],
        },
        verification_strategy=cascade_strategy,
    )
