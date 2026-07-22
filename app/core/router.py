"""
Neural Gateway — main orchestration.

route() is the single entry point: it runs the full staged pipeline
(parse -> feasibility -> policy -> Bayesian quality -> Pareto reduction ->
utility scoring -> confidence estimation -> plan generation -> escalation
ladder) and returns a complete, auditable RoutingDecision.

record_outcome() is the learning hook: it feeds observed outcomes back
into a model's Bayesian priors so future routing improves over time.
"""

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from app.config import (
    CONFIDENCE_ABSTAIN_THRESHOLD,
    CONFIDENCE_ESCALATE_THRESHOLD,
    CONFIDENCE_HIGH_THRESHOLD,
    PARSER_ESCALATE_THRESHOLD,
    REGISTRY_VERSION,
    ROUTER_VERSION,
)
from app.core.feasibility import feasibility_filter
from app.core.planning import (
    choose_verifier_models,
    generate_multi_stage_plan,
    generate_single_model_plan,
)
from app.core.policy import apply_policy_to_models, evaluate_policy
from app.core.scoring import (
    attach_contextual_quality,
    choose_effective_profile,
    compute_utilities,
    estimate_confidence,
    pareto_frontier,
)
from app.core.semantic_parser import parse_task_request
from app.core.database import get_all_models
from app.models.schemas import RequestConstraints, RoutingDecision, TenantContext


def route(
    prompt: str,
    input_formats: Optional[List[str]] = None,
    estimated_tokens: int = 2000,
    estimated_output_tokens: int = 1200,
    artifact_hints: Optional[List[Dict[str, Any]]] = None,
    request_constraints: Optional[RequestConstraints] = None,
    tenant_context: Optional[TenantContext] = None,
    files: Optional[List[str]] = None,
    profile_name: str = "balanced",
    registry: Optional[List[Dict[str, Any]]] = None,
    shadow_model: Optional[str] = None,
) -> RoutingDecision:
    """Main entry point for the Neural Gateway. Returns a complete RoutingDecision."""
    start_time = time.time()
    decision_id = str(uuid.uuid4())
    registry = registry if registry is not None else get_all_models()

    # ---- 1-2. Parse the request / build canonical task representation ----
    task = parse_task_request(
        prompt=prompt,
        input_formats=input_formats,
        estimated_tokens=estimated_tokens,
        estimated_output_tokens=estimated_output_tokens,
        artifact_hints=artifact_hints,
        request_constraints=request_constraints,
        tenant_context=tenant_context,
        files=files,
    )

    # ---- 3. Feasibility filtering ----
    feasible, feasibility_reasons = feasibility_filter(task, registry)
    if not feasible:
        return RoutingDecision(
            selected_plan=None, abstain=True, escalate_to_human=True,
            decision_record={
                "decision_id": decision_id,
                "status": "no_feasible_models",
                "feasibility_reasons": feasibility_reasons,
                "task_summary": {
                    "primary_family": task.primary_family,
                    "domain": task.domain,
                    "required_formats": task.required_formats,
                },
                "router_version": ROUTER_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": (time.time() - start_time) * 1000,
            },
        )

    # ---- 4. Governance policy ----
    policy = evaluate_policy(task)
    if policy.must_abstain:
        return RoutingDecision(
            selected_plan=None, abstain=True, escalate_to_human=False,
            decision_record={
                "decision_id": decision_id,
                "status": "policy_abstain",
                "deny_reason": policy.deny_reason,
                "policy_notes": policy.notes,
                "router_version": ROUTER_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": (time.time() - start_time) * 1000,
            },
        )

    gated, policy_reasons = apply_policy_to_models(task, policy, feasible)
    if not gated:
        return RoutingDecision(
            selected_plan=None, abstain=True, escalate_to_human=True,
            decision_record={
                "decision_id": decision_id,
                "status": "no_models_after_policy",
                "feasibility_reasons": feasibility_reasons,
                "policy_reasons": policy_reasons,
                "policy_notes": policy.notes,
                "router_version": ROUTER_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": (time.time() - start_time) * 1000,
            },
        )

    # ---- 5. Contextual Bayesian quality ----
    enriched = attach_contextual_quality(gated, task)

    # ---- 6. Pareto reduction (with top-3 floor) ----
    frontier = pareto_frontier(enriched, eps=0.02)
    if len(frontier) < 3 and len(enriched) >= 3:
        # Use name-based lookup — avoids fragile dict identity comparison
        # (compute_utilities does model.copy() so object identity may differ)
        frontier_names = {m["name"] for m in frontier}
        remaining = [m for m in enriched if m["name"] not in frontier_names]
        remaining.sort(key=lambda m: m["q"]["runtime_adjusted_mean"], reverse=True)
        frontier.extend(remaining[:3 - len(frontier)])

    # ---- 7. Utility scoring ----
    effective_profile = choose_effective_profile(task, profile_name)
    all_scored = compute_utilities(enriched, task, effective_profile)
    frontier_names = {m["name"] for m in frontier}
    scored = [m for m in all_scored if m["name"] in frontier_names]

    # ---- 8. Confidence estimation ----
    confidence_data = estimate_confidence(scored, task, effective_profile)

    # The primary is always the highest expected-utility model. Thompson
    # sampling quantifies uncertainty around that decision; it does not use a
    # different scoring formula to choose a contradictory primary.
    if not scored:
        # All frontier models were filtered by compute_utilities — abstain conservatively
        return RoutingDecision(
            selected_plan=None, abstain=True, escalate_to_human=True,
            decision_record={
                "decision_id": decision_id,
                "status": "no_scoreable_models",
                "router_version": ROUTER_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": (time.time() - start_time) * 1000,
            },
        )
    primary_model = scored[0]
    selected_win_probability = confidence_data["win_probabilities"].get(primary_model["name"], 0.0)
    other_probabilities = [
        p for name, p in confidence_data["win_probabilities"].items()
        if name != primary_model["name"]
    ]
    selected_margin = selected_win_probability - max(other_probabilities, default=0.0)
    confidence_data["selected_model"] = primary_model["name"]
    confidence_data["selected_confidence"] = selected_win_probability
    confidence_data["selected_margin"] = selected_margin
    fallback_models = [m for m in scored if m["name"] != primary_model["name"]][:3]

    # ---- Cascade Routing ----
    is_coding = (task.primary_family == "coding" or "coding" in task.secondary_families)
    is_json = task.requires_json
    cascade_strategy = None
    if task.complexity == "low" and task.risk_tier == "low":
        cheap_candidates = ["gemini-2.5-flash", "llama-3.1-8b", "llama-3.2-3b"]
        cheap_models = [m for m in all_scored if any(c in m["name"].lower() for c in cheap_candidates)]
        if cheap_models:
            # Only set cascade when a cheap model actually exists in the registry
            if is_coding:
                cascade_strategy = "ast_execution"
            elif is_json:
                cascade_strategy = "json_schema"
            else:
                cascade_strategy = "semantic_similarity"
                
            primary_model = cheap_models[0]
            # Cascade routing: cheap model selected for low-complexity task with fallback safety net
            selected_win_probability = max(0.70, confidence_data["win_probabilities"].get(primary_model["name"], 0.7))
            confidence_data["selected_model"] = primary_model["name"]
            confidence_data["selected_confidence"] = selected_win_probability
            
            other_probs = [
                p for name, p in confidence_data["win_probabilities"].items()
                if name != primary_model["name"]
            ]
            confidence_data["selected_margin"] = selected_win_probability - max(other_probs, default=0.0)
            fallback_models = [m for m in all_scored if m["name"] != primary_model["name"]][:3]

    # ---- Verifier planning ----
    verifier_models = choose_verifier_models(
        all_models=gated,
        verifier_types=policy.require_verifier_types,
        selected_model_name=primary_model["name"],
        max_verifiers=2,
    )

    # ---- 9. Execution plan generation ----
    single_plan = generate_single_model_plan(
        model=primary_model, task=task, confidence_data=confidence_data,
        verifier_models=verifier_models, policy=policy,
        fallback_models=fallback_models, profile_name=effective_profile,
    )
    plan = single_plan
    if not task.request_constraints.must_use_single_model and len(task.required_stages) >= 2:
        multi_plan = generate_multi_stage_plan(
            models=all_scored, task=task, confidence_data=confidence_data,
            verifier_models=verifier_models, policy=policy,
            profile_name=effective_profile,
        )
        if multi_plan and multi_plan.expected_quality >= single_plan.expected_quality + 0.03:
            plan = multi_plan

    if cascade_strategy:
        plan.plan_type = "cascade"
        plan.verification_strategy = cascade_strategy
        plan.explanation["cascade_note"] = "Cascade routing triggered: using cheap model with verification."

    # A verifier or sequential multi-stage plan can push the end-to-end cost
    # and latency beyond constraints even if each individual model passed
    # feasibility. Never return an executable plan that breaks its SLA.
    rc = task.request_constraints
    plan_breaks_sla = (
        (rc.max_cost_usd is not None and plan.expected_cost_usd > rc.max_cost_usd)
        or (rc.max_latency_ms is not None and plan.expected_latency_ms > rc.max_latency_ms)
    )
    if plan_breaks_sla:
        return RoutingDecision(
            selected_plan=None,
            abstain=True,
            escalate_to_human=True,
            decision_record={
                "decision_id": decision_id,
                "status": "plan_exceeds_hard_constraints",
                "router_version": ROUTER_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": (time.time() - start_time) * 1000,
                "task_summary": {
                    "primary_family": task.primary_family,
                    "domain": task.domain,
                    "complexity": task.complexity,
                },
                "plan_constraints": {
                    "expected_cost_usd": plan.expected_cost_usd,
                    "expected_latency_ms": plan.expected_latency_ms,
                    "max_cost_usd": rc.max_cost_usd,
                    "max_latency_ms": rc.max_latency_ms,
                },
            },
        )

    # ---- 10. Abstention and escalation ladder ----
    abstain = False
    escalate = policy.must_escalate
    
    # Dynamic confidence scaling for registries:
    # When N models are evaluated, Thompson win probability dilutes to ~1/N per candidate.
    # Require winner to beat random chance (1/N) by 1.2x, capped at CONFIDENCE_ABSTAIN_THRESHOLD.
    n_candidates = max(1, len(scored))
    dynamic_threshold = min(CONFIDENCE_ABSTAIN_THRESHOLD, 1.2 / n_candidates)
    minimum_confidence = max(dynamic_threshold, task.request_constraints.min_confidence)
    
    # Low confidence flags escalation for human review/observability, but per explicit directive,
    # the router NEVER abstains due to low confidence — it always returns the best generated execution plan.
    if selected_win_probability < minimum_confidence:
        escalate = True
    elif selected_win_probability < CONFIDENCE_ESCALATE_THRESHOLD:
        escalate = True
    if task.parser_confidence < PARSER_ESCALATE_THRESHOLD:
        escalate = True
    if task.risk_tier == "high" and selected_win_probability < CONFIDENCE_HIGH_THRESHOLD:
        escalate = True

    # ---- Shadow routing hook ----
    shadow_info = None
    if shadow_model:
        shadow_candidates = [m for m in scored if m["name"] == shadow_model]
        if shadow_candidates:
            shadow_info = {
                "shadow_model": shadow_model,
                "shadow_utility": shadow_candidates[0]["u"]["expected_utility"],
                "shadow_quality": shadow_candidates[0]["q"]["runtime_adjusted_mean"],
            }

    # ---- 11. Final decision record ----
    elapsed_ms = (time.time() - start_time) * 1000
    decision_record = {
        "decision_id": decision_id,
        "status": "routed" if not abstain else "abstained",
        "router_version": ROUTER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_ms": elapsed_ms,
        "task_summary": {
            "primary_family": task.primary_family,
            "secondary_families": task.secondary_families,
            "domain": task.domain,
            "risk_tier": task.risk_tier,
            "risk_type": task.risk_type,
            "complexity": task.complexity,
            "workflow_profile": task.workflow_profile,
            "required_stages": task.required_stages,
            "expected_output": task.expected_output,
            "requires_json": task.requires_json,
            "requires_ocr": task.requires_ocr,
            "requires_web_search": task.requires_web_search,
            "requires_citations": task.requires_citations,
            "requires_verifier": task.requires_verifier,
            "parser_confidence": task.parser_confidence,
            "conflict_flags": task.conflict_flags,
            "detected_languages": task.detected_languages,
            "total_file_size_bytes": task.total_file_size_bytes,
            "inferred_topics": [a.inferred_topic for a in task.artifacts if a.inferred_topic],
            "ambiguity_score": task.ambiguity_score,
            "decomposition_needed": task.decomposition_needed,
        },
        "pipeline_trace": {
            "registry_models": len(registry),
            "feasible_after_filter": len(feasible),
            "feasibility_exclusions": feasibility_reasons,
            "policy_exclusions": policy_reasons,
            "after_policy": len(gated),
            "after_pareto": len(frontier),
            "scored_candidates": len(scored),
            "profile_used": effective_profile,
            "policy_notes": policy.notes,
            "policy_verifier_types": policy.require_verifier_types,
        },
        "confidence": {
            "top_confidence": selected_win_probability,
            "margin": selected_margin,
            "winning_model": primary_model["name"],
            "win_probabilities": confidence_data["win_probabilities"],
            "min_confidence_threshold": task.request_constraints.min_confidence,
        },
        "selected_plan_id": plan.plan_id if plan else None,
        "selected_plan_type": plan.plan_type if plan else None,
        "escalate_to_human": escalate,
        "abstain": abstain,
        "shadow_routing": shadow_info,
        "observability": {
            "routing_latency_ms": elapsed_ms,
            "models_evaluated": len(scored),
            "confidence_sims": confidence_data["n_simulations"],
        },
        "learning_hook": {
            "outcome_pending": True,
            "primary_model": primary_model["name"],
            "task_family": task.primary_family,
            "domain": task.domain,
            "confidence": selected_win_probability,
            "note": "Record authenticated execution outcomes to update priors via Bayesian learning.",
        },
    }

    hash_input = json.dumps({
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "task_family": task.primary_family,
        "domain": task.domain,
        "profile": effective_profile,
        "feasible_models": sorted([m["name"] for m in feasible]),
        "registry_version": REGISTRY_VERSION,
    }, sort_keys=True)
    decision_record["reproducibility_hash"] = hashlib.sha256(hash_input.encode()).hexdigest()[:24]

    return RoutingDecision(
        selected_plan=plan if not abstain else None,  # never expose a plan when abstaining
        abstain=abstain,
        escalate_to_human=escalate,
        decision_record=decision_record,
    )


def record_outcome(
    registry: List[Dict[str, Any]],
    model_name: str,
    task_family: str,
    success: bool,
    quality_score: float = 0.0,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    user_accepted: bool = True,
    safety_flagged: bool = False,
) -> None:
    """Update model evaluation state and Bayesian priors from an observed outcome.

    NOTE: mutates `registry` in place. Callers using the shared in-process
    MODEL_REGISTRY should hold an appropriate lock in multi-threaded/async
    contexts (see app/api/routes.py for the FastAPI-level lock).
    """
    for model in registry:
        if model["name"] == model_name:
            ev = model["evaluation"]
            ev["samples"] += 1
            ev["quality_sum"] += quality_score
            ev["latency_sum_ms"] += latency_ms
            ev["cost_sum"] += cost_usd
            ev["last_updated"] = datetime.now(timezone.utc).isoformat()
            if success:
                ev["wins"] += 1
            else:
                ev["losses"] += 1
            if user_accepted:
                ev["user_acceptance_sum"] += 1.0
            if safety_flagged:
                ev["safety_flags"] += 1
            # Update family-specific and global priors once per observation.
            # Quality is a bounded soft signal, while success remains the
            # dominant outcome label.
            # Clamp quality_score to [0,1] to keep Beta posterior valid
            quality_score = max(0.0, min(1.0, quality_score))
            observation = 0.7 * float(bool(success)) + 0.3 * quality_score
            fam_prior = model["priors"]["task_family"].get(task_family)
            if fam_prior:
                fam_prior["alpha"] += observation
                fam_prior["beta"] += 1.0 - observation
                # Decay to prevent stale priors dominating
                if fam_prior["alpha"] + fam_prior["beta"] > 200:
                    fam_prior["alpha"] = max(fam_prior["alpha"] * 0.95, 1.0)
                    fam_prior["beta"] = max(fam_prior["beta"] * 0.95, 1.0)
            else:
                # Create a new family prior seeded from this first observation
                model["priors"]["task_family"][task_family] = {
                    "alpha": max(1.0 + observation, 1.0),
                    "beta": max(1.0 + (1.0 - observation), 1.0),
                }
            # Always update global prior so it stays current
            g = model["priors"].get("global")
            if g:
                g["alpha"] += observation
                g["beta"] += 1.0 - observation
                if g["alpha"] + g["beta"] > 500:
                    g["alpha"] = max(g["alpha"] * 0.97, 1.0)
                    g["beta"] = max(g["beta"] * 0.97, 1.0)
            break
    else:
        # Loop completed without break — model not found in registry
        import logging as _logging
        _logging.getLogger("neural_gateway.router").warning(
            f"record_outcome: model '{model_name}' not found in registry — "
            "Bayesian prior NOT updated. Check that model names match the full "
            "OpenRouter ID (e.g. 'openai/gpt-4o', not 'gpt-4o')."
        )
