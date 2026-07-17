"""
FastAPI routes exposing the Neural Gateway as a service.

Endpoints:
    POST /route          -> run the full routing pipeline, return a RoutingDecision
    POST /outcome         -> feed an observed outcome back into Bayesian priors
    GET  /models           -> list the canonical registry (summary view)
    GET  /models/{name}     -> full registry entry for one model
    GET  /health              -> liveness/readiness probe
    GET  /versions              -> current version stamps for every subsystem
"""

import threading
import anyio
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Header, HTTPException, Depends

from app.api.schemas import (
    ExecutionPlanOut,
    ModelSummaryOut,
    OutcomeIn,
    RequestConstraintsIn,
    RouteRequest,
    RouteResponse,
    StageRouteOut,
    TenantContextIn,
    FeedbackRequest,
    TrainParserRequest,
)
from app.config import (
    CALIBRATION_VERSION,
    DEFAULT_ALLOWED_PROVIDERS,
    PARSER_VERSION,
    POLICY_VERSION,
    REGISTRY_VERSION,
    ROUTER_VERSION,
    SCORING_VERSION,
    TELEMETRY_SNAPSHOT_VERSION,
    ADMIN_API_KEY,
    ALLOW_SERVER_FILE_PATHS,
)
from app.core.formatting import format_decision_summary
from app.core.router import record_outcome, route
from app.core.database import get_all_models, get_model, upsert_model, delete_model, add_feedback
from app.models.schemas import RequestConstraints, TenantContext
from app.core.verifiers import ExecutionVerifier, SchemaVerifier
import httpx

router = APIRouter()

# A threading lock to protect read-modify-write cycles in /outcome and /models.
# For multi-worker deployments, this should be a distributed lock or handled natively in SQL.
_write_lock = threading.Lock()
_parser_lock = threading.Lock()


def _require_admin(x_neural_gateway_admin_key: Optional[str] = Header(default=None)) -> None:
    """Protect mutable control-plane endpoints with a configured secret."""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Model mutation is disabled: NEURAL_GATEWAY_ADMIN_API_KEY is not configured")
    if x_neural_gateway_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid administrative API key")


def _to_request_constraints(rc_in: Optional[RequestConstraintsIn]) -> RequestConstraints:
    if rc_in is None:
        rc = RequestConstraints()
    else:
        rc = RequestConstraints(**rc_in.model_dump())
    # Apply the NEURAL_GATEWAY_ALLOWED_PROVIDERS env-var default when the caller didn't
    # set its own filter. Per-request `allowed_providers` always wins.
    if rc.allowed_providers is None and DEFAULT_ALLOWED_PROVIDERS:
        rc.allowed_providers = list(DEFAULT_ALLOWED_PROVIDERS)
    return rc


def _to_tenant_context(tc_in: Optional[TenantContextIn]) -> TenantContext:
    if tc_in is None:
        return TenantContext()
    return TenantContext(**tc_in.model_dump())


def _strip_noise(record: Dict[str, Any]) -> Dict[str, Any]:
    # Trim per-model dumps from the response — callers only need the
    # primary + fallbacks + verifiers, not the full list of rejected or
    # simulated candidates. Removes:
    #   - confidence.win_probabilities  (Thompson-Sampling per-model dict)
    #   - pipeline_trace.feasibility_exclusions (rejected models + reason)
    #   - pipeline_trace.policy_exclusions      (policy-rejected models)
    # Counts (registry_models, after_pareto, etc.) stay for observability.
    scrubbed = dict(record)
    conf = scrubbed.get("confidence")
    if isinstance(conf, dict) and "win_probabilities" in conf:
        scrubbed["confidence"] = {k: v for k, v in conf.items() if k != "win_probabilities"}
    pt = scrubbed.get("pipeline_trace")
    if isinstance(pt, dict):
        scrubbed["pipeline_trace"] = {
            k: v for k, v in pt.items()
            if k not in ("feasibility_exclusions", "policy_exclusions")
        }
    return scrubbed


def _plan_to_out(plan) -> Optional[ExecutionPlanOut]:
    if plan is None:
        return None
    trace = {k: v for k, v in plan.trace.items() if k != "win_probabilities"}
    return ExecutionPlanOut(
        plan_id=plan.plan_id,
        plan_type=plan.plan_type,
        selected_model=plan.selected_model,
        stage_routes=[StageRouteOut(**asdict(sr)) for sr in plan.stage_routes],
        fallback_models=plan.fallback_models,
        verifier_models=plan.verifier_models,
        expected_latency_ms=plan.expected_latency_ms,
        expected_cost_usd=plan.expected_cost_usd,
        expected_quality=plan.expected_quality,
        confidence=plan.confidence,
        utility=plan.utility,
        confidence_margin=plan.confidence_margin,
        profile_used=plan.profile_used,
        explanation=plan.explanation,
        trace=trace,
    )


@router.post("/route", response_model=RouteResponse)
def route_request(payload: RouteRequest) -> RouteResponse:
    if payload.files and not ALLOW_SERVER_FILE_PATHS:
        raise HTTPException(
            status_code=400,
            detail="Server-local file paths are disabled; upload artifacts to a managed store and pass trusted IDs instead.",
        )
    try:
        decision = route(
            prompt=payload.prompt,
            input_formats=payload.input_formats,
            estimated_tokens=payload.estimated_tokens,
            estimated_output_tokens=payload.estimated_output_tokens,
            artifact_hints=payload.artifact_hints,
            request_constraints=_to_request_constraints(payload.request_constraints),
            tenant_context=_to_tenant_context(payload.tenant_context),
            files=payload.files,
            profile_name=payload.profile_name,
            shadow_model=payload.shadow_model,
            registry=get_all_models(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RouteResponse(
        abstain=decision.abstain,
        escalate_to_human=decision.escalate_to_human,
        selected_plan=_plan_to_out(decision.selected_plan),
        decision_record=_strip_noise(decision.decision_record),
        summary_text=format_decision_summary(decision),
    )


@router.post("/execute", tags=["neural_gateway-gateway"])
async def execute_route_proxy(payload: RouteRequest, x_openrouter_key: str = Header(...)):
    """Data Plane endpoint: Route the prompt AND execute the LLM inference against OpenRouter."""
    decision = await anyio.to_thread.run_sync(lambda: route(
        prompt=payload.prompt,
        input_formats=payload.input_formats,
        estimated_tokens=payload.estimated_tokens,
        estimated_output_tokens=payload.estimated_output_tokens,
        artifact_hints=payload.artifact_hints,
        request_constraints=_to_request_constraints(payload.request_constraints),
        tenant_context=_to_tenant_context(payload.tenant_context),
        files=payload.files,
        profile_name=payload.profile_name,
        shadow_model=payload.shadow_model,
        registry=get_all_models(),
    ))
    
    if decision.abstain or not decision.selected_plan:
        raise HTTPException(status_code=400, detail="Router abstained or no plan generated.")
        
    plan = decision.selected_plan
    
    async def call_openrouter(model_name: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {x_openrouter_key}",
                    "HTTP-Referer": "https://neural_gateway-neural-gateway.local",
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": payload.prompt}]
                }
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"OpenRouter Error: {resp.text}")
            return resp.json()["choices"][0]["message"]["content"]

    # Handle Frugal Cascade Execution
    if plan.plan_type == "cascade" and plan.stage_routes:
        cheap_model = plan.selected_model
        fallback_model = plan.fallback_models[0] if plan.fallback_models else None
        # Use verification_strategy set by router.py — NOT verifier_models (those are model names)
        strategy = getattr(plan, "verification_strategy", None)
        
        response_text = await call_openrouter(cheap_model)
        
        # Verify
        passed = True
        if strategy == "ast_execution":
            passed = ExecutionVerifier().verify(response_text)
        elif strategy == "json_schema":
            passed = SchemaVerifier().verify(response_text)
            
        if passed:
            return {"status": "success", "model_used": cheap_model, "cascaded": False, "response": response_text}
        else:
            if not fallback_model:
                raise HTTPException(status_code=500, detail="Verification failed and no fallback available.")
            # Cascade to expensive model
            escalated_response = await call_openrouter(fallback_model)
            return {"status": "success", "model_used": fallback_model, "cascaded": True, "response": escalated_response}
            
    else:
        # Standard Single-Shot Execution
        model_name = plan.selected_model
        response_text = await call_openrouter(model_name)
        return {"status": "success", "model_used": model_name, "cascaded": False, "response": response_text}


@router.post("/outcome", status_code=202)
def record_outcome_route(payload: OutcomeIn, _: None = Depends(_require_admin)) -> Dict[str, str]:
    with _write_lock:
        models = get_all_models()
        if not any(m["name"] == payload.model_name for m in models):
            raise HTTPException(status_code=404, detail=f"Unknown model: {payload.model_name}")
        record_outcome(
            registry=models,
            model_name=payload.model_name,
            task_family=payload.task_family,
            success=payload.success,
            quality_score=payload.quality_score,
            latency_ms=payload.latency_ms,
            cost_usd=payload.cost_usd,
            user_accepted=payload.user_accepted,
            safety_flagged=payload.safety_flagged,
        )
        updated_model = next((m for m in models if m["name"] == payload.model_name), None)
        if updated_model:
            upsert_model(updated_model)
    return {"status": "accepted"}


@router.get("/models", response_model=List[ModelSummaryOut])
def list_models() -> List[ModelSummaryOut]:
    return [
        ModelSummaryOut(
            name=str(m.get("name", "")),
            provider=str(m.get("provider", "")),
            tier=str(m.get("tier", "")),
            status=m.get("status", "unknown"),
            open_weight=m.get("open_weight", False),
            context_window=(m.get("context") or {}).get("window", 0),
            relative_cost_score=(m.get("pricing") or {}).get("relative_cost_score", 0.0),
            incident_status=(m.get("ops_dynamic") or {}).get("incident_status", "none"),
        )
        for m in get_all_models()
    ]


@router.get("/models/{name}")
def get_model_route(name: str) -> Dict[str, Any]:
    m = get_model(name)
    if not m:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")
    return m


@router.post("/models", status_code=201)
def create_or_update_model(payload: Dict[str, Any] = Body(...), _: None = Depends(_require_admin)) -> Dict[str, str]:
    if "name" not in payload:
        raise HTTPException(status_code=400, detail="Model must have a 'name'")
    required_keys = ["provider", "tier", "ops_dynamic", "pricing", "context", "priors"]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")
    
    with _write_lock:
        upsert_model(payload)
    return {"status": "success", "message": f"Model {payload['name']} upserted."}


@router.delete("/models/{name}", status_code=204)
async def delete_model_route(name: str, _: None = Depends(_require_admin)) -> None:
    if not delete_model(name):
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")


@router.get("/health", tags=["meta"])
async def health_check():
    return {"status": "ok", "registry_models": len(get_all_models())}

@router.post("/train_parser", tags=["neural_gateway-gateway"])
async def train_parser(payload: TrainParserRequest):
    try:
        from app.core.embedding_parser import get_parser, _cached_parse
        import subprocess
        import sys
        from pathlib import Path
        
        parser = get_parser()
        
        new_example = {
            "text": payload.prompt,
            "primary_family": payload.primary_family,
            "domain": payload.domain,
            "risk_tier": payload.risk_tier,
            "complexity": payload.complexity,
            "risk_type": "standard",
            "expected_output": "free_text",
            "document_type": "generic",
            "decomposition_needed": False,
            "needs_verification": False
        }
        
        project_root = Path(__file__).resolve().parents[2]
        dataset_path = project_root / "data" / "semantic_examples.json"
        
        with _parser_lock:
            parser.add_example(new_example)
            
        return {"status": "success", "message": "Example added and matrix rebuilt"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/versions")
async def versions() -> Dict[str, str]:
    return {
        "router_version": ROUTER_VERSION,
        "parser_version": PARSER_VERSION,
        "policy_version": POLICY_VERSION,
        "scoring_version": SCORING_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "registry_version": REGISTRY_VERSION,
        "telemetry_snapshot_version": TELEMETRY_SNAPSHOT_VERSION,
    }


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest) -> Dict[str, str]:
    """Submit parser correction feedback to update the dynamic memory bank."""
    add_feedback(req.prompt, req.correct_family)
    return {"status": "success", "message": "Feedback integrated into memory bank"}
