import urllib.request
import json
import logging
import os
from typing import Any, Dict, List
from datetime import datetime, timezone

from app.core.database import bulk_upsert_models, get_all_models
from app.models.registry_builder import (
    _classify_tier,
    _is_open_weight,
    _build_modalities,
    _build_capabilities,
    _build_ops_static,
    _build_ops_dynamic,
    _build_priors,
    _build_evaluation,
    _build_performance,
    _build_benchmarks,
    _build_domains,
    _build_safety,
    _routing_meta,
    _build_verifier_fit,
)

logger = logging.getLogger("neural_gateway.openrouter_sync")


def _calculate_relative_cost(input_cost: float, output_cost: float) -> float:
    """Compute a blended relative cost score (0-1). Higher = more expensive."""
    blended = 0.35 * input_cost + 0.65 * output_cost
    return min(blended / 30.0, 1.0)


def fetch_openrouter_models() -> List[Dict[str, Any]]:
    """Fetch live models from OpenRouter and map them to Neural Gateway schema."""
    url = "https://openrouter.ai/api/v1/models"
    try:
        req = urllib.request.urlopen(url, timeout=10)
        data = json.loads(req.read())
        return data.get("data", [])
    except Exception as e:
        logger.error(f"Failed to fetch OpenRouter models: {e}")
        return []


def map_model_to_registry(or_model: Dict[str, Any]) -> Dict[str, Any]:
    """Map an OpenRouter model dictionary to the Neural Gateway registry schema."""
    # Handle id format: "openai/gpt-4o"
    full_id = or_model.get("id", "")
    parts = full_id.split("/")
    provider = parts[0] if len(parts) > 1 else "Unknown"
    name = parts[-1] if len(parts) > 1 else full_id

    tier = _classify_tier(name)
    
    benchmarks = or_model.get("benchmarks") or {}
    aa_data = benchmarks.get("artificial_analysis") or {}
    
    perf = _build_performance(name, provider, tier, aa_data)
    domains = _build_domains(provider, name, tier, perf)

    
    # Calculate pricing (OpenRouter provides per-token cost, we need per-million)
    pricing_data = or_model.get("pricing", {})
    try:
        input_cost = float(pricing_data.get("prompt", 0.0)) * 1_000_000
    except (ValueError, TypeError):
        input_cost = 0.0
    try:
        output_cost = float(pricing_data.get("completion", 0.0)) * 1_000_000
    except (ValueError, TypeError):
        output_cost = 0.0

    relative_cost_score = _calculate_relative_cost(input_cost, output_cost)

    entry: Dict[str, Any] = {
        "name": full_id,  # Keep the full openrouter ID as the unique name so we can call it easily via LiteLLM/OpenAI
        "provider": provider.capitalize(),
        "tier": tier,
        "status": "active",
        "api_available": True,
        "open_weight": _is_open_weight(provider, name),
        "allowed_regions": ["global"],
        "modalities": _build_modalities(name, tier),
        "capabilities": _build_capabilities(name, tier),
        "context": {"window": or_model.get("context_length", 8000)},
        "pricing": {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "relative_cost_score": round(relative_cost_score, 3),
        },
        "ops_static": _build_ops_static(tier),
        "ops_dynamic": _build_ops_dynamic(tier),
        "priors": _build_priors(name),
        "evaluation": _build_evaluation(),  # Blank slate
        "performance": perf,
        "benchmarks": _build_benchmarks(perf),
        "domains": domains,
        "safety": _build_safety(tier),
        "routing": _routing_meta(tier == "Economy"),
        "verifier_fit": _build_verifier_fit(tier),
        "evidence": {
            "source": "unverified_provider_metadata",
            "eligible_for_auto_route": False,
            "evaluated_task_families": [],
        },
    }

    # Patch in real Artificial Analysis throughput/latency if available.
    # AA provides: output_tokens_per_second (throughput) and latency_ms (TTFT p50).
    # These are the gold-standard numbers — always prefer them over tier defaults.
    if aa_data:
        if aa_data.get("output_tokens_per_second") is not None:
            entry["ops_dynamic"]["tokens_per_sec"] = float(aa_data["output_tokens_per_second"])
        if aa_data.get("latency") is not None:
            # AA latency is in seconds, convert to ms
            entry["ops_dynamic"]["recent_latency_ms"] = float(aa_data["latency"]) * 1000.0
        elif aa_data.get("time_to_first_token") is not None:
            entry["ops_dynamic"]["recent_latency_ms"] = float(aa_data["time_to_first_token"]) * 1000.0

    return entry


def sync_openrouter_models() -> None:
    """Pull models from OpenRouter and sync them to the SQLite database."""
    logger.info("Fetching live models from OpenRouter API...")
    or_models = fetch_openrouter_models()
    
    if not or_models:
        logger.warning("No models retrieved from OpenRouter. Sync aborted.")
        return

    logger.info(f"Retrieved {len(or_models)} models. Syncing to database...")
    
    current_models = get_all_models()
    current_map = {m["name"]: m for m in current_models}
    
    if current_map:
        logger.info(f"Database already contains {len(current_map)} models. Refreshing from OpenRouter.")
        
    updated_models = []
    for or_model in or_models:
        model_id = or_model.get("id", "")
        # Neural Gateway should route to base models, not other meta-routers
        if model_id.startswith("openrouter/"):
            continue
            
        pricing = or_model.get("pricing", {})
        if not pricing or float(pricing.get("prompt", 0)) == 0 and float(pricing.get("completion", 0)) == 0 and "free" not in model_id.lower():
            # Skip models with no pricing info unless explicitly free
            continue

        mapped = map_model_to_registry(or_model)
        # Preserve Bayesian priors and evaluation state if model already exists
        if mapped["name"] in current_map:
            existing = current_map[mapped["name"]]
            mapped["priors"] = existing.get("priors", mapped["priors"])
            mapped["evaluation"] = existing.get("evaluation", mapped["evaluation"])
        updated_models.append(mapped)
        
    bulk_upsert_models(updated_models)

    logger.info(f"Successfully synced {len(updated_models)} models to database ({len(or_models)} raw fetched from OpenRouter).")
