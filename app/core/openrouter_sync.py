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
    _seeded_rng,
)

logger = logging.getLogger("atlas.openrouter_sync")


def _calculate_relative_cost(input_cost: float, output_cost: float) -> float:
    """Compute a blended relative cost score (0-1). Higher = more expensive."""
    blended = 0.35 * input_cost + 0.65 * output_cost
    return min(blended / 30.0, 1.0)


def fetch_openrouter_models() -> List[Dict[str, Any]]:
    """Fetch live models from OpenRouter and map them to ATLAS schema."""
    url = "https://openrouter.ai/api/v1/models"
    try:
        req = urllib.request.urlopen(url, timeout=10)
        data = json.loads(req.read())
        return data.get("data", [])
    except Exception as e:
        logger.error(f"Failed to fetch OpenRouter models: {e}")
        return []


def load_real_benchmarks() -> Dict[str, Any]:
    """Load ground-truth benchmark data from JSON."""
    path = os.path.join(os.path.dirname(__file__), "..", "models", "real_benchmarks.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load real benchmarks: {e}")
        return {}


def map_model_to_registry(or_model: Dict[str, Any], real_benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    """Map an OpenRouter model dictionary to the ATLAS registry schema."""
    # Handle id format: "openai/gpt-4o"
    full_id = or_model.get("id", "")
    parts = full_id.split("/")
    provider = parts[0] if len(parts) > 1 else "Unknown"
    name = parts[-1] if len(parts) > 1 else full_id

    # The rng here provides a deterministic baseline for missing priors/benchmarks 
    # based on the model name.
    rng = _seeded_rng(name, salt="atlas-registry-v1")
    tier = _classify_tier(name)
    perf = _build_performance(name, provider, tier, rng)
    domains = _build_domains(provider, name, tier, rng)
    
    # Inject ground-truth benchmarks if available
    if full_id in real_benchmarks:
        real_data = real_benchmarks[full_id]
        if "performance" in real_data:
            perf.update(real_data["performance"])
        if "domains" in real_data:
            domains.update(real_data["domains"])
    
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
        "ops_static": _build_ops_static(tier, rng),
        "ops_dynamic": _build_ops_dynamic(tier, rng),
        "priors": _build_priors(name, rng),
        "evaluation": _build_evaluation(),  # Blank slate
        "performance": perf,
        "benchmarks": _build_benchmarks(perf, rng),
        "domains": domains,
        "safety": _build_safety(tier, rng),
        "routing": _routing_meta(tier == "Economy"),
        "verifier_fit": _build_verifier_fit(tier, rng),
    }
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
        
    real_benchmarks = load_real_benchmarks()
    
    updated_models = []
    for or_model in or_models:
        model_id = or_model.get("id", "")
        # Atlas Router should route to base models, not other meta-routers
        if model_id.startswith("openrouter/"):
            continue
            
        pricing = or_model.get("pricing", {})
        try:
            if float(pricing.get("prompt", 0.0)) < 0 or float(pricing.get("completion", 0.0)) < 0:
                continue
        except (ValueError, TypeError):
            pass

        mapped = map_model_to_registry(or_model, real_benchmarks)
        # Preserve Bayesian priors and evaluation state if model already exists
        if mapped["name"] in current_map:
            existing = current_map[mapped["name"]]
            mapped["priors"] = existing.get("priors", mapped["priors"])
            mapped["evaluation"] = existing.get("evaluation", mapped["evaluation"])
        updated_models.append(mapped)
        
    bulk_upsert_models(updated_models)
        
    logger.info(f"Successfully seeded database with {len(or_models)} live models from OpenRouter.")
