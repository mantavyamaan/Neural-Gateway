"""
Optimization and Decision Making layer.

Contains:
  - Signal maps translating task family / domain into registry lookups.
  - Contextual Bayesian quality estimation (priors + task fit + domain
    expertise + runtime health).
  - Pareto frontier reduction to eliminate dominated candidates.
  - Workflow-specific utility scoring (configurable optimization profiles).
  - Confidence estimation via Thompson Sampling.

This module is pure math over plain dicts/dataclasses — no I/O, no
registry mutation (call sites always work on deepcopies), which keeps it
easy to unit test and safe to call concurrently.
"""

import hashlib
import math

from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from app.models.schemas import TaskFeatures

# --------------------------------------------------------------------------
# Signal maps
# --------------------------------------------------------------------------

TASK_FAMILY_SIGNAL_MAP: Dict[str, Dict[str, float]] = {
    "coding": {
        "perf:coding": 0.50, "perf:agentic_tasks": 0.15,
        "bench:swe_bench": 0.15, "bench:humaneval": 0.10,
        "perf:instruction_following": 0.10
    },
    "reasoning": {
        "perf:reasoning": 0.42, "perf:scientific_reasoning": 0.18,
        "bench:gpqa": 0.20, "bench:mmlu": 0.10,
        "perf:instruction_following": 0.10
    },
    "mathematics": {
        "perf:mathematics": 0.50, "bench:aime": 0.25,
        "perf:reasoning": 0.15, "bench:gpqa": 0.10
    },
    "chat": {
        "perf:creative_writing": 0.20, "perf:instruction_following": 0.40,
        "domain:general": 0.20, "domain:customer_support": 0.20
    },
    "vision": {
        "perf:vision_understanding": 0.50, "bench:mmmu": 0.25,
        "perf:ocr": 0.10, "perf:table_understanding": 0.15
    },
    "ocr": {
        "perf:ocr": 0.55, "bench:docvqa": 0.20,
        "perf:table_understanding": 0.15, "perf:document_qa": 0.10
    },
    "document_qa": {
        "perf:document_qa": 0.40, "perf:long_context": 0.20,
        "bench:docvqa": 0.15, "perf:summarization": 0.10,
        "perf:table_understanding": 0.15
    },
    "summarization": {
        "perf:summarization": 0.50, "perf:long_context": 0.20,
        "perf:instruction_following": 0.20, "perf:document_qa": 0.10
    },
    "translation": {
        "perf:translation": 0.70, "perf:instruction_following": 0.30
    },
    "agent": {
        "perf:agentic_tasks": 0.45, "perf:instruction_following": 0.20,
        "perf:reasoning": 0.20, "perf:coding": 0.15
    },
    "audio": {
        "perf:audio_understanding": 0.60, "perf:summarization": 0.20,
        "perf:instruction_following": 0.20
    },
    "image_generation": {
        "perf:image_generation": 0.90, "perf:instruction_following": 0.10
    },
    "video_generation": {
        "perf:video_generation": 0.90, "perf:instruction_following": 0.10
    }
}

DOMAIN_SAFETY_MAP: Dict[str, str] = {
    "medical": "medical", "legal": "legal", "finance": "finance",
    "security": "cybersecurity", "general": "general",
    "software": "general", "science": "general", "research": "general",
    "education": "general", "mathematics": "general",
    "customer_support": "general",
    "crm": "general", "hrm": "general", "project": "general", "accounts": "general"
}

DOMAIN_EXPERTISE_MAP: Dict[str, str] = {
    "medical": "medical", "legal": "legal", "finance": "finance",
    "security": "software", "software": "software", "science": "science",
    "research": "research", "education": "education", "general": "general",
    "mathematics": "mathematics", "customer_support": "customer_support",
    "crm": "customer_support", "hrm": "customer_support",
    "project": "software", "accounts": "finance"
}

WEIGHT_PROFILES = {
    "quality_first": {"quality": 0.36, "uncertainty": 0.16, "cost": 0.10, "latency": 0.08, "reliability": 0.14, "riskfit": 0.10, "runtime": 0.06},
    "budget_first": {"quality": 0.30, "uncertainty": 0.05, "cost": 0.45, "latency": 0.10, "reliability": 0.05, "riskfit": 0.03, "runtime": 0.02},
    "latency_first": {"quality": 0.20, "uncertainty": 0.10, "cost": 0.10, "latency": 0.30, "reliability": 0.10, "riskfit": 0.08, "runtime": 0.12},
    "balanced": {"quality": 0.30, "uncertainty": 0.10, "cost": 0.30, "latency": 0.10, "reliability": 0.10, "riskfit": 0.05, "runtime": 0.05},
    "high_risk": {"quality": 0.28, "uncertainty": 0.18, "cost": 0.04, "latency": 0.04, "reliability": 0.18, "riskfit": 0.18, "runtime": 0.10},
    "customer_support_summarization": {"quality": 0.22, "uncertainty": 0.10, "cost": 0.22, "latency": 0.18, "reliability": 0.10, "riskfit": 0.08, "runtime": 0.10},
    "contract_review_intake": {"quality": 0.30, "uncertainty": 0.15, "cost": 0.06, "latency": 0.06, "reliability": 0.15, "riskfit": 0.18, "runtime": 0.10},
    "coding_assistant": {"quality": 0.32, "uncertainty": 0.14, "cost": 0.10, "latency": 0.10, "reliability": 0.12, "riskfit": 0.08, "runtime": 0.14},
    "research_drafting": {"quality": 0.34, "uncertainty": 0.14, "cost": 0.08, "latency": 0.06, "reliability": 0.12, "riskfit": 0.10, "runtime": 0.16},
    "invoice_ocr_pipeline": {"quality": 0.28, "uncertainty": 0.14, "cost": 0.12, "latency": 0.08, "reliability": 0.12, "riskfit": 0.10, "runtime": 0.16},
    "multilingual_chat": {"quality": 0.24, "uncertainty": 0.10, "cost": 0.14, "latency": 0.18, "reliability": 0.10, "riskfit": 0.08, "runtime": 0.16},
    "real_time_voice_agent": {"quality": 0.20, "uncertainty": 0.10, "cost": 0.10, "latency": 0.24, "reliability": 0.12, "riskfit": 0.08, "runtime": 0.16},
    "audio_summary": {"quality": 0.26, "uncertainty": 0.12, "cost": 0.14, "latency": 0.14, "reliability": 0.12, "riskfit": 0.08, "runtime": 0.14},
}


# --------------------------------------------------------------------------
# Signal lookup / fit functions
# --------------------------------------------------------------------------

def signal_value(model: Dict[str, Any], signal: str) -> float:
    kind, key = signal.split(":", 1)
    if kind == "perf":
        return model["performance"].get(key, 0.5)
    if kind == "bench":
        return model["benchmarks"].get(key, 0.5)
    if kind == "domain":
        return model["domains"].get(key, 0.5)
    return 0.5


def family_fit(model: Dict[str, Any], task: TaskFeatures) -> float:
    sigmap = TASK_FAMILY_SIGNAL_MAP.get(task.primary_family)
    if not sigmap:
        return model["domains"].get("general", 0.5)
    base = sum(signal_value(model, sig) * w for sig, w in sigmap.items())
    domkey = DOMAIN_EXPERTISE_MAP.get(task.domain, "general")
    domfit = model["domains"].get(domkey, 0.5)
    fit = 0.75 * base + 0.25 * domfit
    if task.expected_output == "structured_json":
        fit = 0.70 * fit + 0.30 * model["performance"].get("json_reliability", 0.5)
    if task.min_context_window > 100_000:
        fit = 0.70 * fit + 0.30 * model["performance"].get("long_context", 0.5)
    if task.requires_ocr:
        fit = 0.70 * fit + 0.30 * model["performance"].get("ocr", 0.5)
    if "spreadsheet" in task.input_formats:
        fit = 0.80 * fit + 0.20 * model["performance"].get("spreadsheet_reasoning", 0.5)
    if "audio" in task.input_formats:
        fit = 0.70 * fit + 0.30 * model["performance"].get("audio_understanding", 0.5)
    return float(min(max(fit, 0.0), 1.0))


def stage_fit(model: Dict[str, Any], stage_name: str, task: TaskFeatures) -> float:
    stage_map = {
        "domain_reasoning": lambda: model["domains"].get(DOMAIN_EXPERTISE_MAP.get(task.domain, "general"), 0.5),
        "structured_output": lambda: model["performance"].get("json_reliability", 0.5),
        "audio_understanding": lambda: model["performance"].get("audio_understanding", 0.5),
        "vision_understanding": lambda: model["performance"].get("vision_understanding", 0.5),
        "ocr": lambda: model["performance"].get("ocr", 0.5),
        "coding": lambda: model["performance"].get("coding", 0.5),
        "document_qa": lambda: model["performance"].get("document_qa", 0.5),
        "summarization": lambda: model["performance"].get("summarization", 0.5),
    }
    if stage_name in stage_map:
        return stage_map[stage_name]()
    return family_fit(model, task)


def composite_workflow_fit(model: Dict[str, Any], task: TaskFeatures) -> float:
    scores = [stage_fit(model, stage, task) for stage in task.required_stages]
    if not scores:
        return family_fit(model, task)
    return float(np.mean(scores))


def risk_support(model: Dict[str, Any], task: TaskFeatures) -> float:
    safety_key = DOMAIN_SAFETY_MAP.get(task.domain, "general")
    base = model["safety"].get(safety_key, model["safety"].get("general", 0.7))
    if task.risk_type == "regulated_advice":
        base = 0.8 * base + 0.2 * model["safety"].get("regulated_advice", base)
    if task.risk_tier == "low":
        return min(1.0, base + 0.05)
    return base


# --------------------------------------------------------------------------
# Static / runtime operational scoring
# --------------------------------------------------------------------------

def static_cost_score(model: Dict[str, Any]) -> float:
    return model["pricing"]["relative_cost_score"]


def static_latency_score(model: Dict[str, Any]) -> float:
    return model["ops_static"]["latency_score"]


def reliability_score(model: Dict[str, Any]) -> float:
    return model["ops_static"]["reliability"]


def freshness_score(model: Dict[str, Any]) -> float:
    sec = model["ops_dynamic"].get("telemetry_freshness_sec", 60)
    return max(0.0, 1.0 - min(sec / 600.0, 1.0))


def incident_penalty(status: str) -> float:
    return {"green": 0.0, "yellow": 0.08, "orange": 0.18, "red": 0.35}.get(status, 0.10)


def runtime_health_score(model: Dict[str, Any]) -> float:
    dyn = model["ops_dynamic"]
    components = [
        (0.20, max(0.0, 1.0 - min(dyn["recent_latency_ms"] / 5000.0, 1.0))),
        (0.18, max(0.0, 1.0 - min(dyn["recent_failure_rate"] / 0.15, 1.0))),
        (0.16, dyn["current_availability"]),
        (0.10, max(0.0, 1.0 - dyn["rate_limit_pressure"])),
        (0.08, max(0.0, 1.0 - dyn["queue_pressure"])),
        (0.10, max(0.0, 1.0 - incident_penalty(dyn["incident_status"]))),
        (0.08, max(0.0, 1.0 - 0.5 * dyn["budget_pressure"])),
        (0.10, freshness_score(model)),
    ]
    return float(min(max(sum(w * v for w, v in components), 0.0), 1.0))


def effective_prior(model: Dict[str, Any], family: str) -> Tuple[float, float]:
    fam = model["priors"]["task_family"].get(family, model["priors"]["global"])
    alpha, beta = float(fam["alpha"]), float(fam["beta"])
    return alpha, beta


def estimate_request_cost_usd(model: Dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    return ((input_tokens / 1_000_000) * model["pricing"]["input_cost"] +
            (output_tokens / 1_000_000) * model["pricing"]["output_cost"])


def predict_output_tokens(task: TaskFeatures) -> int:
    """Dynamic output token prediction based on task characteristics."""
    if task.estimated_output_tokens != 1200:
        return task.estimated_output_tokens
    FAMILY_BASE = {
        "coding": 1800, "agent": 2000, "reasoning": 1500,
        "mathematics": 800, "document_qa": 600, "summarization": 500,
        "translation": 400, "extraction": 300, "chat": 350,
        "creative": 1200, "vision": 400, "ocr": 600,
        "audio": 400, "image_generation": 100, "video_generation": 100,
    }
    base = FAMILY_BASE.get(task.primary_family, 600)
    COMPLEXITY_MULT = {"low": 0.6, "medium": 1.0, "high": 1.8}
    mult = COMPLEXITY_MULT.get(getattr(task, 'complexity', 'medium') or 'medium', 1.0)
    if getattr(task, 'requires_json', False) or getattr(task, 'expected_output', '') == 'structured_json':
        mult *= 1.3
    input_scale = min(1.5, max(0.5, task.estimated_tokens / 500.0))
    result = int(base * mult * min(input_scale, 2.0))
    return max(50, min(result, 8000))


def estimate_request_latency_ms(model: Dict[str, Any], task: TaskFeatures, n_stages: int = 1) -> float:
    """
    Estimate total end-to-end latency for a request.

    Formula:  Total = TTFT + Generation_Time
      - TTFT (Time To First Token): from ops_dynamic.recent_latency_ms, scaled by
        modality and n_stages.
      - Generation Time: estimated_output_tokens / tokens_per_sec * 1000 ms.
        This is the dominant factor for long-form responses (code, essays, websites).

    Without the generation component a 3000-token website would appear to complete
    in ~700 ms when reality is closer to 50-60 seconds for a Frontier model.
    """
    ttft = model["ops_dynamic"]["recent_latency_ms"]

    # Modality overhead on TTFT (pre-fill cost for multimodal inputs)
    modality_mult = 1.0
    if "audio" in task.input_formats or "video" in task.input_formats:
        modality_mult += 0.25
    if "pdf" in task.input_formats and task.requires_ocr:
        modality_mult += 0.20

    scaled_ttft = ttft * modality_mult * n_stages

    # Generation time = output tokens / throughput
    output_tokens = predict_output_tokens(task)
    tps = model["ops_dynamic"].get("tokens_per_sec", 80.0)  # fallback for legacy rows
    generation_ms = (output_tokens / tps) * 1000.0

    return float(scaled_ttft + generation_ms)


# --------------------------------------------------------------------------
# Bayesian contextual quality
# --------------------------------------------------------------------------

def beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta + 1e-9)


def beta_variance(alpha: float, beta: float) -> float:
    return (alpha * beta) / (((alpha + beta + 1e-9) ** 2) * (alpha + beta + 1.0 + 1e-9))


def convex_combine(values: List[float], weights: List[float]) -> float:
    total = sum(weights)
    if total == 0:
        return float(np.mean(values))
    return sum(v * w for v, w in zip(values, weights)) / total


def attach_contextual_quality(
    models: List[Dict[str, Any]],
    task: TaskFeatures,
) -> List[Dict[str, Any]]:
    enriched = []
    for model in models:
        g = model["priors"]["global"]
        global_mean = beta_mean(g["alpha"], g["beta"])
        falpha, fbeta = effective_prior(model, task.primary_family)
        fam_mean = beta_mean(falpha, fbeta)
        fam_var = beta_variance(falpha, fbeta)
        fam_fit = family_fit(model, task)
        wf_fit = composite_workflow_fit(model, task)
        dom_fit = model["domains"].get(DOMAIN_EXPERTISE_MAP.get(task.domain, "general"), 0.5)
        rt_fit = runtime_health_score(model)
        fr_fit = freshness_score(model)
        contextual_mean = convex_combine(
            [global_mean, fam_mean, fam_fit, wf_fit, dom_fit, fr_fit],
            [0.12, 0.22, 0.18, 0.23, 0.15, 0.10]
        )
        runtime_adjusted_mean = 0.76 * contextual_mean + 0.24 * rt_fit
        model["q"] = {
            "global_mean": global_mean,
            "family_mean": fam_mean,
            "family_variance": fam_var,
            "family_fit": fam_fit,
            "workflow_fit": wf_fit,
            "domain_fit": dom_fit,
            "runtime_fit": rt_fit,
            "freshness_fit": fr_fit,
            "contextual_mean": contextual_mean,
            "runtime_adjusted_mean": runtime_adjusted_mean,
            "uncertainty": math.sqrt(max(fam_var, 1e-9)),
            "alpha": falpha,
            "beta": fbeta,
        }
        enriched.append(model)
    return enriched


# --------------------------------------------------------------------------
# Pareto frontier reduction
# --------------------------------------------------------------------------

def pareto_vector(model: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        model["q"]["runtime_adjusted_mean"],
        -static_cost_score(model),
        static_latency_score(model),
        reliability_score(model),
        model["q"]["runtime_fit"],   # use cached value — avoids recomputing 8-component health score
    )


def dominates_with_margin(v1: Tuple[float, float, float, float, float], v2: Tuple[float, float, float, float, float], eps: float = 0.02) -> bool:
    ge_all = all(a >= b for a, b in zip(v1, v2))
    gt_any = any(a > b + eps for a, b in zip(v1, v2))
    return ge_all and gt_any


def pareto_frontier(
    models: List[Dict[str, Any]],
    vectors: Optional[List[tuple]] = None,
    eps: float = 0.02,
) -> List[Dict[str, Any]]:
    if not models:
        return []
    if vectors is None:
        vectors = [pareto_vector(m) for m in models]
    N = len(vectors)
    if N == 1:
        return list(models)
    V = np.array(vectors, dtype=np.float64)
    D = V.shape[1]
    # Per-dimension epsilon: tight for cost (index 1), standard for others
    # Dimensions from pareto_vector: (quality, neg_cost, latency, reliability, runtime)
    base_eps = np.full(D, eps, dtype=np.float64)
    if D > 1:
        base_eps[1] = min(eps, 0.003)  # cost dimension: very tight epsilon so cheap models aren't falsely dominated
    # diffs[i,j,d] = V[i,d] - V[j,d]  (how much i exceeds j on dim d)
    diffs = V[:, np.newaxis, :] - V[np.newaxis, :, :]
    ge_all = np.all(diffs >= -base_eps, axis=-1)   # i >= j (within eps) on all dims
    gt_any = np.any(diffs > base_eps, axis=-1)      # i > j (beyond eps) on at least one
    not_self = ~np.eye(N, dtype=bool)
    dominated = np.any(ge_all & gt_any & not_self, axis=0)  # any i dominates j?
    return [m for m, d in zip(models, dominated) if not d]


# --------------------------------------------------------------------------
# Utility scoring
# --------------------------------------------------------------------------

def minmax_normalize(values: List[float], lo: float = 0.0, hi: float = 1.0) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-9:
        mid = (lo + hi) / 2.0
        return [mid] * len(values)
    span = vmax - vmin
    return [lo + (hi - lo) * (v - vmin) / span for v in values]


def choose_effective_profile(task: TaskFeatures, requested: str) -> str:
    if task.risk_tier in {"high", "extreme"}:  # treat extreme as high-risk routing profile
        return "high_risk"
    if task.workflow_profile in WEIGHT_PROFILES:
        return task.workflow_profile
    # Auto-optimize for cost on trivial chat tasks
    if (task.primary_family == "chat"
            and task.risk_tier == "low"
            and task.complexity == "low"
            and not task.requires_verifier
            and not task.requires_citations
            and not task.requires_ocr
            and task.expected_output != "structured_json"):
        return "budget_first"
    return requested if requested in WEIGHT_PROFILES else "balanced"


def compute_utilities(
    models: List[Dict[str, Any]],
    task: TaskFeatures,
    profile_name: str = "balanced",
) -> List[Dict[str, Any]]:
    profile_name = choose_effective_profile(task, profile_name)
    w = WEIGHT_PROFILES[profile_name]

    qvals = [m["q"]["runtime_adjusted_mean"] for m in models]
    uvals = [m["q"]["uncertainty"] for m in models]
    cvals = [static_cost_score(m) for m in models]
    lvals = [static_latency_score(m) for m in models]
    rvals = [reliability_score(m) for m in models]
    rfvals = [risk_support(m, task) for m in models]
    rtvals = [runtime_health_score(m) for m in models]

    qn = minmax_normalize(qvals)
    un = minmax_normalize(uvals)
    cn = minmax_normalize(cvals)
    ln = minmax_normalize(lvals)
    rn = minmax_normalize(rvals)
    rfn = minmax_normalize(rfvals)
    rtn = minmax_normalize(rtvals)

    out = []
    for model, q, u, c, l, r, rf, rt in zip(models, qn, un, cn, ln, rn, rfn, rtn):
        m = model.copy()
        utility = (
            w["quality"] * q
            - w["uncertainty"] * u
            - w["cost"] * c
            + w["latency"] * l        # latency_score: higher = faster, so ADD (good signal)
            + w["reliability"] * r
            + w["riskfit"] * rf
            + w["runtime"] * rt
        )
        if task.ambiguity_score > 0.5:
            uncertainty_val = model["q"]["family_variance"]
            utility -= 0.03 * (1.0 + uncertainty_val)
        if task.decomposition_needed and len(task.required_stages) > 2:
            utility -= 0.01 * len(task.required_stages)

        rc = task.request_constraints
        est_latency = estimate_request_latency_ms(model, task)
        
        # Use heuristic regressor for output length
        pred_output = predict_output_tokens(task)
        est_cost = estimate_request_cost_usd(model, task.estimated_tokens, pred_output)
        
        sla_violation = False
        if rc.max_latency_ms is not None and est_latency > rc.max_latency_ms:
            utility -= 1000.0
            sla_violation = True
        if rc.max_cost_usd is not None and est_cost > rc.max_cost_usd:
            utility -= 1000.0
            sla_violation = True

        m["u"] = {
            "profile_used": profile_name,
            "expected_utility": utility,
            "n_quality": q, "n_uncertainty": u, "n_cost": c,
            "n_latency": l, "n_reliability": r, "n_riskfit": rf, "n_runtime": rt,
            "est_latency_ms": est_latency,
            "est_cost_usd": est_cost,
            "sla_violation": sla_violation,
        }
        out.append(m)
    return sorted(out, key=lambda x: x["u"]["expected_utility"], reverse=True)


# --------------------------------------------------------------------------
# Confidence estimation via Thompson Sampling
# --------------------------------------------------------------------------

def estimate_confidence(
    models: List[Dict[str, Any]],
    task: TaskFeatures,
    profile_name: str = "balanced",
    n_sim: int = 1500,
) -> Dict[str, Any]:
    if not models:
        return {"win_probabilities": {}, "top_model": None, "top_confidence": 0.0, "second_confidence": 0.0}

    profile_name = choose_effective_profile(task, profile_name)
    w = WEIGHT_PROFILES[profile_name]
    # Include primary_family AND domain AND a hash of the prompt so that
    # different prompts with the same candidate model set produce different
    # winners (avoids one model always winning on every request).
    prompt_hash = hashlib.sha256(task.raw_prompt.encode()).hexdigest()[:12]
    seed_str = (
        ",".join(sorted(m.get("name", "") for m in models))
        + task.primary_family
        + task.domain
        + prompt_hash
    )
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    
    n_models = len(models)
    
    means = np.array([m["q"]["runtime_adjusted_mean"] for m in models])
    variances = np.array([max(m["q"]["family_variance"], 1e-4) for m in models])
    
    # Vectorized mean-preserving method-of-moments for Beta parameters.
    # Clamp variance to 99.9% of max theoretical variance to guarantee commons > 0
    # and preserve the original mean exactly (no non-mean-preserving scaling).
    means = np.clip(means, 1e-6, 1.0 - 1e-6)
    max_vars = means * (1.0 - means) * 0.999   # strictly inside the valid range
    variances = np.clip(variances, 1e-6, max_vars)
    
    commons = (means * (1.0 - means) / variances) - 1.0
    commons = np.maximum(commons, 0.01)          # guarantee commons > 0 without scaling
    alphas = np.maximum(means * commons, 1e-3)   # floor prevents Beta(0,x) errors
    betas  = np.maximum((1.0 - means) * commons, 1e-3)
    
    # Sample matrix of shape (n_sim, n_models)
    sq_matrix = rng.beta(alphas, betas, size=(n_sim, n_models))
    
    base_utilities = np.array([m["u"]["expected_utility"] for m in models])
    
    # Retrieve normalised cost, latency, reliability, runtime per model so the
    # full utility function is perturbed — not just the quality dimension.
    n_costs = np.array([m["u"]["n_cost"] for m in models])
    n_lats  = np.array([m["u"]["n_latency"] for m in models])
    n_rels  = np.array([m["u"]["n_reliability"] for m in models])
    n_rts   = np.array([m["u"]["n_runtime"] for m in models])

    # Sample noise for cost, reliability and runtime (small std dev = mild perturbation)
    # Reduced from 0.05 to 0.01 so that a clear winner can emerge above the 30% abstain threshold.
    cost_noise = rng.normal(0.0, 0.01, size=(n_sim, n_models))
    rel_noise  = rng.normal(0.0, 0.01, size=(n_sim, n_models))
    rt_noise   = rng.normal(0.0, 0.01, size=(n_sim, n_models))

    # Full perturbed utility across all 7 dimensions
    utility_matrix = (
        base_utilities
        + w["quality"]     * (sq_matrix - means)            # quality perturbation from beta sample
        - w["cost"]        * np.clip(cost_noise, -0.2, 0.2) # cost uncertainty
        + w["reliability"] * np.clip(rel_noise,  -0.2, 0.2) # reliability uncertainty
        + w["runtime"]     * np.clip(rt_noise,   -0.2, 0.2) # runtime uncertainty
    )

    # Find the indices of the maximum utilities for each simulation
    winners = np.argmax(utility_matrix, axis=1)
    
    # Count wins
    win_counts = np.bincount(winners, minlength=n_models)
    win_probs_arr = win_counts / n_sim
    
    win_probs = {m["name"]: float(win_probs_arr[i]) for i, m in enumerate(models)}
    
    ordered = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
    top_name, top_prob = ordered[0] if ordered else (None, 0.0)
    second_prob = ordered[1][1] if len(ordered) > 1 else 0.0
    
    return {
        "win_probabilities": win_probs,
        "top_model": top_name,
        "top_confidence": top_prob,
        "margin": top_prob - second_prob,
        "ordered": ordered,
        "n_simulations": n_sim,
        "calibration_note": "Calibrate against held-out labeled routing outcomes for production use.",
    }
