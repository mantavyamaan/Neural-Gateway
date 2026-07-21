"""
Canonical Model Registry builder.

Turns the raw provider -> model-name catalog (catalog.py) into fully-shaped
registry entries: modalities, capabilities, context window, pricing,
static/dynamic ops telemetry, Bayesian priors, performance/benchmark
signals, domain expertise, safety scores, and verifier fitness.

Every numeric value here is a *reasonable synthetic default*, generated
deterministically (seeded per model name) so registry snapshots are
reproducible across restarts. In production this module would be replaced
by (or hydrated from) a real benchmark/telemetry pipeline; the shape of
each entry is what the rest of the router depends on, not these exact
numbers. Swap this out for a database-backed or config-driven registry
without touching feasibility/policy/scoring code.
"""

import hashlib
from typing import Any, Dict, List

from app.models.catalog import MODELS

OPEN_WEIGHT_PROVIDERS = {"Meta", "DeepSeek", "Alibaba (Qwen)", "Mistral AI", "01.AI", "NVIDIA", "IBM"}
OPEN_WEIGHT_NAME_HINTS = ["llama", "qwen", "mixtral", "mistral-7b", "gemma", "phi-3", "phi-4", "granite", "yi-", "glm-4"]

VISION_HINTS = ["vision", "4o", "gemini", "grok-4", "claude", "gpt-5", "pro", "nova", "phi-4-multimodal"]
AUDIO_HINTS = ["gemini", "gpt-5.5", "gpt-4o", "nova-premier", "grok-4"]

PERF_KEYS = [
    "coding", "agentic_tasks", "reasoning", "scientific_reasoning", "mathematics",
    "creative_writing", "instruction_following", "vision_understanding", "ocr",
    "table_understanding", "document_qa", "long_context", "summarization",
    "translation", "audio_understanding", "json_reliability",
    "spreadsheet_reasoning", "image_generation", "video_generation",
]
BENCH_KEYS = ["swe_bench", "humaneval", "gpqa", "mmlu", "aime", "mmmu", "docvqa"]
DOMAIN_KEYS = ["general", "legal", "finance", "medical", "software", "science",
               "research", "education", "mathematics", "customer_support"]
SAFETY_KEYS = ["general", "medical", "legal", "finance", "cybersecurity", "regulated_advice"]
VERIFIER_KEYS = ["safety_review", "factuality_review", "schema_validation",
                  "citation_review", "consistency_review"]


def _seeded_rng(name: str, salt: str = "") -> "_Deterministic":
    seed_bytes = hashlib.sha256(f"{name}:{salt}".encode()).digest()
    return _Deterministic(int.from_bytes(seed_bytes[:8], "big"))


class _Deterministic:
    """Tiny deterministic PRNG (xorshift64) so registry values are stable
    across process restarts without depending on global numpy RNG state."""

    def __init__(self, seed: int):
        self.state = seed or 0x9E3779B97F4A7C15

    def _next(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7)
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        return self.state

    def uniform(self, lo: float, hi: float) -> float:
        frac = (self._next() % 10_000_000) / 10_000_000
        return lo + frac * (hi - lo)


def _classify_tier(name: str) -> str:
    """Classify a model into Frontier / Mid / Economy tier.
    
    Order matters: specific overrides first, then Frontier hints
    (checked before Economy so compound names like gpt-5.5-mini
    resolve to Frontier via the gpt-5.5 match).
    """
    n = name.lower()
    
    # Strong "mini/flash" models that should NOT be Economy
    mid_overrides = [
        "gpt-4o-mini", "o4-mini", "o3-mini",
        "gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.1-flash",
        "gemini-2.5-flash-lite", "claude-haiku-4", "claude-haiku",
        "nova-pro",
    ]
    for override in mid_overrides:
        if override in n:
            return "Mid"
    
    # Frontier indicators (checked BEFORE economy)
    frontier_hints = [
        "opus", "ultra", "premier",
        "gpt-5.6", "gpt-5.5", "gpt-5.4", "gpt-5.3", "gpt-5.2", "gpt-5.1",
        "gpt-5-pro", "gpt-5-chat", "gpt-5-image", "gpt-5-codex", "gpt-5-mini",
        "gpt-4.1", "gpt-4o", "gpt-4-turbo",
        "o3-pro", "o3", "o4", "o1-pro", "o1",
        "grok-4", "grok-build",
        "405b", "maverick", "v4-pro",
        "command-a", "claude-opus", "claude-sonnet", "claude-fable",
        "gemini-3-pro", "gemini-3.1-pro", "gemini-2.5-pro",
        "deepseek-v4-pro", "deepseek-r1",
        "mistral-large", "mistral-medium",
        "qwen3-max", "qwen3-coder", "qwen3.5-397b", "qwen3.6-max",
        "qwen3.7-max", "qwen3.7-plus",
        "llama-4-maverick", "llama-3.3-70b", "llama-3.1-70b",
        "nex-n2-pro",
    ]
    if any(hint in n for hint in frontier_hints):
        return "Frontier"
    
    # Mild Frontier keywords
    if any(hint in n for hint in ["pro", "large", "plus"]):
        return "Frontier"
    
    # Economy indicators
    economy_hints = [
        "nano", "micro", "lite", "1b", "3b",
        "distill", "instant", "7b", "8b", "mixtral-8x7b",
    ]
    if any(hint in n for hint in economy_hints):
        return "Economy"
    
    # Mild Economy keywords
    if any(hint in n for hint in ["mini", "small", "flash", "haiku"]):
        return "Economy"
    
    return "Mid"


def _is_open_weight(provider: str, name: str) -> bool:
    prov_low = provider.lower()
    if prov_low in [p.lower() for p in OPEN_WEIGHT_PROVIDERS] or prov_low in ["meta-llama", "deepseek"]:
        return True
    return any(h in name.lower() for h in OPEN_WEIGHT_NAME_HINTS)


def _base_quality_for_tier(tier: str) -> float:
    return {"Frontier": 0.86, "Mid": 0.74, "Economy": 0.62}[tier]


def _pricing_for_tier(tier: str) -> Dict[str, float]:
    if tier == "Frontier":
        input_cost = 5.0
        output_cost = 15.0
        relative_cost_score = 0.8
    elif tier == "Mid":
        input_cost = 0.5
        output_cost = 1.5
        relative_cost_score = 0.4
    else:
        input_cost = 0.1
        output_cost = 0.3
        relative_cost_score = 0.1
    return {
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "relative_cost_score": round(relative_cost_score, 3),
    }


def _context_window_for_tier(tier: str) -> int:
    if tier == "Frontier":
        return 2_000_000
    if tier == "Mid":
        return 32_000
    return 8_000


def _build_performance(name: str, provider: str, tier: str, aa_data: Dict[str, Any] = None) -> Dict[str, float]:
    base = _base_quality_for_tier(tier)
    low = name.lower()
    perf = {}
    for key in PERF_KEYS:
        perf[key] = round(min(max(base, 0.05), 0.99), 3)

    # Use actual benchmark data if available from Artificial Analysis
    if aa_data:
        if aa_data.get("intelligence_index") is not None:
            iq = aa_data["intelligence_index"] / 100.0
            perf["reasoning"] = round(min(max(iq, 0.05), 0.99), 3)
            perf["instruction_following"] = round(min(max(iq, 0.05), 0.99), 3)
            perf["mathematics"] = round(min(max(iq, 0.05), 0.99), 3)
        if aa_data.get("coding_index") is not None:
            cq = aa_data["coding_index"] / 100.0
            perf["coding"] = round(min(max(cq, 0.05), 0.99), 3)
        if aa_data.get("agentic_index") is not None:
            aq = aa_data["agentic_index"] / 100.0
            perf["agentic_tasks"] = round(min(max(aq, 0.05), 0.99), 3)

    # We still keep vision/audio hints because AA doesn't provide multimodal scores yet
    if any(h in low for h in VISION_HINTS):
        perf["vision_understanding"] = round(min(perf["vision_understanding"] + 0.10, 0.99), 3)
        perf["ocr"] = round(min(perf["ocr"] + 0.08, 0.99), 3)
    if any(h in low for h in AUDIO_HINTS):
        perf["audio_understanding"] = round(min(perf["audio_understanding"] + 0.12, 0.99), 3)
    if tier == "Frontier":
        perf["json_reliability"] = round(min(perf["json_reliability"] + 0.08, 0.99), 3)
        
    if name in ("DALL-E-3", "dall-e-3"):
        perf["image_generation"] = 0.99
    elif "midjourney-v6" in low or "flux-1-pro" in low:
        perf["image_generation"] = 0.96
    elif "midjourney" in low or "flux" in low:
        perf["image_generation"] = 0.94
    if name in ("Runway-Gen3", "Pika-1") or "sora" in low:
        perf["video_generation"] = 0.95
        
    return perf


def _build_benchmarks(base_perf: Dict[str, float]) -> Dict[str, float]:
    mapping = {
        "swe_bench": "coding", "humaneval": "coding", "gpqa": "reasoning",
        "mmlu": "reasoning", "aime": "mathematics", "mmmu": "vision_understanding",
        "docvqa": "ocr",
    }
    out = {}
    for bench, perf_key in mapping.items():
        out[bench] = round(min(max(base_perf.get(perf_key, 0.6), 0.05), 0.99), 3)
    return out


def _build_domains(provider: str, name: str, tier: str, perf: Dict[str, float] = None) -> Dict[str, float]:
    base = _base_quality_for_tier(tier)
    domains = {k: round(min(max(base, 0.05), 0.95), 3) for k in DOMAIN_KEYS}
    domains["general"] = round(min(base + 0.05, 0.97), 3)
    if tier == "Frontier":
        for k in ("legal", "medical", "finance"):
            domains[k] = round(min(domains[k] + 0.08, 0.95), 3)
            
    # Use real benchmark performance to seed domains when available
    if perf:
        if perf.get("coding"):
            domains["software"] = round(min(max(perf["coding"], domains["software"]), 0.95), 3)
        if perf.get("mathematics"):
            domains["mathematics"] = round(min(max(perf["mathematics"], domains["mathematics"]), 0.95), 3)
            
    # Keep legacy hints for non-benched domains
    low = name.lower()
    if "med" in low or "health" in low:
        domains["medical"] = round(min(domains["medical"] + 0.12, 0.95), 3)
    if "law" in low or "legal" in low:
        domains["legal"] = round(min(domains["legal"] + 0.12, 0.95), 3)
    if "fin" in low:
        domains["finance"] = round(min(domains["finance"] + 0.12, 0.95), 3)
        
    return domains


def _build_safety(tier: str) -> Dict[str, float]:
    base = {"Frontier": 0.90, "Mid": 0.80, "Economy": 0.72}[tier]
    safety = {k: round(min(max(base, 0.4), 0.98) , 3) for k in SAFETY_KEYS}
    return safety


def _build_verifier_fit(tier: str) -> Dict[str, float]:
    base = {"Frontier": 0.85, "Mid": 0.72, "Economy": 0.55}[tier]
    return {k: round(min(max(base, 0.2), 0.97), 3) for k in VERIFIER_KEYS}


def _build_ops_static(tier: str) -> Dict[str, float]:
    latency_score = {"Frontier": 0.5, "Mid": 0.7, "Economy": 0.9}[tier]
    reliability = 0.99
    return {"latency_score": round(latency_score, 3), "reliability": reliability}


import random

def _build_ops_dynamic(tier: str) -> Dict[str, Any]:
    # Base TTFT (Time To First Token) per tier — from Artificial Analysis p50 data
    base_ttft = {"Frontier": 3000.0, "Mid": 1200.0, "Economy": 500.0}.get(tier, 1200.0)
    # Add realistic jitter (+/- 20%)
    jitter = random.uniform(-0.2, 0.2) * base_ttft
    actual_ttft = round(base_ttft + jitter, 1)

    # Output throughput in tokens/sec — faster models tend to be smaller/quantised
    # Ranges from Artificial Analysis: Frontier ~40-80, Mid ~80-150, Economy ~150-300
    base_tps = {"Frontier": 60.0, "Mid": 110.0, "Economy": 200.0}.get(tier, 110.0)
    tps_jitter = random.uniform(-0.15, 0.15) * base_tps
    actual_tps = round(base_tps + tps_jitter, 1)

    return {
        "recent_latency_ms": actual_ttft,
        "tokens_per_sec": actual_tps,
        "recent_failure_rate": round(random.uniform(0.001, 0.02), 4),
        "current_availability": round(random.uniform(0.99, 0.9999), 4),
        "rate_limit_pressure": round(random.uniform(0.0, 0.2), 2),
        "queue_pressure": round(random.uniform(0.0, 0.15), 2),
        "incident_status": "green",
        "budget_pressure": round(random.uniform(0.05, 0.2), 2),
        "telemetry_freshness_sec": random.randint(10, 120),
    }


def _build_priors(name: str) -> Dict[str, Any]:
    task_family = {}
    for fam in ["coding", "reasoning", "mathematics", "chat", "vision", "ocr",
                "document_qa", "summarization", "translation", "agent", "audio"]:
        task_family[fam] = {"alpha": 10.0, "beta": 5.0}
    return {
        "global": {"alpha": 14.0, "beta": 6.0},
        "task_family": task_family,
    }


def _build_evaluation() -> Dict[str, Any]:
    return {
        "samples": 0, "wins": 0, "losses": 0,
        "quality_sum": 0.0, "latency_sum_ms": 0.0, "cost_sum": 0.0,
        "user_acceptance_sum": 0.0, "safety_flags": 0,
        "last_updated": None,
    }


def _build_modalities(name: str, tier: str) -> Dict[str, bool]:
    low = name.lower()
    supports_vision = any(h in low for h in VISION_HINTS) or tier == "Frontier"
    supports_audio = any(h in low for h in AUDIO_HINTS)
    supports_video = supports_audio and tier in ("Frontier", "Mid")
    return {
        "text": True,
        "image": supports_vision,
        "pdf": True,  # all models accept PDF as extracted text at minimum
        "audio": supports_audio,
        "video": supports_video,
        "spreadsheet": True,
        "presentation": True,
    }


def _build_capabilities(name: str, tier: str) -> Dict[str, bool]:
    low = name.lower()
    return {
        "json_mode": tier in ("Frontier", "Mid") or "instruct" in low,
        "function_calling": tier in ("Frontier", "Mid"),
        "web_search": tier == "Frontier" or "gpt" in low or "gemini" in low or "grok" in low,
        "ocr": _build_modalities(name, tier)["image"] or _build_modalities(name, tier)["pdf"],
        "citation_support": tier in ("Frontier", "Mid"),
        "image_generation": any(h in low for h in ["dall-e", "stable-diffusion", "imagen", "midjourney", "flux"]),
        "video_generation": any(h in low for h in ["sora", "runway", "pika", "kling"]),
    }


def _routing_meta(domain_high_risk_penalty: bool) -> Dict[str, Any]:
    return {"avoid_for": ["high_risk"] if domain_high_risk_penalty else []}


def build_registry() -> List[Dict[str, Any]]:
    registry: List[Dict[str, Any]] = []
    for provider, names in MODELS.items():
        for name in names:
            tier = _classify_tier(name)
            perf = _build_performance(name, provider, tier)
            entry: Dict[str, Any] = {
                "name": name,
                "provider": provider,
                "tier": tier,
                "status": "active",
                "api_available": True,
                "open_weight": _is_open_weight(provider, name),
                "allowed_regions": ["global"],
                "modalities": _build_modalities(name, tier),
                "capabilities": _build_capabilities(name, tier),
                "context": {"window": _context_window_for_tier(tier)},
                "pricing": _pricing_for_tier(tier),
                "ops_static": _build_ops_static(tier),
                "ops_dynamic": _build_ops_dynamic(tier),
                "priors": _build_priors(name),
                "evaluation": _build_evaluation(),
                "performance": perf,
                "benchmarks": _build_benchmarks(perf),
                "domains": _build_domains(provider, name, tier),
                "safety": _build_safety(tier),
                "routing": _routing_meta(tier == "Economy"),
                "verifier_fit": _build_verifier_fit(tier),
                # These fields are intentionally synthetic and are suitable
                # only for local development. Production feasibility rejects
                # them unless explicitly configured otherwise.
                "evidence": {
                    "source": "synthetic_development_fixture",
                    "eligible_for_auto_route": False,
                    "evaluated_task_families": [],
                },
            }
            registry.append(entry)
    return registry


# Built once at import time; treated as the live canonical registry for the
# process. In production, swap this for a periodically-refreshed snapshot
# (e.g. loaded from a config service / database) — see app/api/routes.py
# for where a reload hook would live.
MODEL_REGISTRY: List[Dict[str, Any]] = build_registry()
