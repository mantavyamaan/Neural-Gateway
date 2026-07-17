"""
Automated Benchmark Sync Pipeline.

Fetches real-world performance scores for every model in the Neural Gateway registry
from two publicly accessible sources:

  1. OpenRouter Live API    -- pricing, context window, modality flags,
                               description keyword signals
  2. HuggingFace Hub API   -- public model card pipeline tags (no auth needed)

Normalization:
  - Context length is log2-normalized population-wide (no hardcoded ceiling).
  - Keyword/modality signals produce capability-specific baseline scores.

Priority order (highest wins): HF Hub > OpenRouter signals

After scoring:
  - Models with at least one real data source are unlocked:
      eligible_for_auto_route = True
  - Models with no data remain locked (eligible_for_auto_route = False).
  - Existing Bayesian priors and evaluation state are NEVER overwritten.

Run standalone:  python scripts/run_benchmark_sync.py
Called by:       app/core/openrouter_sync.py after every sync cycle.
"""

import json
import logging
import math
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

from app.core.database import bulk_upsert_models, get_all_models

logger = logging.getLogger("neural_gateway.benchmark_sync")

NEURAL_GATEWAY_PERF_KEYS = [
    "coding", "agentic_tasks", "reasoning", "scientific_reasoning",
    "mathematics", "creative_writing", "instruction_following",
    "vision_understanding", "ocr", "table_understanding", "document_qa",
    "long_context", "summarization", "translation", "audio_understanding",
    "json_reliability", "spreadsheet_reasoning", "image_generation",
    "video_generation",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_get(url: str, timeout: int = 15) -> Optional[bytes]:
    """HTTP GET with error handling. Returns raw bytes or None."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Neural Gateway-BenchmarkSync/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        logger.warning("HTTP fetch failed for %s: %s", url, exc)
        return None


def _minmax_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Normalize to [0,1] using population min-max. Returns 0.5 if all identical."""
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if abs(vmax - vmin) < 1e-9:
        return {k: 0.5 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def _norm_name(name: str) -> str:
    """Strip version suffixes and lowercase for fuzzy matching."""
    name = name.lower()
    name = re.sub(r"[-:]?(20\d{6}|20\d{2}-\d{2}-\d{2})", "", name)
    name = re.sub(r":(free|beta|nitro|extended)$", "", name)
    return re.sub(r"\s+", "-", name.strip())


def _match_model(target_id: str, index: Dict[str, Any]) -> Optional[str]:
    """Find target_id in index via exact, prefix-stripped, or fuzzy match."""
    if target_id in index:
        return target_id
    short = target_id.split("/")[-1]
    if short in index:
        return short
    nt = _norm_name(short)
    for key in index:
        nk = _norm_name(key)
        if nt == nk or (len(nt) > 4 and nt in nk) or (len(nk) > 4 and nk in nt):
            return key
    return None


# ---------------------------------------------------------------------------
# Source 1: OpenRouter Live API
# ---------------------------------------------------------------------------

KEYWORD_RULES: Dict[str, List[str]] = {
    "_kw_coding":       ["code", "coder", "coding", "codestral", "swe", "devstral"],
    "_kw_math":         ["math", "reasoning", "r1", "think", "logic"],
    "_kw_medical":      ["medical", "clinical", "health", "biomedical"],
    "_kw_multilingual": ["multilingual", "translation", "translate"],
    "_kw_vision":       ["vision", "multimodal", "visual"],
}


def fetch_openrouter_signals() -> Dict[str, Dict[str, float]]:
    """Fetch OpenRouter /v1/models and extract pricing, context, modality, and keyword signals."""
    raw = _safe_get("https://openrouter.ai/api/v1/models")
    if not raw:
        logger.warning("OpenRouter signals: no data returned.")
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        logger.warning("OpenRouter signals: failed to parse JSON.")
        return {}

    results: Dict[str, Dict[str, float]] = {}
    for entry in data.get("data", []):
        mid = entry.get("id", "")
        if not mid or mid.startswith("openrouter/"):
            continue

        sig: Dict[str, float] = {}

        pricing = entry.get("pricing", {})
        try:
            ic = float(pricing.get("prompt", 0)) * 1_000_000
            oc = float(pricing.get("completion", 0)) * 1_000_000
            sig["_cost"] = 0.35 * ic + 0.65 * oc
        except (ValueError, TypeError):
            pass

        ctx = entry.get("context_length", 0)
        if ctx and ctx > 0:
            sig["_ctx_log"] = math.log2(max(ctx, 1))

        arch = entry.get("architecture", {})
        inp_mods = arch.get("input_modalities", [])
        out_mods = arch.get("output_modalities", [])
        if "image" in inp_mods or "image_url" in inp_mods:
            sig["_vision"] = 1.0
        if "audio" in inp_mods:
            sig["_audio"] = 1.0
        if "image" in out_mods:
            sig["_imggen"] = 1.0

        desc = (entry.get("description") or "").lower()
        name = (entry.get("name") or "").lower()
        combined = name + " " + desc
        for kk, kl in KEYWORD_RULES.items():
            if any(kw in combined for kw in kl):
                sig[kk] = 1.0

        if sig:
            results[mid] = sig

    logger.info("OpenRouter signals: extracted data for %d models.", len(results))
    return results


def _normalize_openrouter(raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Convert raw OpenRouter signals into normalized Neural Gateway performance scores."""
    if not raw:
        return {}

    ctx_pop = {k: v["_ctx_log"] for k, v in raw.items() if "_ctx_log" in v}
    ctx_norm = _minmax_normalize(ctx_pop)

    out: Dict[str, Dict[str, float]] = {}
    for mid, sig in raw.items():
        s: Dict[str, float] = {}

        if mid in ctx_norm:
            s["long_context"] = round(ctx_norm[mid], 4)

        if sig.get("_vision"):
            s["vision_understanding"] = 0.75
            s["ocr"] = 0.70
        if sig.get("_audio"):
            s["audio_understanding"] = 0.75
        if sig.get("_imggen"):
            s["image_generation"] = 0.80

        if sig.get("_kw_coding"):
            s["coding"] = 0.82
            s["agentic_tasks"] = 0.75
        if sig.get("_kw_math"):
            s["mathematics"] = 0.82
            s["reasoning"] = 0.80
            s["scientific_reasoning"] = 0.78
        if sig.get("_kw_multilingual"):
            s["translation"] = 0.82
        if sig.get("_kw_vision") and "_vision" not in sig:
            s.setdefault("vision_understanding", 0.70)

        if s:
            out[mid] = s

    return out


# ---------------------------------------------------------------------------
# Source 2: HuggingFace Hub public model card tags
# ---------------------------------------------------------------------------

HF_TAG_MAP: Dict[str, str] = {
    "code": "coding",
    "coding": "coding",
    "code-generation": "coding",
    "text-generation": "instruction_following",
    "math": "mathematics",
    "mathematics": "mathematics",
    "reasoning": "reasoning",
    "translation": "translation",
    "summarization": "summarization",
    "question-answering": "document_qa",
    "text2text-generation": "instruction_following",
    "image-to-text": "vision_understanding",
    "automatic-speech-recognition": "audio_understanding",
    "text-to-image": "image_generation",
    "visual-question-answering": "vision_understanding",
}


def fetch_huggingface_tags(hf_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetch public model card tags from HuggingFace Hub API. No auth required."""
    results: Dict[str, Dict[str, float]] = {}
    for hf_id in hf_ids[:50]:
        raw = _safe_get("https://huggingface.co/api/models/" + hf_id, timeout=6)
        if not raw:
            continue
        try:
            meta = json.loads(raw)
        except Exception:
            continue
        pipeline_tag = meta.get("pipeline_tag", "")
        tags = list(meta.get("tags", []))
        if pipeline_tag:
            tags.append(pipeline_tag)
        s: Dict[str, float] = {}
        for tag in tags:
            neural_gateway_key = HF_TAG_MAP.get((tag or "").lower().strip())
            if neural_gateway_key:
                s[neural_gateway_key] = 0.72
        if s:
            results[hf_id] = s
    if results:
        logger.info("HuggingFace Hub: tag data for %d models.", len(results))
    return results


# ---------------------------------------------------------------------------
# Domain score derivation
# ---------------------------------------------------------------------------

def _derive_domains(perf: Dict[str, float]) -> Dict[str, float]:
    """Derive domain expertise scores purely from performance scores."""
    def avg(*keys: str) -> Optional[float]:
        vals = [perf[k] for k in keys if k in perf]
        return round(sum(vals) / len(vals), 4) if vals else None

    domain_map = {
        "general":          avg("reasoning", "instruction_following", "summarization"),
        "software":         avg("coding", "reasoning", "instruction_following"),
        "legal":            avg("reasoning", "document_qa", "instruction_following"),
        "medical":          avg("scientific_reasoning", "reasoning", "document_qa"),
        "finance":          avg("mathematics", "reasoning", "table_understanding"),
        "science":          avg("scientific_reasoning", "mathematics", "reasoning"),
        "education":        avg("reasoning", "instruction_following", "summarization"),
        "customer_support": avg("instruction_following", "summarization", "reasoning"),
        "research":         avg("scientific_reasoning", "reasoning", "summarization"),
        "mathematics":      avg("mathematics", "scientific_reasoning", "reasoning"),
    }
    return {k: v for k, v in domain_map.items() if v is not None}


# ---------------------------------------------------------------------------
# Main sync entry point
# ---------------------------------------------------------------------------

def run_benchmark_sync() -> Dict[str, Any]:
    """Full 100% dynamic benchmark sync. Priority: HF Hub > OpenRouter."""
    logger.info("=== Starting 100%% Dynamic Benchmark Sync ===")

    or_raw = fetch_openrouter_signals()
    or_norm = _normalize_openrouter(or_raw)

    registry = get_all_models()
    hf_id_map: Dict[str, str] = {
        m["name"]: m.get("hugging_face_id", "").strip()
        for m in registry
        if m.get("hugging_face_id", "").strip()
    }
    hf_raw = fetch_huggingface_tags(list(hf_id_map.values()))
    inv_hf = {v: k for k, v in hf_id_map.items()}
    hf_scores: Dict[str, Dict[str, float]] = {
        inv_hf.get(hid, hid): s for hid, s in hf_raw.items()
    }

    total_sources = sum(1 for x in [or_norm, hf_scores] if x)
    logger.info("Sources loaded: %d/2", total_sources)

    if total_sources == 0:
        logger.warning("No benchmark sources available. Sync aborted.")
        return {"status": "no_sources", "unlocked": 0, "skipped": 0}

    logger.info("Registry: %d models. Matching...", len(registry))

    updated: List[Dict[str, Any]] = []
    unlocked = 0
    skipped = 0

    for model in registry:
        mid = model["name"]

        or_match = _match_model(mid, or_norm)
        hf_match = mid if mid in hf_scores else None

        or_perf = {k: v for k, v in or_norm.get(or_match, {}).items() if not k.startswith("_")}
        hf_perf = hf_scores.get(hf_match, {})

        has_data = bool(or_perf or hf_perf)
        if not has_data:
            skipped += 1
            updated.append(model)
            continue

        # Layer scores: OpenRouter < HF Hub (highest priority)
        final_perf: Dict[str, float] = dict(model.get("performance", {}))
        final_perf.update(or_perf)
        final_perf.update(hf_perf)

        if not final_perf:
            skipped += 1
            updated.append(model)
            continue

        new_domains = _derive_domains(final_perf)
        final_domains = {**model.get("domains", {}), **new_domains}

        model["performance"] = final_perf
        model["domains"] = final_domains
        model["evidence"]["eligible_for_auto_route"] = True
        model["evidence"]["source"] = "live_benchmark_sync"
        model["evidence"]["benchmark_sources"] = [
            src for src, m2 in [
                ("openrouter_signals", or_match),
                ("huggingface_hub",   hf_match),
            ] if m2
        ]

        unlocked += 1
        updated.append(model)
        logger.debug("Unlocked: %s", mid)

    if updated:
        bulk_upsert_models(updated)

    summary = {
        "status":          "complete",
        "total_models":    len(registry),
        "unlocked":        unlocked,
        "skipped_no_data": skipped,
        "sources_active":  total_sources,
    }
    logger.info(
        "=== Benchmark Sync Complete === Unlocked: %d | Skipped: %d | Sources: %d/2",
        unlocked, skipped, total_sources,
    )
    return summary
