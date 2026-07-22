"""
Neural Gateway — system constants and versioning.

Every production routing system must be reproducible under audit. Each
independently-evolving component carries its own version stamp so that a
routing decision questioned weeks later can be reconstructed exactly.

Version axes:
    ROUTER_VERSION      -> orchestration logic (the route() pipeline)
    PARSER_VERSION       -> semantic interpretation layer
    POLICY_VERSION        -> governance rules
    SCORING_VERSION        -> utility / quality math
    REGISTRY_VERSION         -> canonical model registry snapshot
    CALIBRATION_VERSION       -> Bayesian prior calibration snapshot
    TELEMETRY_SNAPSHOT_VERSION -> runtime telemetry snapshot
"""

import os
from typing import List, Optional

from dotenv import load_dotenv

# Populate os.environ from the project's .env file (repo root) so that
# NEURAL_GATEWAY_ALLOWED_PROVIDERS / other config vars edited there are picked up
# without needing to `export` them in the shell. Values already set in
# the process environment win over the file, matching dotenv's default.
load_dotenv()


def _parse_provider_env(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    providers = [p.strip() for p in raw.split(",") if p.strip()]
    return providers or None


# Default provider allowlist applied to every /route request when the caller
# doesn't set its own `request_constraints.allowed_providers`. Callers can
# still override per-request. Set to e.g. "OpenAI" or "Anthropic" (or
# "OpenAI,Anthropic") when you only hold API keys for those providers — the
# router then only selects models (primary + fallbacks + verifiers) from that
# subset. Leave unset for the full multi-provider behavior.
DEFAULT_ALLOWED_PROVIDERS: Optional[List[str]] = _parse_provider_env(
    os.getenv("NEURAL_GATEWAY_ALLOWED_PROVIDERS")
)


ROUTER_VERSION = "neural_gateway-router-v1.0.0"
PARSER_VERSION = "parser-v1.0.0"
POLICY_VERSION = "policy-v1.0.0"
SCORING_VERSION = "scoring-v1.0.0"
CALIBRATION_VERSION = "calibration-v1.0.0"
REGISTRY_VERSION = "registry-v1.0.0-2026-06-snapshot"
TELEMETRY_SNAPSHOT_VERSION = "telemetry-snap-2026-06-27"

SUPPORTED_FORMATS = [
    "text", "image", "pdf", "audio", "video", "spreadsheet", "presentation"
]

# Escalation ladder: below ABSTAIN the router refuses to answer; below
# ESCALATE it answers but flags for human review; above HIGH it proceeds
# autonomously even on high-risk tasks. All three can be overridden per
# deployment via env vars (defaults kept for backward compat).
#
# The default ABSTAIN threshold (0.40) assumes a multi-provider registry
# where the top model naturally wins a large share of Thompson-Sampling
# simulations. Narrower candidate pools (e.g. single-provider setups via
# NEURAL_GATEWAY_ALLOWED_PROVIDERS) produce lower peak confidences because models
# within one family score similarly — set NEURAL_GATEWAY_ABSTAIN_THRESHOLD=0.30
# (or lower) to keep the router from abstaining on those.
def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


CONFIDENCE_ABSTAIN_THRESHOLD = _float_env("NEURAL_GATEWAY_ABSTAIN_THRESHOLD", 0.40)
CONFIDENCE_ESCALATE_THRESHOLD = _float_env("NEURAL_GATEWAY_ESCALATE_THRESHOLD", 0.55)
CONFIDENCE_HIGH_THRESHOLD = _float_env("NEURAL_GATEWAY_HIGH_THRESHOLD", 0.75)
PARSER_ESCALATE_THRESHOLD = _float_env("NEURAL_GATEWAY_PARSER_ESCALATE_THRESHOLD", 0.65)

# Safe-by-default production behaviour. Entries derived only from name-based
# heuristics are useful for local demos, but must never be presented as an
# evidence-based automatic production choice.
REQUIRE_MEASURED_EVIDENCE = _bool_env("NEURAL_GATEWAY_REQUIRE_MEASURED_EVIDENCE", True)
ALLOW_SERVER_FILE_PATHS = _bool_env("NEURAL_GATEWAY_ALLOW_SERVER_FILE_PATHS", False)
ADMIN_API_KEY = os.getenv("NEURAL_GATEWAY_ADMIN_API_KEY")
CORS_ORIGINS = [
    origin.strip() for origin in os.getenv(
        "NEURAL_GATEWAY_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",") if origin.strip()
]

# Embedding model for the domain classifier (loaded once at import time
# by app/core/domain_classifier.py). Any sentence-transformers-compatible
# model works; default is a 384-dim MiniLM (~90MB, English-focused).
NEURAL_GATEWAY_EMBEDDING_MODEL = os.getenv("NEURAL_GATEWAY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

