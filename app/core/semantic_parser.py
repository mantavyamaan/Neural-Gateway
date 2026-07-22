"""
Deterministic + Semantic Parsing.

Splits request understanding into two independent tracks:
  - deterministic_extract(): hard, keyword-based constraint extraction
    (JSON mode, function calling, web search, OCR, citations, verifier).
  - fallback_structured_semantic_parse(): heuristic semantic classifier
    (task family, domain, risk, workflow decomposition). This is the
    default "parser". A real structured-output LLM parser can be plugged
    in via `llm_parser` in parse_task_request()/route() without changing
    any downstream stage.

parse_task_request() merges artifact inspection + deterministic facts +
semantic inference into the single canonical TaskFeatures object that the
rest of the pipeline consumes.
"""

import re
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from app.config import SUPPORTED_FORMATS
from app.core.artifact_inspection import detect_conflicts, inspect_artifacts
from app.core.embedding_parser import parse_prompt_to_semantic_struct
from app.core.database import get_all_feedback
from app.models.schemas import (
    ArtifactProfile,
    RequestConstraints,
    StructuredSemanticParse,
    TaskFeatures,
    TenantContext,
)


def _word_match(text: str, keywords: List[str]) -> bool:
    """Match keywords using word boundaries to avoid false positives."""
    for kw in keywords:
        if re.search(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', text, re.IGNORECASE):
            return True
    return False


def deterministic_extract(prompt: str, input_formats: List[str], estimated_tokens: int) -> Dict[str, Any]:
    p = prompt.lower()
    required_formats = sorted(set(input_formats) | {"text"})
    return {
        "required_formats": required_formats,
        "requires_json": _word_match(p, ["return json", "respond in json", "json output", "json schema", "structured json"]),
        "requires_function_calling": _word_match(p, ["tool", "function call", "call a tool", "api call", "use tools"]),
        "requires_web_search": _word_match(p, ["latest", "today", "current", "search the web", "news", "real-time"]),
        "requires_ocr": (
            any(fmt in input_formats for fmt in ["pdf", "image"])
            and _word_match(p, ["scan", "extract text", "read the document", "ocr", "handwritten", "digitize"])
        ),
        "requires_citations": _word_match(p, ["cite", "citations", "sources", "references", "bibliography"]),
        "requires_verifier": _word_match(p, ["verify", "double-check", "validator", "review", "compliance", "audit"]),
        "min_context_window": estimated_tokens,
    }


def infer_complexity(prompt: str, estimated_tokens: int, semantic: StructuredSemanticParse) -> str:
    """Conservative task-complexity classification independent of token count."""
    if estimated_tokens > 150_000:
        return "high"
    if estimated_tokens > 30_000:
        return "medium"

    p = prompt.lower()
    
    # Regex detection for magnitude and explicit advanced concepts
    if re.search(r'\b\d+[- ]page(s)?\b', p) or re.search(r'\b\d+[- ]word(s)?\b', p) or re.search(r'\b(advanced|massive|huge|expert|in-depth)\b', p):
        return "high"
    high_signals = [
        "entire codebase", "architecture", "multi-file", "multiple files",
        "production ready", "production-ready", "migration", "distributed",
        "concurrent", "refactor", "end-to-end", "e2e", "security audit",
        "threat model", "integration test", "deployment", "all these problems",
        "complicated", "highly difficult", "extremely difficult", "very hard", 
        "hard problem", "complex task", "difficult problem", "a lot of like this",
        "quantum mechanics", "string theory", "advanced physics", "deep learning"
    ]
    medium_signals = [
        "implement", "debug", "review", "test", "design", "analyze",
        "compare", "several", "requirements", "complex",
        "explain", "detail", "theory", "concept", "mechanics", "how does", "why is",
        "difference between", "meaning of"
    ]
    high_count = sum(signal in p for signal in high_signals)
    medium_count = sum(signal in p for signal in medium_signals)
    
    # Fix: require 2 high signals normally, OR 1 signal if it's an inherently
    # complex task family — the previous condition had a logically dead second
    # clause (high_count >= 1 already covers high_count > 0, making the 'and'
    # branch unreachable).
    if high_count >= 2 or (high_count >= 1 and semantic.primary_family in {"coding", "agent", "reasoning"}):
        return "high"
    if medium_count >= 2 or semantic.decomposition_needed or len(prompt) > 200:
        return "medium"
    return "low"


def validate_structured_parse(data: StructuredSemanticParse) -> StructuredSemanticParse:
    allowed_families = {
        "coding", "reasoning", "mathematics", "chat", "vision",
        "ocr", "document_qa", "summarization", "translation", "agent", "audio",
        "video_generation", "image_generation"
    }
    if data.primary_family not in allowed_families:
        raise ValueError(f"Invalid primary family: {data.primary_family}")
    if data.risk_tier not in {"low", "medium", "high", "extreme"}:
        raise ValueError(f"Invalid risk tier: {data.risk_tier}")
    if data.expected_output not in {"structured_json", "free_text", "code"}:
        raise ValueError(f"Invalid expected output: {data.expected_output}")
    if not (0.0 <= data.ambiguity_score <= 1.0):
        raise ValueError("Ambiguity score must be in [0, 1].")
    return data


def infer_workflow_profile(primary_family: str, domain: str, input_formats: List[str], prompt: str, complexity: str = "low") -> str:
    p = prompt.lower()
    if "support" in p or domain == "customer_support":
        return "customer_support_summarization"
    if domain == "legal" and "contract" in p:
        return "contract_review_intake"
    if primary_family == "coding":
        return "coding_assistant"
    if primary_family == "ocr" and any(fmt in input_formats for fmt in ["image", "pdf"]):
        return "invoice_ocr_pipeline"
    if primary_family == "audio":
        return "real_time_voice_agent" if "real-time" in p else "audio_summary"
    if primary_family == "translation":
        return "multilingual_chat"
    if domain == "research":
        return "research_drafting"
    
    # Default to quality_first for high complexity tasks as per user request
    if complexity == "high":
        return "quality_first"
    if complexity == "medium":
        return "balanced"
        
    return "budget_first"


def parse_task_request(
    prompt: str,
    input_formats: Optional[List[str]] = None,
    estimated_tokens: int = 2000,
    estimated_output_tokens: int = 1200,
    artifact_hints: Optional[List[Dict[str, Any]]] = None,
    llm_parser: Optional[Callable] = None,
    request_constraints: Optional[RequestConstraints] = None,
    tenant_context: Optional[TenantContext] = None,
    files: Optional[List[str]] = None,
) -> TaskFeatures:
    rc = request_constraints or RequestConstraints()
    tc = tenant_context or TenantContext()

    # ---- Determine artifacts and formats ----
    if files:
        artifacts = inspect_artifacts(prompt=prompt, artifact_hints=artifact_hints, files=files)
        input_formats = sorted({a.format for a in artifacts} | {"text"})
    else:
        input_formats = input_formats or ["text"]
        bad = [f for f in input_formats if f not in SUPPORTED_FORMATS]
        if bad:
            raise ValueError(f"Unsupported formats: {bad}. Allowed: {SUPPORTED_FORMATS}")
        artifacts = inspect_artifacts(input_formats=input_formats, prompt=prompt,
                                      artifact_hints=artifact_hints)

    # (format validation for user-supplied formats is done above inside the else branch)

    # ---- Conflict detection + semantic aggregation ----
    conflict_flags = detect_conflicts(prompt, artifacts)
    previews = [a.extracted_text_preview for a in artifacts if a.extracted_text_preview]
    extracted_text_summary = " ".join(previews)[:2000] if previews else None
    detected_languages = sorted({a.detected_language for a in artifacts if a.detected_language})
    total_file_size = sum(a.file_size_bytes or 0 for a in artifacts)
    estimated_tokens += total_file_size // 5

    hard = deterministic_extract(prompt, input_formats, estimated_tokens)
    if not hard["requires_ocr"]:
        for a in artifacts:
            if a.format in {"pdf", "image"} and (a.scan_likelihood or 0) >= 0.6:
                hard["requires_ocr"] = True
                break

    # Check memory bank first
    feedback_examples = get_all_feedback()
    soft = None
    for ex in feedback_examples:
        if ex["prompt"].lower() == prompt.lower():  # exact match only — prevent dangerously broad substring hits
            primary = ex["correct_family"]
            soft = StructuredSemanticParse(
                primary_family=primary,
                secondary_families=[],
                required_stages=[primary],
                workflow_graph=[{
                    "stage_id": 1,
                    "stage_name": primary,
                    "depends_on": [],
                    "fallbacks_prepared": True,
                }],
                domain="general",
                risk_tier="low",
                risk_type="standard",
                expected_output="free_text",
                ambiguity_score=0.10,
                actionability="advisory",
                document_type="generic",
                decomposition_needed=False,
                needs_verification=False,
                parser_confidence=0.99,
                reason_summary=f"Routed via Heuristic Parser (Learned from Memory Bank). Matched historical correction: '{ex['prompt']}'."
            )
            break
            
    if soft is None:
        soft = parse_prompt_to_semantic_struct(prompt)
        
        # Family alias normalization: map non-canonical names to valid schema values
        FAMILY_ALIASES = {
            "math": "mathematics",
            "general": "chat",       # 'general' is not a valid family
            "extraction": "extraction",
            "creative_writing": "creative",
            "agentic": "agent",
        }
        if soft.primary_family in FAMILY_ALIASES:
            soft.primary_family = FAMILY_ALIASES[soft.primary_family]
        
        # Populate stages and graph if missing
        required_stages = soft.required_stages if soft.required_stages else [soft.primary_family]
        if soft.primary_family == "ocr" and "document_qa" not in required_stages:
            required_stages.append("document_qa")
        
        if not soft.workflow_graph:
            workflow_graph = []
            for i, stage in enumerate(required_stages):
                workflow_graph.append({
                    "stage_id": i + 1,
                    "stage_name": stage,
                    "depends_on": [] if i == 0 else [i],
                })
            soft.workflow_graph = workflow_graph
            soft.required_stages = required_stages


    # ---- Topic-driven domain/document override ----
    from app.core.domain_classifier import classify_domain, RISK_METADATA, LEARNED_DOMAIN_CONFIDENCE_THRESHOLD
    
    inferred_domain, conf = classify_domain(prompt)
    if inferred_domain and conf >= LEARNED_DOMAIN_CONFIDENCE_THRESHOLD:
        if soft.domain in ("general", "generic") or soft.domain is None:
            soft.domain = inferred_domain
    else:
        inferred_topics = [a.inferred_topic for a in artifacts if a.inferred_topic]
        if inferred_topics and soft.domain == "general":
            topic_to_domain = {
                "legal_contract": ("legal", "high", "regulated_advice", "contract"),
                "financial_document": ("finance", "high", "regulated_advice", "financial_document"),
                "medical_record": ("medical", "high", "regulated_advice", "medical_record"),
                "security_report": ("security", "high", "security_sensitive", "security_report"),
                "research_paper": ("research", "low", "standard", "research_paper"),
                "support_record": ("customer_support", "low", "standard", "support_record"),
            }
            topic = inferred_topics[0]
            if topic in topic_to_domain:
                soft.domain, soft.risk_tier, soft.risk_type, soft.document_type = topic_to_domain[topic]

    if soft.domain in RISK_METADATA:
        rt, rtype, dtype = RISK_METADATA[soft.domain]
        soft.risk_tier = rt
        soft.risk_type = rtype
        soft.document_type = dtype

    # Domain normalization: 'software' maps to 'general' for routing purposes
    # (identical safety profile; prevents coding tasks being labelled 'software' domain)
    if soft.domain == "software":
        soft.domain = "general"

    if getattr(soft, "complexity", None) and soft.complexity != "unknown":
        complexity = soft.complexity
    else:
        complexity = infer_complexity(prompt, estimated_tokens, soft)
        
    if "audio" in input_formats:
        soft.primary_family = "audio"
        if "audio" not in soft.required_stages:
            soft.required_stages.insert(0, "audio")

    # Only escalate risk for complex tasks in already safety-sensitive domains
    # (prevents coding/math tasks from being falsely escalated to high risk)
    if complexity == "high" and soft.domain in {"medical", "legal", "finance", "security"} and soft.risk_tier == "low":
        soft.risk_tier = "medium"

    # Risk calibration: safe task families in non-sensitive domains default to low risk.
    # The LR/heuristics may over-predict 'medium' or 'high' when training data is noisy.
    SAFE_FAMILIES = {"coding", "mathematics", "translation", "summarization",
                     "extraction", "creative", "chat", "document_qa", "reasoning"}
    SENSITIVE_DOMAINS = {"medical", "legal", "finance", "security"}
    if (soft.primary_family in SAFE_FAMILIES
            and soft.domain not in SENSITIVE_DOMAINS
            and soft.risk_tier in {"medium", "high"}
            and soft.risk_type not in {"regulated_advice", "security_sensitive", "pii_sensitive"}):
        soft.risk_tier = "low"

        
    workflow_profile = infer_workflow_profile(soft.primary_family, soft.domain, input_formats, prompt, complexity)
    requires_verifier = (
        hard["requires_verifier"]
        or soft.needs_verification
        or soft.risk_tier in {"high", "extreme"}
        or rc.mandatory_verifier
    )
    safety_sensitive = soft.risk_tier == "high" or soft.risk_type in {"regulated_advice", "security_sensitive"}
    if rc.no_web_access and rc.require_web_search:
        raise ValueError("Contradictory constraints: no_web_access and require_web_search cannot both be true.")
    if rc.no_web_access:
        hard["requires_web_search"] = False
    return TaskFeatures(
        raw_prompt=prompt,
        input_formats=sorted(set(input_formats) | {"text"}),
        estimated_tokens=estimated_tokens,
        estimated_output_tokens=estimated_output_tokens,
        artifacts=artifacts,
        primary_family=soft.primary_family,
        secondary_families=soft.secondary_families,
        required_stages=soft.required_stages,
        workflow_graph=soft.workflow_graph,
        complexity=complexity,
        domain=soft.domain,
        risk_tier=soft.risk_tier,
        risk_type=soft.risk_type,
        required_formats=hard["required_formats"],
        requires_json=hard["requires_json"] or rc.require_json,
        requires_function_calling=hard["requires_function_calling"],
        requires_web_search=hard["requires_web_search"] or rc.require_web_search,
        requires_ocr=hard["requires_ocr"] or rc.require_ocr,
        requires_citations=hard["requires_citations"] or rc.require_citations,
        requires_image_generation=soft.primary_family == "image_generation",
        requires_video_generation=soft.primary_family == "video_generation",
        requires_verifier=requires_verifier,
        min_context_window=hard["min_context_window"],
        expected_output=soft.expected_output,
        ambiguity_score=soft.ambiguity_score,
        safety_sensitive=safety_sensitive,
        actionability=soft.actionability,
        document_type=soft.document_type,
        decomposition_needed=soft.decomposition_needed,
        workflow_profile=workflow_profile,
        parser_confidence=soft.parser_confidence,
        conflict_flags=conflict_flags,
        extracted_text_summary=extracted_text_summary,
        detected_languages=detected_languages,
        total_file_size_bytes=total_file_size,
        request_constraints=rc,
        tenant_context=tc,
    )
