"""
embedding_parser.py
Pure vector-embedding semantic parser. No LLM, no PyTorch.
fastembed (ONNX) + numpy. Sub-millisecond KNN after encode.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

from app.models.schemas import StructuredSemanticParse

# Try importing fastembed gracefully
try:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


CATEGORICAL_FIELDS = (
    "primary_family", "domain", "risk_tier",
    "risk_type", "expected_output", "document_type", "complexity"
)
BOOLEAN_FIELDS = ("decomposition_needed", "needs_verification")

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "extreme": 3}

# ----------------------- Safety override layer ------------------------
# Deterministic, monotonic: can only RAISE risk, never lower it.
# (tier, risk_type, domain-hint) triggered by lexical patterns.

# Tier 1: ALWAYS active — hard override regardless of ML confidence
TIER1_RULES: list[tuple[re.Pattern, str, str, str | None]] = [
    (re.compile(r'\b(suicid\w+|self.?harm|kill (myself|himself|herself))\b', re.I),
     'extreme', 'regulated_advice', 'medical'),
    (re.compile(r'\b(drop table|rm -rf|delete (all|every)|wipe (the )?database)\b', re.I),
     'high', 'operational', 'security'),
    (re.compile(r'\b(api.?key|private key|secret[ ]?key|credentials?|exploit|sql injection|xss|privilege escalation|bypass auth)\b', re.I),
     'high', 'security_sensitive', 'security'),
    (re.compile(r'\b(chest pain|overdose|suicidal ideation)\b', re.I),
     'high', 'regulated_advice', 'medical'),
    # High-confidence legal signals — very specific terms unlikely to appear in non-legal contexts
    (re.compile(r'\b(liabilit\w+|non.?compete|nda\b|indemnif\w+|breach of contract|sue\b|lawsuit)\b', re.I),
     'high', 'regulated_advice', 'legal'),
    # High-confidence medical signals — drug interactions, dosage questions
    (re.compile(r'\b(drug interaction|medication.*interact|blood thinner|dangerous.*medication|allergic reaction|symptom(s)? of|what (are|is) the symptom)\b', re.I),
     'high', 'regulated_advice', 'medical'),
]

# Tier 2: confidence-gated — only override if ML confidence < 0.75
TIER2_RULES: list[tuple[re.Pattern, str, str, str | None]] = [
    (re.compile(r'\b(diagnos\w+|symptom|dosage|prescri\w+|tumor|malignant|mg of|medication|side effects?)\b', re.I),
     'high', 'regulated_advice', 'medical'),
    (re.compile(r'\b(statute|court|legal advice|attorney|lawyer|rights if)\b', re.I),
     'high', 'regulated_advice', 'legal'),
    (re.compile(r'\b(invest(ment)?s?\b|financial portfolio|stock portfolio|investment portfolio|tax (return|filing)|401k|loan approval|creditworthiness)\b', re.I),
     'high', 'regulated_advice', 'finance'),
    (re.compile(r'\b(ssn|social security number|passport number|aadhaar|pan card|medical record number)\b', re.I),
     'high', 'pii_sensitive', None),
    (re.compile(r'\b(prod(uction)? deploy)\b', re.I),
     'high', 'operational', None),
]


def apply_safety_overrides(prompt: str, parse: StructuredSemanticParse) -> StructuredSemanticParse:
    normalized = re.sub(r'\s+', ' ', prompt.lower().strip())
    # Tier 1: always active
    for pattern, min_tier, risk_type, domain in TIER1_RULES:
        if pattern.search(normalized):
            if RISK_ORDER.get(min_tier, 0) > RISK_ORDER.get(parse.risk_tier, 0):
                parse.risk_tier = min_tier
                parse.risk_type = risk_type
                parse.needs_verification = True
            if domain:
                parse.domain = domain
    # Tier 2: only override if parser confidence is below threshold
    if getattr(parse, 'parser_confidence', 1.0) < 0.75:
        for pattern, min_tier, risk_type, domain in TIER2_RULES:
            if pattern.search(normalized):
                if RISK_ORDER.get(min_tier, 0) > RISK_ORDER.get(parse.risk_tier, 0):
                    parse.risk_tier = min_tier
                    parse.risk_type = risk_type
                    parse.needs_verification = True
                if domain:
                    parse.domain = domain
    return parse



# ----------------------------- Parser ----------------------------------

class EmbeddingSemanticParser:
    def __init__(
        self,
        dataset_path: str | Path = "data/semantic_examples.json",
        model_name: str = "BAAI/bge-large-en-v1.5",
        cache_dir: str | Path = ".embed_cache",
        top_k: int = 8,
        temperature: float = 12.0,          # softmax sharpness on similarities
        low_confidence_threshold: float = 0.45,
    ):
        self.top_k = top_k
        self.temperature = temperature
        self.low_confidence_threshold = low_confidence_threshold
        self.model_name = model_name

        if not FASTEMBED_AVAILABLE:
            raise RuntimeError("fastembed is not installed.")

        self.model = TextEmbedding(model_name=model_name)

        dataset_path = Path(__file__).resolve().parents[2] / dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        raw = dataset_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        self.examples: list[dict] = data.get("examples", [])
        if not self.examples:
            raise ValueError("semantic_examples.json contains no examples")

        self.matrix, self.clf_family, self.clf_domain, self.clf_risk = self.load_or_build_cache(
            raw, Path(__file__).resolve().parents[2] / cache_dir
        )
        self.cross_encoder = None
        
    def get_cross_encoder(self):
        if self.cross_encoder is None:
            self.cross_encoder = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        return self.cross_encoder

    def get_safe_default(self) -> StructuredSemanticParse:
        return StructuredSemanticParse(
            primary_family="chat",
            domain="general",
            risk_tier="medium",
            risk_type="standard",
            expected_output="free_text",
            document_type="generic",
            ambiguity_score=1.0,
            decomposition_needed=False,
            needs_verification=True
        )

    # ---- startup: embed dataset once, train Logistic Regression, cache all
    def load_or_build_cache(self, raw_json: str, cache_dir: Path):
        cache_dir.mkdir(exist_ok=True)
        # Hash includes fastembed model name and dataset content for robust invalidation
        # We hash the parsed, sorted JSON to prevent line-ending (\n vs \r\n) or spacing mismatches
        hasher = hashlib.sha256()
        stable_json = json.dumps(self.examples, sort_keys=True)
        hasher.update(stable_json.encode())
        hasher.update(self.model_name.encode())
        digest = hasher.hexdigest()[:16]
        
        cache_file = cache_dir / f"examples_{digest}.npy"
        clf_fam_file = cache_dir / f"clf_fam_{digest}.joblib"
        clf_dom_file = cache_dir / f"clf_dom_{digest}.joblib"
        clf_risk_file = cache_dir / f"clf_risk_{digest}.joblib"
        
        if (
            cache_file.exists() and clf_fam_file.exists() and 
            clf_dom_file.exists() and clf_risk_file.exists()
        ):
            return (
                np.load(cache_file),
                joblib.load(clf_fam_file),
                joblib.load(clf_dom_file),
                joblib.load(clf_risk_file)
            )

        # Build Vectors
        texts = [ex["text"] for ex in self.examples]
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        vecs /= norms  # pre-normalize
        np.save(cache_file, vecs)
        
        # Train Classifiers
        clf_family = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf_family.fit(vecs, [ex["primary_family"] for ex in self.examples])
        joblib.dump(clf_family, clf_fam_file)
        
        clf_domain = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf_domain.fit(vecs, [ex["domain"] for ex in self.examples])
        joblib.dump(clf_domain, clf_dom_file)
        
        clf_risk = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf_risk.fit(vecs, [ex.get("risk_tier", "low") for ex in self.examples])
        joblib.dump(clf_risk, clf_risk_file)
        
        return vecs, clf_family, clf_domain, clf_risk
        
    def add_example(self, example: dict):
        """Dynamically add an example, save it, and hot-swap the internal matrix."""
        
        # Calculate embedding for the new example
        new_vec = np.array(next(iter(self.model.embed([example["text"]]))), dtype=np.float32)
        new_vec /= np.linalg.norm(new_vec)
        
        # Protect against dataset poisoning/fighting: check near-duplicates
        is_update = False
        update_idx = -1
        if self.matrix.shape[0] > 0:
            sims = self.matrix @ new_vec
            best_idx = np.argmax(sims)
            if sims[best_idx] > 0.98:
                # If a user submits the same prompt again (e.g. to fix a misclick), UPDATE the existing one instead of rejecting or duplicating
                is_update = True
                update_idx = best_idx
                
        dataset_path = Path(__file__).resolve().parents[2] / "data" / "semantic_examples.json"
        raw = dataset_path.read_text(encoding="utf-8")
        data = json.loads(raw)
                
        if is_update:
            self.examples[update_idx].update({
                "primary_family": example.get("primary_family"),
                "domain": example.get("domain"),
                "risk_tier": example.get("risk_tier"),
                "complexity": example.get("complexity", "low")
            })
            # Also update in the saved JSON array
            for i, ex in enumerate(data.setdefault("examples", [])):
                if ex.get("text") == self.examples[update_idx]["text"]:
                    data["examples"][i].update({
                        "primary_family": example.get("primary_family"),
                        "domain": example.get("domain"),
                        "risk_tier": example.get("risk_tier"),
                        "complexity": example.get("complexity", "low")
                    })
                    break
        else:
            # Append to memory
            self.examples.append(example)
            # Append to JSON
            data.setdefault("examples", []).append(example)
            
        # Save back to JSON
        new_raw = json.dumps(data, indent=2)
        dataset_path.write_text(new_raw, encoding="utf-8")
        
        # Fast incremental matrix update (avoids re-embedding 2850 texts which takes ~20 seconds)
        import hashlib
        cache_dir = Path(__file__).resolve().parents[2] / ".embed_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Fast incremental matrix update (avoids re-embedding 2850 texts which takes ~20 seconds)
        import hashlib
        cache_dir = Path(__file__).resolve().parents[2] / ".embed_cache"
        cache_dir.mkdir(exist_ok=True)
        
        hasher = hashlib.sha256()
        stable_json = json.dumps(self.examples, sort_keys=True)
        hasher.update(stable_json.encode())
        hasher.update(self.model_name.encode())
        digest = hasher.hexdigest()[:16]
        
        if not is_update:
            self.matrix = np.vstack([self.matrix, new_vec])
            
        np.save(cache_dir / f"examples_{digest}.npy", self.matrix)
        
        self.clf_family = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.clf_family.fit(self.matrix, [ex["primary_family"] for ex in self.examples])
        joblib.dump(self.clf_family, cache_dir / f"clf_fam_{digest}.joblib")
        
        self.clf_domain = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.clf_domain.fit(self.matrix, [ex["domain"] for ex in self.examples])
        joblib.dump(self.clf_domain, cache_dir / f"clf_dom_{digest}.joblib")
        
        self.clf_risk = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.clf_risk.fit(self.matrix, [ex.get("risk_tier", "low") for ex in self.examples])
        joblib.dump(self.clf_risk, cache_dir / f"clf_risk_{digest}.joblib")
        
        # Clear the lru_cache for the parser
        _cached_parse.cache_clear()

    # ---- hot path
    def parse(self, prompt: str | list) -> StructuredSemanticParse:
        try:
            # Handle multi-turn context
            if isinstance(prompt, list):
                # Extract system prompt if present
                system_prompt = next((m.get("content", "") for m in prompt if m.get("role") == "system"), "")
                # Extract last user message
                last_user = next((m.get("content", "") for m in reversed(prompt) if m.get("role") == "user"), "")
                # Merge for context
                prompt_text = f"{system_prompt}\n{last_user}".strip() if system_prompt else last_user
            else:
                prompt_text = prompt

            # Smart truncation: first 2000 chars to stay <10ms and capture intent without losing context
            truncated_prompt = prompt_text[:2000]
            
            # Embed symmetric
            q = np.array(next(iter(self.model.embed([truncated_prompt]))), dtype=np.float32)
            norm = np.linalg.norm(q)
            if norm > 0:
                q /= norm
                
            q_2d = np.expand_dims(q, axis=0)

            # --- Logistic Regression Predictions (Primary Fields) ---
            lr_family = str(self.clf_family.predict(q_2d)[0])
            lr_domain = str(self.clf_domain.predict(q_2d)[0])
            lr_risk = str(self.clf_risk.predict(q_2d)[0])
            
            probs_family = self.clf_family.predict_proba(q_2d)[0]
            top_2 = np.argsort(probs_family)[-2:]
            prob_top1 = probs_family[top_2[1]]
            prob_top2 = probs_family[top_2[0]] if len(probs_family) > 1 else 0.0
            margin = float(prob_top1 - prob_top2)
            ambiguity = float(max(0.0, 1.0 - margin))
            
            # --- KNN Logic (Secondary Fields & Explainability) ---
            sims = self.matrix @ q                     # cosine, since all normalized
            if len(sims) == 0:
                return apply_safety_overrides(prompt_text, self.get_safe_default())
            k = min(self.top_k, len(sims))
            top_idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            top_sims = sims[top_idx]

            # temperature-softmax weights over top-k similarities
            w = np.exp(self.temperature * (top_sims - top_sims.max()))
            w /= w.sum()

            neighbors = [self.examples[i] for i in top_idx]
            
            # Cross-Encoder Reranking for high ambiguity cases
            if ambiguity > 0.6 and FASTEMBED_AVAILABLE:
                try:
                    reranker = self.get_cross_encoder()
                    neighbor_texts = [ex["text"] for ex in neighbors]
                    rerank_scores = list(reranker.rerank(truncated_prompt, neighbor_texts))
                    # Rerank returns a list of floats corresponding to input documents
                    sorted_indices = np.argsort(rerank_scores)[::-1]
                    neighbors = [neighbors[i] for i in sorted_indices]
                    w = w[np.array(sorted_indices)]  # reorder weights to match new neighbor order
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Cross-encoder reranking failed: {e}")

            # KNN-LR alignment: only use neighbors matching LR primary family for secondary field voting
            aligned = [(ex, wt) for ex, wt in zip(neighbors, w) if ex.get('primary_family') == lr_family]
            if len(aligned) >= 3:
                aligned_neighbors, aligned_w_list = zip(*aligned)
                aligned_w = np.array(aligned_w_list, dtype=float)
                aligned_w /= aligned_w.sum()
            else:
                aligned_neighbors, aligned_w = neighbors, w

            # per-field weighted vote for secondary fields
            fields: dict[str, str] = {}
            for field in CATEGORICAL_FIELDS:
                votes: dict[str, float] = defaultdict(float)
                for ex, weight in zip(aligned_neighbors, aligned_w):
                    votes[str(ex.get(field, "unknown"))] += float(weight)
                winner = max(votes, key=votes.get)
                fields[field] = str(winner)

            bools: dict[str, bool] = {}
            for field in BOOLEAN_FIELDS:
                score = sum(wt for ex, wt in zip(aligned_neighbors, aligned_w) if ex.get(field))
                bools[field] = bool(score > 0.5)

            top1 = float(top_sims[0])

            # Multi-factor ambiguity using the full formula (wires previously dead methods)
            votes_family: dict[str, float] = defaultdict(float)
            votes_domain: dict[str, float] = defaultdict(float)
            for ex, wt in zip(neighbors, w):
                votes_family[str(ex.get('primary_family', 'unknown'))] += float(wt)
                votes_domain[str(ex.get('domain', 'unknown'))] += float(wt)
            fam_entropy = self.norm_entropy(list(votes_family.values()))
            dom_entropy = self.norm_entropy(list(votes_domain.values()))
            top1_sim = float(top_sims[0]) if len(top_sims) > 0 else 0.0
            ambiguity = self.ambiguity(margin, top1_sim, fam_entropy, dom_entropy)

            parse = StructuredSemanticParse(
                primary_family=lr_family,
                domain=lr_domain,
                risk_tier=lr_risk,
                risk_type=fields["risk_type"],
                expected_output=fields["expected_output"],
                document_type=fields["document_type"],
                ambiguity_score=round(ambiguity, 3),
                decomposition_needed=bools["decomposition_needed"],
                needs_verification=bools["needs_verification"],
                parser_confidence=round(float(prob_top1), 3),
                complexity=fields.get("complexity")
            )

            # Statistical escalation: if any top-K neighbor is extreme/high risk, escalate.
            high_risk_found = False
            for ex in neighbors:
                if RISK_ORDER.get(ex.get("risk_tier", "low"), 0) >= RISK_ORDER["high"]:
                    high_risk_found = True
                    break
                    
            if high_risk_found and parse.ambiguity_score > 0.6:
                if RISK_ORDER.get(parse.risk_tier, 0) < RISK_ORDER["high"]:
                    parse.risk_tier = "high"
                    parse.needs_verification = True

            # out-of-distribution guard: nothing in the dataset is close
            if top1 < self.low_confidence_threshold:
                parse.primary_family = "chat"
                parse.domain = "general"
                parse.expected_output = "free_text"
                parse.document_type = "generic"
                parse.ambiguity_score = max(parse.ambiguity_score, 0.85)
                # uncertain != safe: default risk UP, not down
                if RISK_ORDER.get(parse.risk_tier, 0) < RISK_ORDER["medium"]:
                    parse.risk_tier = "medium"
                parse.needs_verification = True

            return apply_safety_overrides(prompt_text, parse)
        except Exception as e:
            # Fail conservative
            import logging
            logging.getLogger(__name__).error(f"Embedding parser failed: {e}")
            safe_text = prompt if isinstance(prompt, str) else ""
            return apply_safety_overrides(safe_text, self.get_safe_default())

    # ---- helpers
    @staticmethod
    def norm_entropy(weights: list[float]) -> float:
        if len(weights) <= 1:
            return 0.0
        total = sum(weights)
        probs = [x / total for x in weights if x > 0]
        h = -sum(p * math.log(p) for p in probs)
        return h / math.log(len(probs)) if len(probs) > 1 else 0.0

    def ambiguity(self, margin: float, top1: float, fam_entropy: float, dom_entropy: float) -> float:
        distance_term = max(0.0, 1.0 - top1)          # far from everything known
        disagreement = 0.6 * fam_entropy + 0.4 * dom_entropy  # neighbors disagree
        margin_penalty = max(0.0, 0.2 - margin) # Penalize if the margin is very small
        
        score = 0.4 * distance_term + 0.4 * disagreement + 0.2 * margin_penalty
        return float(min(1.0, max(0.0, score)))


# ------------------------- Module-level API ----------------------------

_parser: EmbeddingSemanticParser | None = None

def get_parser() -> EmbeddingSemanticParser:
    global _parser
    if _parser is None:
        _parser = EmbeddingSemanticParser()
    return _parser

# LRU Cache keyed by prompt string
@lru_cache(maxsize=1000)
def _cached_parse(prompt: str) -> StructuredSemanticParse:
    return get_parser().parse(prompt)

def parse_prompt_to_semantic_struct(prompt: str) -> StructuredSemanticParse:
    import copy
    return copy.deepcopy(_cached_parse(prompt))
