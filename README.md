# Neural Gateway

> **🌐 [Try the Live Demo →](https://mantavyamaan.github.io/Neural-Gateway/demo)**

It is a complete, end-to-end AI agent and neural routing gateway, exposed as a FastAPI service. Neural Gateway decides *which model or execution plan* should handle a request, taking into account task requirements, governance policy, runtime health, cost, latency, uncertainty, and SLAs. It then acts as an intelligent proxy to actively generate and stream the final response directly back to the user.


## How it's organized

```
neural_gateway/
├── app/
│   ├── main.py                 # FastAPI app entrypoint (uvicorn target)
│   ├── config.py                # version stamps + confidence thresholds
│   ├── models/
│   │   ├── schemas.py            # internal dataclasses (TaskFeatures, ExecutionPlan, ...)
│   │   ├── catalog.py             # provider -> model name catalog
│   │   ├── registry_builder.py     # fallback heuristics for unmapped models
│   │   └── real_benchmarks.json    # ground-truth benchmark overrides
│   ├── core/
│   │   ├── database.py              # SQLite database and persistence layer
│   │   ├── openrouter_sync.py       # fetches live models and pricing from OpenRouter API
│   │   ├── artifact_inspection.py   # PDF/image/audio/video/xlsx/pptx inspection
│   │   ├── semantic_parser.py        # deterministic + heuristic task parsing
│   │   ├── feasibility.py             # hard constraint filtering
│   │   ├── policy.py                   # governance / policy engine
│   │   ├── scoring.py                   # Bayesian quality, Pareto, utility, confidence
│   │   ├── planning.py                   # single-model & multi-stage plan generation
│   │   ├── router.py                      # route() — the main pipeline
│   │   └── formatting.py                   # human-readable decision summaries
│   └── api/
│       ├── schemas.py                       # pydantic request/response models
│       └── routes.py                         # FastAPI endpoints
├── tests/
│   └── test_router.py                         # end-to-end scenario tests
├── requirements.txt
├── .env.example
└── README.md
```

This mirrors the routing pipeline described in the Neural Gateway design doc:

```
Request -> Artifact Inspection -> Semantic Parsing -> Task Representation
   -> Feasibility Filtering -> Policy Enforcement -> Bayesian Quality
   -> Pareto Reduction -> Utility Scoring -> Confidence Estimation
   -> Execution Plan -> Route / Escalate / Abstain
```

## Setup (VS Code / local)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional artifact-inspection libraries (`pymupdf`, `Pillow`, `openpyxl`,
`python-pptx`, `mutagen`) are in `requirements.txt` but the router degrades
gracefully if any are missing — it just falls back to prompt-keyword
heuristics for that modality. `ffprobe` (from ffmpeg) is used for
audio/video duration if present on the host `PATH`.

## Prerequisites (ONNX & Vector Embeddings)

Neural Gateway uses a high-precision K-Nearest Neighbors (KNN) Vector Embedding Engine powered by **BAAI/bge-large-en-v1.5** to semantically parse and categorize prompts before they are routed.

Because it uses `fastembed` (ONNX Runtime), it runs **entirely on the CPU** in under 70 milliseconds and requires zero GPU VRAM. It operates 100% locally and offline without any need for an Ollama server or PyTorch.

1. **Install Python 3.10+**
2. The embedding model is automatically downloaded and cached by `fastembed` on the very first startup. No manual downloading is required.

## Run the Service & UI

Neural Gateway now features a premium **Streamlit Frontend** with Glassmorphism design, Custom Model Allowlisting, and Stage 2 LLM execution (directly streaming responses from OpenRouter using the dynamically selected optimal model).

You need two terminals to run the full stack:

**Terminal 1 (Backend API):**
```bash
uvicorn app.main:app --reload --port 8000 // uvicorn app.main:app --reload --port 8080 if already have something running in that port
```
*(Interactive Swagger docs available at **http://127.0.0.1:8000/docs**)*

**Terminal 2 (Frontend Dashboard):**
```bash
streamlit run frontend.py   // streamlit run frontend.py --server.port 8505   if already have something running in that port
```

## Example request

```bash
curl -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Implement a concurrent web crawler in Python with rate limiting and structured JSON output.",
    "input_formats": ["text"],
    "estimated_tokens": 3000,
    "estimated_output_tokens": 4000,
    "profile_name": "balanced"
  }'
```

Response shape (abridged):

```json
{
  "abstain": false,
  "escalate_to_human": false,
  "selected_plan": {
    "plan_id": "...",
    "plan_type": "single_model",
    "selected_model": "GPT-5.4",
    "fallback_models": ["Claude-Sonnet-4.6", "Gemini-3-Pro"],
    "verifier_models": [],
    "expected_latency_ms": 2150.4,
    "expected_cost_usd": 0.0182,
    "confidence": 0.61
  },
  "decision_record": { "...": "full auditable trace" },
  "summary_text": "=== Neural Gateway Router Decision === ..."
}
```

## Endpoints

| Method | Path             | Purpose |
|--------|------------------|---------|
| POST   | `/route`         | Run the full routing pipeline for a request (Control Plane) |
| POST   | `/execute`       | Run the pipeline, execute the LLM via OpenRouter, and verify it (Data Plane) |
| POST   | `/train_parser`  | Dynamically train the vector embedding matrix with a correction |
| POST   | `/outcome`       | Feed an observed outcome back into a model's Task-Conditional Bayesian priors |
| GET    | `/models`        | List the canonical registry (summary view) |
| GET    | `/models/{name}` | Full registry entry for one model |
| POST   | `/models`        | Dynamically add or update a model in the SQLite registry |
| DELETE | `/models/{name}` | Remove a model from the registry |
| GET    | `/versions`      | Current version stamps for every subsystem |
| GET    | `/health`        | Liveness probe |

## The Model Registry (Dynamic SQLite + OpenRouter)

Neural Gateway uses a real-time, dynamic **SQLite database** to store its model registry, making it a production-ready routing engine.

When the application starts, `app/core/openrouter_sync.py` connects to the **OpenRouter API** to download the latest available models, their exact context window limits, and live pricing.

To ensure routing decisions are mathematically precise, it cross-references these models against `app/models/real_benchmarks.json`, which contains manually curated, **ground-truth benchmark scores** (like SWE-Bench and MMLU equivalents) for flagship models like GPT-4o, Claude 3.5 Sonnet, and Llama 3.

For obscure community models that aren't mapped in our benchmark JSON, it explicitly defaults them to an "insufficient evidence" state. By default, `NEURAL_GATEWAY_REQUIRE_MEASURED_EVIDENCE=true` prevents these unknown models from receiving auto-routed traffic unless the tenant overrides the behavior. Tests are also run deterministically to ensure the suite is blazingly fast and works entirely offline.

## The Semantic Vector Parsing Engine

By default, task understanding in Neural Gateway is handled by the **EmbeddingSemanticParser** in `app/core/embedding_parser.py`. It mathematically categorizes requests by computing the Dense Vector against three highly-trained **Logistic Regression Classifiers** and a local **Cross-Encoder** reranker.

* **Dynamic Training & CI Gate:** You can instantly make the Gateway smarter using the **Train the Engine** UI. When you submit a correction, the backend API (`/train_parser`) encodes your exact prompt into a new vector. Before saving it permanently, it runs a strict local Evaluation Suite. If the new vector degrades the matrix below 90% accuracy, it blocks the change and reverts the RAM matrix, ensuring poison-proof learning.
* **Deterministic Safety Override Layer:** The system automatically escalates risky keywords (like "suicide" or "drop table") to High/Extreme risk tiers, strictly enforcing safety regardless of statistical voting.

## Testing

```bashdd
pytest tests/ -v
```

## Production requirements

Neural Gateway is a routing control plane, not a model executor. A production
deployment must invoke the selected plan in a separate execution service,
verify outputs, and submit authenticated outcomes. Name-derived fallback
scores are development fixtures only: with the default
`NEURAL_GATEWAY_REQUIRE_MEASURED_EVIDENCE=true`, models without curated benchmark or
measured production evidence are rejected before scoring. Populate the
registry with versioned capability probes, task-family evaluations, and real
provider telemetry before enabling automatic routing.

Set `NEURAL_GATEWAY_ADMIN_API_KEY` before enabling the mutable model and outcome
endpoints. 

**Security & File Access Defaults:**
By default, `ALLOW_SERVER_FILE_PATHS` in `app/config.py` is disabled to prevent Path Traversal attacks. When testing local files or artifacts, you MUST upload them via the frontend UI rather than supplying local paths to the `/execute` API, as the API drops all absolute host paths for security. Use authenticated uploads or object-store references instead for production deployments.

Tests mirror the original design-doc demonstration scenarios: a coding
task, a high-risk legal contract review, audio summarization, an
offensive-security policy denial, a budget-constrained support task,
long-context research with citations, and file-driven conflict detection.

## Design principles (from the spec)

1. **Hard constraints override soft preferences** — feasibility filtering
   runs before any scoring.
2. **Policy is independent of scoring** — governance never hides inside
   utility weights.
3. **Runtime conditions matter** — latency, availability, queue pressure,
   and incident status all feed into routing, not just benchmarks.
4. **Uncertainty is quantified** — Thompson Sampling estimates confidence
   rather than returning a bare point estimate.
5. **Derive signals, don't duplicate them** — everything traces back to
   the canonical registry.
6. **Every decision is reproducible** — each `RoutingDecision` carries a
   reproducibility hash plus per-subsystem version stamps.
7. **Multi-stage workflows get structured planning** — OCR, document QA,
   summarization, etc. can be routed to different specialist models within
   one plan.
8. **Mathematical Purity** — Cost/Latency are strictly enforced as absolute filters, ranking uses purely absolute scale utilities (no min-maxing), Bayesian priors never double count observations, and confidence scores explicitly distinguish between Thompson Sampling win-rates and actual predicted task success.

