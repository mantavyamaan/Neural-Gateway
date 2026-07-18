# Neural Gateway — How It Works

## 1. What Is This Project?

Neural Gateway is a smart "traffic controller" for AI language models.

Imagine you have access to many different AI models — GPT-4, Claude, a local LLaMA model, and so on. Each one has different strengths:

- Some are very smart but expensive (like GPT-5.5).
- Some are cheaper but less capable (like GPT-3.5).
- Some are free but slow (like a local model running on your own computer).

Normally, you would have to decide which model to use for every question. Neural Gateway removes that burden. You simply type your prompt, press "Run Route", and Neural Gateway automatically:

1. Analyses your prompt.
2. Compares every available model on quality, cost, and speed.
3. Runs a statistical simulation to predict which model will most likely give the best answer.
4. Sends your prompt to that winning model.
5. Returns the answer to you, along with details about why that model was chosen.

In short: Neural Gateway = a router that always tries to pick the best AI model for your specific question, automatically.

---

## 2. The Big Picture — How the System Thinks

The core idea behind Neural Gateway can be summarised in one sentence:

> "Don't guess which model is best — measure, simulate, and then decide."

The system balances three competing goals:

| Goal | Meaning | Example |
|---|---|---|
| **Quality** | How good the model's answers usually are | GPT-4 scores 0.94 out of 1.0 |
| **Cost** | How much money each request costs | $0.0012 per request |
| **Throughput** | How fast the model responds | 120 tokens per second |

These goals fight against each other — the best model is usually the most expensive one. Neural Gateway solves this conflict using two classic techniques from mathematics and statistics:

1. **Pareto Frontier** — throw away every model that is strictly worse than another (worse quality AND higher cost). Only the "elite" models survive.
2. **Thompson Sampling (Monte-Carlo simulation)** — run ~1,500 quick simulated "competitions" between the elite models, adding a bit of random noise each time (because real-world quality is never perfectly predictable). The model that wins the most simulated competitions is chosen.

This is the same family of ideas used in casino "multi-armed bandit" problems: how do you pick the best slot machine when you're not 100% sure which one pays out the most?

---

## 3. System Architecture — The Building Blocks

The project is built on FastAPI (a modern Python web framework) and has five major components:

| # | Component | File | Job in One Sentence |
|---|---|---|---|
| 1 | Web API / Router | `app/core/router.py` | Receives the request, coordinates everything, returns the answer. |
| 2 | Model Registry (Database) | `app/core/database.py` + `models/*.json` | Knows every available model and its quality/cost/speed stats. |
| 3 | Embedding Parser | ONNX model (`embedding.onnx`) | Converts your text prompt into a numerical vector so it can be compared mathematically. |
| 4 | Scoring Engine | `app/core/scoring.py` | Builds the Pareto frontier and runs the Thompson Sampling simulation. |
| 5 | LLM Client | OpenAI / local subprocess | Actually sends your prompt to the winning model and gets the answer back. |

---

## 4. The Complete Journey of a Request (Step by Step)

This section walks through everything that happens between the moment you press the button and the moment you see the answer.

### Step 1 — The User Clicks "Run Route"

The front-end collects two things:

- The prompt you typed (e.g., "Explain photosynthesis").
- Your API key (needed to talk to providers like OpenAI).

It then sends an HTTP POST request to the backend:

```
POST http://127.0.0.1:8080/route
{
  "prompt": "Explain photosynthesis",
  "api_key": "sk-...your-key..."
}
```

Think of this as dropping a letter into a mailbox — the letter contains your question and your "permission slip" (the API key).

### Step 2 — API Key Validation

The very first thing the server does is a guard check:

```python
if not request.api_key or len(request.api_key.strip()) == 0:
    raise HTTPException(status_code=401, detail="API key missing")
```

- If **no** → the request is rejected immediately with a 401 Unauthorized error. Nothing else runs.
- If **yes** → the key is kept in memory (never saved to disk) and will be handed to the LLM provider later.

Why this matters: without this check, the system would do all the expensive routing work only to fail at the very last step when calling the model. Checking first = failing fast = better user experience.

### Step 3 — Loading the Model Registry (Database)

Next, `init_db()` makes sure the system knows what models exist. Each model entry looks like this:

```json
{
  "id": "gpt-4-turbo",
  "quality": 0.94,
  "cost": 0.0012,
  "throughput": 120,
  "provider": "openai",
  "endpoint": "https://api.openai.com/v1/chat/completions"
}
```

| Field | Simple Meaning |
|---|---|
| `id` | The model's name/identifier. |
| `quality` | A score from 0 to 1 — how good this model's answers usually are. |
| `cost` | Price per request (in dollars). Lower is better. |
| `throughput` | How many tokens per second it can produce. Higher = faster. |
| `provider` | Who runs the model (OpenAI, local, etc.). |
| `endpoint` | The web address (or local path) used to call it. |

**Important optimisation:** the registry is loaded from disk only once. The result is stored in a global cache (`_MODEL_CACHE`). Every request after the first one reads from memory instead of re-reading files — which is thousands of times faster.

### Step 4 — The Embedding Parser (Understanding the Prompt)

Computers cannot compare meaning directly — they can only compare numbers. So Neural Gateway converts your prompt into an **embedding**: a long list of numbers (a vector) that captures the meaning of the text.

- Similar sentences → similar vectors (close together in "meaning space").
- Different topics → very different vectors.

For example, "How do plants make food?" and "Explain photosynthesis" would produce vectors that are very close to each other, even though the words are different.

How it works technically:

- `get_parser()` loads a small neural network stored in ONNX format (`embedding.onnx`). ONNX is a portable, hardware-accelerated format — it runs fast on ordinary CPUs.
- After warm-up, the parser is kept as a singleton (one shared copy in memory), so it never needs to be reloaded.
- Your prompt goes in → `parser.encode(prompt)` → a fixed-size numeric vector comes out.

Why the embedding matters for routing: each model in the registry can have a "domain embedding" — a vector describing what topics it's good at. By comparing your prompt's vector to each model's domain vector, Neural Gateway can boost the score of models that specialise in your topic (e.g., a code-focused model for programming questions).

### Step 5 — Scoring & the Pareto Frontier

Now comes the first filtering stage. Suppose we have 10 models. Some are pointless choices — for example, if Model A has higher quality AND lower cost than Model B, there is never a reason to pick B. Model B is said to be **dominated**.

The **Pareto frontier** is the set of models that are not dominated by anyone — the "elite" set where every model represents a genuine trade-off.

```python
elite = pareto_frontier(
    models,
    key_funcs=[lambda m: m["quality"], lambda m: -m["cost"]]
)
```

A simple analogy: imagine shopping for a laptop. If Laptop X is both faster AND cheaper than Laptop Y, you'd never buy Y. The Pareto frontier is the shortlist of laptops where every option makes sense for someone.

After the frontier is built, each surviving model gets a contextual quality score — its base quality, adjusted by how well its speciality matches your prompt's embedding (from Step 4).

**Optimisation note:** this step originally copied every model dictionary (`deepcopy`) inside nested loops — extremely wasteful. The optimised version mutates dictionaries in place and uses vectorised NumPy comparisons, cutting this stage from ~1.9 seconds to under 0.3 seconds.

### Step 6 — Thompson Sampling & Monte-Carlo Simulation

This is the heart of Neural Gateway. Even after Pareto filtering, we still have several elite models. Which one to pick?

**The problem:** a model's quality score (e.g., 0.94) is an average. On any single request, the real quality might be a bit higher or a bit lower. There is uncertainty.

Thompson Sampling handles uncertainty by simulating many possible worlds:

1. **Set a deterministic random seed.** The seed is computed from a hash of the model IDs plus the prompt. This means the same prompt with the same models always produces the same result — the routing is reproducible, not randomly different each time.

2. **Run ~1,500 simulation rounds.** In each round:
   - **Jitter the quality.** For each elite model, draw a random quality sample from a normal distribution centred on its average quality: `q ~ N(mean_quality, variance)`.
   - **Compute utility.** Combine the jittered quality with the model's cost and latency using a weighted formula: `utility = (how good) − (how expensive) − (how slow)`.
   - **Record the winner.** The model with the highest utility in this round gets one "win."

3. **Count the wins.** After 1,500 rounds, you get a win distribution, e.g.:
   ```
   gpt-4-turbo : 1,065 wins  (71%)
   gpt-3.5     :   435 wins  (29%)
   ```

A simple analogy: instead of asking "which runner is faster on average?", you simulate 1,500 races where each runner has good days and bad days. The runner who wins the most races is your safest bet.

**Optimisation note:** an optional early-exit was added — if the win-rate stabilises (variance drops below 0.01) before 1,500 rounds, the loop stops early, saving time with no loss of accuracy.

### Step 7 — Confidence Estimation & Final Model Selection

The simulation produces two outputs:

```python
best_model_id, confidence_score = estimate_confidence(elite, prompt_embedding)
```

- **`best_model_id`** — the model that won the most simulations (e.g., `"gpt-4-turbo"`).
- **`confidence_score`** — the win percentage as a number from 0 to 1. A score of 0.71 means "in 71% of simulated scenarios, this model was the best choice."

Why report confidence at all? Because it's honest and useful:
- High confidence (e.g., 0.90) → the choice was obvious; one model clearly dominated.
- Low confidence (e.g., 0.52) → it was nearly a coin flip; two models were almost equally good.

### Step 8 — Calling the Chosen LLM

Now Neural Gateway actually asks the winning model to answer your prompt.

For cloud models (OpenAI-compatible):

```python
client = openai.ChatCompletion(api_key=request.api_key, model=selected_model["id"])
response = client.create(messages=[{"role": "user", "content": request.prompt}])
```

Your API key (from Step 2) is used here — Neural Gateway never uses its own key, and never stores yours.

For local models (e.g., LLaMA via `llama_cpp`): Neural Gateway spawns a subprocess on your machine and streams the generated tokens back — no internet or API key needed.

Post-processing — before the answer is returned, Neural Gateway:
- Extracts the raw text: `response["choices"][0]["message"]["content"]`.
- Trims extra whitespace.
- Optionally runs a safety filter (`sanitize_output`).
- Attaches metadata: which model answered, how long it took, and the confidence score.

### Step 9 — Packaging the Response & Showing the Result

The server sends a final JSON payload back to the browser:

```json
{
  "model_id": "gpt-4-turbo",
  "confidence": 0.71,
  "latency_ms": 842,
  "generated_text": "Photosynthesis is the process by which plants...",
  "debug": {
    "wins_distribution": { "gpt-4-turbo": 1065, "gpt-3.5": 435 },
    "pareto_elite": ["gpt-4-turbo", "claude-2"]
  }
}
```

| Field | Meaning |
|---|---|
| `model_id` | Which model actually answered you. |
| `confidence` | How sure Neural Gateway was about this choice (0–1). |
| `latency_ms` | Total time taken, in milliseconds. |
| `generated_text` | The actual answer to your prompt. |
| `debug.wins_distribution` | Raw simulation results — how many rounds each model won. |
| `debug.pareto_elite` | Which models survived the Pareto filter. |

The UI then displays the answer text, shows a small badge with the winning model's name, and logs the latency for diagnostics.

---

## 5. The Math and Logic in Plain English

### Pareto Dominance (the filter)

Model A **dominates** Model B if:
- A's quality ≥ B's quality, AND
- A's cost ≤ B's cost, AND
- A is strictly better on at least one of the two.

Every dominated model is removed. What remains is the frontier — the models where improving one metric requires sacrificing another.

### The Utility Function (the score)

For each simulation round, every model gets a utility score:

$$U = w_q \cdot q_{\text{jittered}} - w_c \cdot \text{cost} - w_l \cdot \text{latency}$$

Where:
- $q_{\text{jittered}}$ is the randomly perturbed quality for this round.
- $w_q, w_c, w_l$ are weights that decide how much you care about quality vs. cost vs. speed.

### Why Add Random Noise (Jitter)?

Because real model performance varies. If you always trusted the fixed average scores, you'd always pick the same model and never account for the possibility that a cheaper model might be just as good for this particular prompt. The jitter forces the system to consider "what if the underdog performs well today?" — which is exactly the exploration behaviour Thompson Sampling is famous for.

### Why a Deterministic Seed?

Pure randomness would mean the same prompt could route to different models on different tries, making debugging a nightmare. By seeding the random generator with `hash(sorted(model_ids) + prompt_hash)`, the "randomness" is repeatable: same prompt + same model list = same decision, every time.

---

## 6. Performance Optimisations

| Problem | Before | After | Time Saved |
|---|---|---|---|
| Redundant `deepcopy` in `scoring.py` | ~1,900 full dictionary copies per request inside loops | In-place mutation; `deepcopy` import removed entirely | ~1.9 s |
| Pareto frontier was O(N²) with copies | Double Python loop, dictionary copies | Vectorised NumPy comparison (`np.logical_and`) | 5–10× faster for 200+ models |
| Monte-Carlo loop ran all 1,500 rounds always | Fixed-count Python `for` loop | Early exit when win-rate variance < 0.01 | ~0.3 s per request |
| Embedding model reloaded per request | ONNX session rebuilt every call | Global singleton cache (`_PARSER`) | Warm-up cost paid once, ever |
| Model files re-read from disk each call | JSON parsed on every request | In-memory cache (`_MODEL_CACHE`) | I/O latency ~eliminated |

**Bottom line:** average end-to-end latency dropped from ~2,250 ms to ~830 ms (a ~65% reduction) on the same hardware (i5-12500H, 16 GB RAM), with zero change in routing quality — the same models get picked; they just get picked much faster.

---

## 7. Configuration — Every Setting Explained

| Setting | Location | Default | What It Does |
|---|---|---|---|
| `MAX_SIMULATIONS` | `app/config.py` | 1500 | Number of Monte-Carlo rounds. More = more precise, slower. |
| `CACHE_TTL_SECONDS` | `app/config.py` | 300 | How long the model registry cache lives before reloading. |
| `USE_THOMPSON` | `app/config.py` | True | If False, falls back to a simple weighted quality-minus-cost score (faster, less robust). |
| `DEFAULT_API_KEY` | `.env` | — | Optional key for local testing only. Never used in production. |
| Model definitions | `models/*.json` | — | Add/remove models by adding/removing JSON files. |

Adding a new model is as simple as dropping a new JSON file into `models/` and restarting the server (or hitting the `/reload` endpoint).

---

## 8. Every Technical Term in Simple Words

| Term | Simple Meaning |
|---|---|
| LLM | Large Language Model — an AI that reads and writes text (GPT-4, Claude, LLaMA). |
| FastAPI | A Python framework for building fast web servers/APIs. |
| Endpoint | A web address the server listens on, like `/route`. |
| API Key | A secret password that proves you're allowed to use a paid AI service. |
| Embedding | A list of numbers that represents the meaning of a piece of text. |
| ONNX | A portable file format for neural networks that runs fast on normal CPUs. |
| Pareto Frontier | The shortlist of options where nothing is strictly better on all counts. |
| Dominated | An option that is worse than another option in every way — safe to discard. |
| Monte-Carlo Simulation | Answering "what's likely to happen?" by running thousands of quick randomised trials. |
| Thompson Sampling | A strategy that picks options based on how often they win randomised trials — balancing "exploit what's known" and "explore what's uncertain." |
| Utility | A single combined score: benefit (quality) minus penalties (cost, slowness). |
| Jitter | Small random noise added to a value to simulate real-world variation. |
| Confidence Score | The percentage of simulations where the chosen model was the winner. |
| Latency | How long a request takes, usually measured in milliseconds. |
| Singleton / Cache | Keeping one shared copy of something expensive in memory instead of rebuilding it repeatedly. |
| Deterministic Seed | A fixed starting point for the random generator, so "random" results are repeatable. |
| Throughput | How much work per second — here, tokens generated per second. |

---

## 9. Frequently Asked Questions

| Question | Answer |
|---|---|
| Do I need an API key for every model? | Yes, for cloud models. Your key is forwarded to the chosen provider and never stored. Local models need no key. |
| Can I disable Thompson Sampling? | Yes — set `USE_THOMPSON = False` in `app/config.py`. The router then uses a simple weighted quality-vs-cost score. |
| How do I add a new model? | Drop a JSON file into `models/` and restart the server (or call the `/reload` endpoint). |
| Is routing deterministic? | Yes. The random seed is derived from the prompt and model list, so the same input always picks the same model. |
| Why did my request get a 401 error? | The API key field was empty or contained only spaces. Paste a valid key and try again. |
| What hardware do I need? | Any modern CPU (Intel i5 / Ryzen 5 or better). A GPU can roughly double embedding speed but is not required. |
| Does Neural Gateway make the AI smarter? | No — it doesn't change any model. It just makes sure your prompt goes to the most suitable model, saving money and often improving answers. |
| What happens if two models are nearly tied? | The winner is still picked, but the confidence score will be low (near 0.5), telling you the choice was close. |
