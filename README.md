# RAG vs GraphRAG: An Empirical Benchmark

An honest, reproducible comparison of **Standard Vector RAG**, **GraphRAG**, and **Hybrid RAG**
on the same corpus, the same queries, and the same models - built to answer one question:

> **When does GraphRAG actually earn its extra cost and latency, and when is Vector RAG simply better?**

Most public comparisons are vendor demos. This repository measures the trade-off directly:
index build time, memory, and RAGAS quality scores across a chunk-size sweep.

---

## READ THIS FIRST

> ### [`research/problems_readme.md`](research/problems_readme.md)
>
> Before debugging anything, changing any dependency, or interpreting any number in this repo,
> read the problems ledger. It contains:
>
> - [Section A](research/problems_readme.md#section-a-solved-problem-archive) - **35 already-solved problems** (the RAGAS 0.4.x + LlamaIndex war chest). If you hit an error, check here before investigating.
> - [Section B](research/problems_readme.md#section-b-open-problems) - **22 open problems** with severity, file, line, and impact.
> - [Section C](research/problems_readme.md#section-c-known-good-patterns-and-invalidated-approaches) - patterns to keep, and approaches that are **permanently invalidated**.
> - [Section D](research/problems_readme.md#section-d-half-applied-provider-migration) - the half-finished provider migration currently breaking the benchmark.
> - [Section E](research/problems_readme.md#section-e-rule-compliance-scorecard) - compliance scorecard against the project standards.
> - [Section F](research/problems_readme.md#section-f-measurement-record) - every number this project has ever produced, with caveats.
>
> The two documents in this repository that you should ever need are **this README** and
> **that ledger**. They exist so nobody has to re-read `.agent/memory/` or re-audit `src/` again.

---

## Current Status: NOT PRODUCING VALID RESULTS

Be direct about this before reading any chart.

| | |
|---|---|
| Vector RAG pipeline | Working (build, persist, cache, query) |
| GraphRAG pipeline | Working, but ~250-500x slower to index |
| Hybrid RAG pipeline | **Written but never executed** by the evaluator |
| RAGAS quality scores | **Currently return `null`** - the judge points at a dead provider |
| `data/results/benchmark_metrics.json` | **0 bytes** |
| Plots in `data/results/plots/` | **Stale copies** of the first run |
| Token / cost accounting | **Not implemented** - constants imported but unused |
| Query latency | Measured, then discarded before persistence |

The single blocking cause is [B1](research/problems_readme.md#b1---ragas-judge-still-points-at-the-old-provider-key-and-model):
`src/evaluator.py` still hardcodes the old API base URL, the old env var, and the old model,
while `src/config.py` has moved to a new provider. The resulting `AuthenticationError` is
swallowed by a bare `except`, so every score becomes `null` instead of failing loudly.

---

## Architecture

```
data/supply_chain_text.txt  (one shared corpus - data parity is mandatory)
            |
            +--> SentenceSplitter(chunk_size, chunk_overlap=50)   [identical for both]
            |
    +-------+--------------------------+--------------------------+
    |                                  |                          |
 VectorStoreIndex               KnowledgeGraphIndex         HybridRAG
 (embeddings, top-k)            (LLM triplet extraction,    (merge both, currently
    |                            SimpleGraphStore)           string-concatenates)
    |                                  |                          |
    +----------------> same 8 queries, same LLM <-----------------+
                              |
                    RAGAS: faithfulness, answer_relevancy, context_precision
                              |
              data/results/benchmark_metrics.{json,csv} -> plots
```

Data parity is enforced structurally: both pipelines read the same file and share a single
global LlamaIndex `Settings` object for LLM, embeddings, chunk size, and overlap.

---

## The Corpus

**Source**: Olist Brazilian e-commerce dataset
(`miminmoons/olist-ecommerce-for-delivery-and-review-prediction` via `kagglehub`).

**Why this dataset**: it has a genuine relational "butterfly effect" - a seller's state
affects shipping time, which affects delivery delay, which affects the review score. That is
exactly the multi-hop structure where GraphRAG should theoretically win. Alternatives
considered and rejected: the Enron email corpus (too unstructured) and Game of Thrones lore
(no verifiable ground truth).

**Preparation** (`src/dataset_prep.py`): 100 rows are flattened into plain-English paragraphs,
one per order, of the form:

```
order_id is e481f5... seller_state is SP customer_state is SP review_score is 4
price is 29.99 seller_grade is Mediocre time_to_ship_hours is 56.97 ...
```

This gives both pipelines natural-language text rather than CSV, so neither is advantaged by
format. Result: `data/supply_chain_text.txt`, 61 KB.

---

## The Query Set

`data/test_queries.json` - 8 queries in three deliberate categories:

| Type | Count | Purpose | Expected winner |
|------|-------|---------|-----------------|
| `precise` | 3 | Single-fact lookup by ID | Vector RAG |
| `natural_language` | 2 | Fuzzy semantic retrieval | Vector RAG |
| `multi_hop` | 3 | Chained relational reasoning | GraphRAG |

The IDs used in the queries are real cryptographic hashes from the dataset, which makes it
effectively impossible for the LLM to answer from pretraining memory. Any correct answer must
come from retrieval.

---

## Repository Layout

| Path | Role |
|------|------|
| `src/config.py` | Single source of truth for models, endpoint, hyperparameters, budget, paths |
| `src/dataset_prep.py` | Kaggle fetch -> 100 rows -> plain-English corpus |
| `src/rag_baseline.py` | Vector RAG: `VectorStoreIndex`, persist/load cache, timed build + query |
| `src/graphrag_baseline.py` | GraphRAG: `KnowledgeGraphIndex` built via a threaded, staggered insert loop |
| `src/hybrid_rag.py` | Merges both retrievals (currently concatenates answers - see B17) |
| `src/evaluator.py` | Orchestrator: sweep, smart-skip cache, RAGAS scoring, incremental atomic saves |
| `src/generate_plots.py` | Matplotlib charts, log-scale build time, discrete x-ticks |
| `research/` | **Externally sourced and fact-checked** knowledge only |
| `research/problems_readme.md` | **The problems ledger. Start here.** |
| `.agent/memory/` | Chronological session memory (35 files, 2026-08-11 to 2026-08-21) |
| `.agent/temp/` | Scratch space. All temporary scripts must live and run here |
| `.continue/rules/` | Authoritative project rules (logging, rigor, memory discipline) |
| `data/` | Corpus, queries, indexes, results, plots (git-ignored) |

---

## Hyperparameters

Recorded here because reproducibility requires it.

| Parameter | Value | Where |
|-----------|-------|-------|
| Chunk sizes swept | `200, 300, 500` | `config.py` `CHUNK_SIZES` |
| Chunk overlap | `50` | `config.py` `CHUNK_OVERLAP` |
| top-k | `3` (config declares `[1, 3, 5]`, only 3 is used) | `evaluator.py` L212 |
| Graph max hops | declared `[1, 2, 3]`, **never used** | `config.py` |
| `max_triplets_per_chunk` | LlamaIndex default (10), never swept | `graphrag_baseline.py` |
| Generation model | `openai/gpt-oss-120b` | `config.py` `MODEL_NAME` |
| Judge model | `meta-llama/llama-3.1-8b-instruct` (**mismatch - see B2**) | `evaluator.py` L117 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, zero cost) | `config.py` |
| Temperature | `0.0` | `config.py` |
| Seed | `42` declared, **never applied to any RNG** | `config.py` |
| Budget cap | INR 20.0, **never enforced** | `config.py` |
| API base URL | `https://agentrouter.org/v1` | `config.py` |
| API key env var | `RAG_GRAPHRAG_KEY` | `config.py` |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Never install into the global interpreter. `requirements.txt` currently has **no version pins**,
which is a known reproducibility risk ([B21](research/problems_readme.md#b21---zero-version-pins))
given how sensitive this stack is to RAGAS minor versions.

Set the API key as a system environment variable (never in code, never committed):

```bash
setx RAG_GRAPHRAG_KEY "your-key-here"
```

**Windows caveat**: a newly created environment variable is not visible to already-running
processes. Restart the terminal and the IDE, or the whole machine, before running
([A3](research/problems_readme.md#a1-model--client--provider-errors)).

---

## Running

```bash
# 1. Build the corpus (once)
python src/dataset_prep.py

# 2. Run the benchmark sweep (chunk sizes 200, 300, 500)
python src/evaluator.py

# 3. Regenerate plots manually (evaluator.py already does this automatically)
python src/generate_plots.py
```

### Caching behaviour you need to understand

Three independent cache layers exist, all built in response to expensive re-runs:

1. **Index cache** - `data/index/rag_{chunk_size}` and `data/index/graphrag_{chunk_size}`.
   If the directory exists it is loaded instead of rebuilt. Indexes are written to a `_tmp`
   directory and promoted only on success, so an interrupted run cannot leave a corrupt cache.
   **Caveat**: the cache key is chunk size only, so changing the model or embeddings silently
   reuses a stale index ([B16](research/problems_readme.md#b16---index-cache-key-is-chunk_size-only)).
2. **Query cache** - `data/results/query_cache_{chunk_size}.json`, written *before* RAGAS runs
   so that generated answers survive an evaluation crash.
3. **Smart skip** - `evaluator.py` inspects `benchmark_metrics.json` and skips a chunk size
   entirely if it already has valid scalar scores for both pipelines, or reruns GraphRAG only
   if just the RAG half is complete.

**To force a full re-run**, delete the relevant entry from `data/results/benchmark_metrics.json`
(and the index directory if you want a genuine rebuild). There is no `--ignore-cache` flag yet
([A28](research/problems_readme.md#a4-evaluator-resilience-and-data-integrity)).

---

## Results So Far

### Index build cost (the only fully measured dimension)

| chunk_size | Vector RAG | GraphRAG | GraphRAG slowdown |
|-----------|-----------|----------|-------------------|
| 200 (157 chunks) | 5.11 s | 1339.40 s | **262x** |
| 300 (99 chunks) | 2.83 s | 1142.33 s | **404x** |
| 500 (50 chunks) | 1.48 s | 739.90 s | **500x** |

This is the clearest finding in the project so far, and it is intuitive: Vector RAG makes one
cheap local embedding pass, while GraphRAG makes one LLM call per chunk to extract triplets.

**Important caveat**: GraphRAG build time is wall-clock and includes a deliberate
0.8-2.0 s sleep per node used to avoid provider rate limits, i.e. roughly 125-315 s of the
1339 s figure is self-imposed throttling, not architecture
([B13](research/problems_readme.md#b13---sleep-based-staggering-pollutes-the-headline-build-time-metric)).

### Quality scores

**There are no trustworthy quality scores yet.** The committed run has `null` in all six RAGAS
columns. Transient values observed during debugging (Vector RAG 0.41 / 0.57 / 0.46 vs GraphRAG
0.31-0.49 / 0.00 / 0.00) are recorded in
[Section F.3](research/problems_readme.md#f3-ragas-scores-that-existed-only-transiently)
of the ledger together with why they cannot yet be believed - notably that GraphRAG's
`answer_relevancy` of 0.0 is an artifact of using an 8B judge model, not a GraphRAG failure.

### Cost

Never measured since the framework rewrite. Token accounting does not exist
([B5](research/problems_readme.md#b5---no-token-accounting-no-cost-calculation-no-budget-abort)).

---

## Project Standards

Enforced by `.continue/rules/rag-benchmark-standards.md` and `.continue/rules/general-rule.md`:

- **Data parity** - both pipelines ingest the exact same corpus, always.
- **Query symmetry** - identical query strings across every pipeline.
- **Zero system bias** - no handicapped prompts, no inflated retrieval windows.
- **Logging** - `[INFO] [DEBUG] [WARN] [ERROR] [METRIC] [TIMING]` prefixes only.
  **No emojis anywhere**, ever, in any print, log, or exception. Verified clean across `src/`.
- **Reproducibility** - seed RNGs and record every hyperparameter.
- **Resilience** - exponential backoff on external calls; one failed query must never crash a suite.
- **Memory discipline** - every session produces a new `.agent/memory/<date>_<time>_memory.md`.
  Never overwrite one.
- **`research/` is external knowledge only** - internet-sourced, official docs, or independently
  fact-checked. Never assumptions or scratch notes.
- **`.agent/temp/` for all temporary work** - never the project root.

A frank per-rule PASS / PARTIAL / FAIL audit lives in
[Section E of the ledger](research/problems_readme.md#section-e-rule-compliance-scorecard).

---

## Recommended Next Steps

In dependency order, from the ledger's severity ranking:

1. **Unblock the benchmark** - make `evaluator.py` read the base URL, key, and model from
   `config.py` (B1, B2, Section D). Nothing else can be trusted until this is done.
2. **Stop swallowing errors** - make RAGAS failures loud and never persist silent `null` rows (B3).
3. **Restore cost control** - token counting, correct provider rates, and the hard budget abort (B4, B5).
4. **Persist latency** - split retrieval from generation and write both to the results (B6).
5. **Enable Hybrid and widen the sweep** - so all three pipelines are compared and `TOP_K_VALUES`
   plus `GRAPH_MAX_HOPS` stop being dead config (B7).
6. **Fix measurement integrity** - build time on cache hits, sleep-inflated graph timings,
   `os.rename` on Windows, hyperparameter-keyed cache dirs (B9, B13, B14, B16).
7. **Use a stronger judge model** so `answer_relevancy` stops collapsing to 0.0 (A30).
8. **Pin dependencies and apply the seed** before publishing anything (B18, B21).

---

## License and Intent

Built as an open, auditable experiment. The goal is not to declare a winner but to publish the
actual trade-off curve - including the parts that do not work yet, which is why the ledger of
open problems is treated as a first-class deliverable rather than something to hide.
