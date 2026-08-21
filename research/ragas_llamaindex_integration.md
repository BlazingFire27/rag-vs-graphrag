# Research: RAGAS v0.4.x Integration Findings
# Source: Web searches conducted 2026-08-15 and 2026-08-16
# Verified against: ragas GitHub, LlamaIndex docs, LangChain docs

## 1. RAGAS v0.4.x EvaluationResult Object
- `evaluate()` returns a custom `EvaluationResult`, NOT a Python dict.
- `dict(results)` causes `KeyError: 0` — do NOT cast directly.
- `results["metric_name"]` returns a LIST of per-row scores, NOT a scalar average.
- To get aggregate: use `np.nanmean(results["metric_name"])`.
- The `print(result)` display shows the averaged scores automatically.
- Source: ragas GitHub issues, Stack Overflow

## 2. RAGAS v0.4.x Embedding Compatibility
- `ragas.embeddings.HuggingFaceEmbeddings` is BROKEN — missing `embed_query` method.
- `LlamaIndexEmbeddingsWrapper` is REJECTED by collections metrics.
- Working solution: `langchain_huggingface.HuggingFaceEmbeddings` wrapped in `ragas.embeddings.LangchainEmbeddingsWrapper`.
- Requires `pip install langchain-huggingface` (standalone package since LangChain decoupled HuggingFace).
- `LangchainEmbeddingsWrapper` shows a DeprecationWarning but WORKS correctly. The suggested replacement (`ragas.embeddings.HuggingFaceEmbeddings`) is the one that is broken.
- Source: ragas GitHub, LangChain migration guide

## 3. RAGAS v0.4.x LLM Factory
- `LangchainLLMWrapper` is deprecated.
- Correct pattern: `from ragas.llms import llm_factory; llm = llm_factory(model="...", client=openai_client)`
- Works with custom OpenAI-compatible endpoints (e.g., AICredits) by passing a configured `openai.OpenAI` client.
- Source: ragas deprecation warnings, ragas docs

## 4. LlamaIndex KnowledgeGraphIndex Persistence
- Persist: `index.storage_context.persist(persist_dir="./storage")`
- Reload: `StorageContext.from_defaults(persist_dir=...)` then `load_index_from_storage(storage_context)`
- CRITICAL: Do NOT pass a fresh `graph_store=SimpleGraphStore()` when loading from disk — this overwrites the cached graph with an empty one.
- Correct: `StorageContext.from_defaults(persist_dir=persist_dir)` alone. Then `self.graph_store = storage_context.graph_store`.
- Source: LlamaIndex docs, GitHub issues

## 5. Atomic File Saving (Python on Windows)
- `os.replace(src, dst)` is the correct cross-platform atomic rename (works on Windows even if dst exists).
- `os.rename(src, dst)` fails on Windows if dst already exists.
- Pattern: write to `file_tmp`, then `os.replace(file_tmp, file)`.
- Source: Python docs, Stack Overflow
