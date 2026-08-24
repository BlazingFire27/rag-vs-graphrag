"""
RAG vs GraphRAG Benchmark Evaluator
Uses LlamaIndex for indexing/retrieval and RAGAS for evaluation.
Runs a sweep across chunk sizes and records all metrics.
"""
import os
import sys
import time
import json
import csv
import tracemalloc
import types
import importlib
import traceback

import numpy as np
from datasets import Dataset

from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic as LlamaIndexAnthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import anthropic as anthropic_sdk
from anthropic.resources.messages import Messages as _AnthropicMessages

from config import (
    API_BASE_URL, API_KEY_ENV_VAR, MODEL_NAME, JUDGE_MODEL_NAME, EMBEDDING_MODEL,
    get_api_key, CHUNK_SIZES, CHUNK_OVERLAP, TEMPERATURE, MAX_TOKENS,
    CORPUS_PATH, QUERIES_PATH, RESULTS_DIR, PLOTS_DIR, ANTHROPIC_CLIENT_HEADERS,
)
from rag_baseline import RAGBaseline
from graphrag_baseline import GraphRAGBaseline
from generate_plots import generate_all_plots

# Monkey-patch: RAGAS 0.4.3 tries to import langchain_community.chat_models.vertexai
# which was removed in langchain-community 0.4.x. Stub it out before importing RAGAS.
def _patch_ragas_vertexai():
    """Stub the missing vertexai module so RAGAS can import cleanly."""
    try:
        importlib.import_module("langchain_community.chat_models.vertexai")
    except (ImportError, ModuleNotFoundError):
        stub = types.ModuleType("langchain_community.chat_models.vertexai")
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = stub

_patch_ragas_vertexai()

# Try importing RAGAS -- graceful fallback if not available
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    from ragas.llms import llm_factory
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
    print("[INFO] RAGAS loaded successfully.")
except ImportError as e:
    print(f"[WARN] RAGAS not available ({e}). Falling back to manual scoring.")
    RAGAS_AVAILABLE = False


def load_corpus():
    if not os.path.exists(CORPUS_PATH) or os.path.getsize(CORPUS_PATH) == 0:
        print(f"[ERROR] Corpus not found or empty at {CORPUS_PATH}. Run dataset_prep.py first.")
        sys.exit(1)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return f.read()


def load_queries():
    with open(QUERIES_PATH, "r") as f:
        return json.load(f)


def setup_llama_index(api_key):
    """Configure LlamaIndex global settings with Anthropic-native LLM."""
    print("[INFO] Configuring LlamaIndex Settings...")
    print(f"[INFO]   LLM: {MODEL_NAME} via {API_BASE_URL} (Anthropic-native)")
    print(f"[INFO]   Embedding: {EMBEDDING_MODEL} (local)")

    Settings.llm = LlamaIndexAnthropic(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=API_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        default_headers=ANTHROPIC_CLIENT_HEADERS,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.num_workers = 6
    print("[INFO] LlamaIndex Settings configured successfully (num_workers=6).")


def run_ragas_evaluation(questions, answers, contexts_list, ground_truths):
    """Run RAGAS evaluation on collected results."""
    if not RAGAS_AVAILABLE:
        print("[WARN] RAGAS not installed, skipping automated scoring.")
        return {}

    print("[INFO] Running RAGAS evaluation...")
    try:
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)

        # --- B23 FIX: Monkey-patch anthropic SDK Messages.create() ---
        # RAGAS 0.4.3 / instructor injects sampling params (temperature, top_p,
        # top_k) directly into anthropic.resources.messages.Messages.create(),
        # but anthropic SDK 1.0.0 removed them from the signature. We patch at
        # the SDK class level so isinstance checks (used by instructor) still pass.
        _STRIP_KWARGS = {"temperature", "top_p", "top_k"}
        if not getattr(_AnthropicMessages, '_b23_patched', False):
            _orig_create = _AnthropicMessages.create
            def _patched_create(self, *args, **kwargs):
                for k in _STRIP_KWARGS:
                    kwargs.pop(k, None)
                return _orig_create(self, *args, **kwargs)
            _AnthropicMessages.create = _patched_create
            _AnthropicMessages._b23_patched = True
            print(f"[DEBUG] Applied B23 monkey-patch: stripping {_STRIP_KWARGS} from Messages.create().")

        # 1. Anthropic-native client, configured ENTIRELY from config.py.
        print(f"[INFO] RAGAS judge LLM: {JUDGE_MODEL_NAME} via {API_BASE_URL} (Anthropic-native)")
        print(f"[INFO] RAGAS judge key source: os.environ['{API_KEY_ENV_VAR}']")
        anthropic_client = anthropic_sdk.Anthropic(
            api_key=get_api_key(),
            base_url=API_BASE_URL,
            default_headers=ANTHROPIC_CLIENT_HEADERS,
        )

        # 2. Build the RAGAS InstructorLLM via llm_factory with provider="anthropic".
        #    JUDGE_MODEL_NAME is identical to MODEL_NAME by construction (A37).
        evaluator_llm = llm_factory(
            model=JUDGE_MODEL_NAME,
            provider="anthropic",
            client=anthropic_client,
        )

        # 3. Robust LangChain wrapper for modern embeddings (A12).
        hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

        results = ragas_evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(llm=evaluator_llm),
                AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
                ContextPrecision(llm=evaluator_llm)
            ],
        )
        print(f"[METRIC] RAGAS Scores: {results}")
        return {
            "faithfulness": round(float(np.nanmean(results["faithfulness"])), 4),
            "answer_relevancy": round(float(np.nanmean(results["answer_relevancy"])), 4),
            "context_precision": round(float(np.nanmean(results["context_precision"])), 4)
        }
    except Exception as e:
        print(f"[WARN] RAGAS evaluation failed: {e}")
        traceback.print_exc()
        return {}


def _incremental_save(all_results, json_path):
    """Atomically save all_results to JSON and CSV after each chunk completes."""
    # JSON
    tmp_json = f"{json_path}_tmp"
    with open(tmp_json, "w") as f:
        json.dump({"sweep_results": all_results}, f, indent=4)
    os.replace(tmp_json, json_path)
    # CSV
    csv_path = os.path.join(RESULTS_DIR, "benchmark_metrics.csv")
    tmp_csv = f"{csv_path}_tmp"
    fieldnames = list(all_results[0].keys())
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    os.replace(tmp_csv, csv_path)
    print(f"[INFO] Incremental save complete ({len(all_results)} entries).")


def evaluate_performance():
    print("=" * 70)
    print("[INFO] RAG vs GraphRAG Benchmark Suite (LlamaIndex + RAGAS)")
    print("=" * 70)

    # --- Pre-Execution Validation ---
    print("[INFO] Validating pre-execution requirements...")
    api_key = get_api_key()
    print(f"[INFO] Using API Key: {api_key[:5]}***")

    corpus = load_corpus()
    queries = load_queries()
    setup_llama_index(api_key)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_results = []
    
    # --- Load Cached Metrics ---
    json_path = os.path.join(RESULTS_DIR, "benchmark_metrics.json")
    cached_metrics = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                for entry in data.get("sweep_results", []):
                    cs = entry.get("chunk_size")
                    # Coerce any raw score arrays into scalar averages (fixes earlier bug)
                    for key in ["rag_faithfulness", "rag_answer_relevancy", "rag_context_precision",
                                "graphrag_faithfulness", "graphrag_answer_relevancy", "graphrag_context_precision"]:
                        val = entry.get(key)
                        if isinstance(val, list):
                            entry[key] = round(float(np.nanmean(val)), 4) if val else None
                    rag_ok = isinstance(entry.get("rag_faithfulness"), (int, float))
                    grag_ok = isinstance(entry.get("graphrag_faithfulness"), (int, float))
                    if rag_ok and grag_ok:
                        cached_metrics[cs] = entry
                    elif rag_ok and not grag_ok:
                        # RAG is done but GraphRAG needs rerun
                        cached_metrics[cs] = {"_rag_only": True, **entry}
            print(f"[INFO] Loaded {len(cached_metrics)} cached chunk sizes from disk.")
        except Exception as e:
            print(f"[WARN] Failed to load cached metrics: {e}")

    # =====================================================================
    # SWEEP: Iterate over chunk sizes
    # =====================================================================
    for chunk_size in CHUNK_SIZES:
        top_k = 3  # Fixed for chunk size sweep
        print("")
        print("=" * 70)
        print(f"[INFO] SWEEP: chunk_size={chunk_size}, top_k={top_k}")
        print("=" * 70)

        cached = cached_metrics.get(chunk_size)

        # --- FULL SKIP: Both pipelines have valid scalar scores ---
        if cached and not cached.get("_rag_only"):
            print(f"[INFO] chunk_size={chunk_size} fully evaluated. Skipping.")
            all_results.append(cached)
            _incremental_save(all_results, json_path)
            continue

        # --- Build Indices (always needed for querying) ---
        print("[INFO] --- Building Vector RAG Index ---")
        tracemalloc.start()
        rag = RAGBaseline(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP, top_k=top_k)
        rag.build_index(corpus)
        rag_mem_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        print("[INFO] --- Building GraphRAG Index ---")
        tracemalloc.start()
        graphrag = GraphRAGBaseline(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)
        graphrag.build_index(corpus)
        graphrag_mem_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        # --- Load or Generate Query Answers ---
        query_cache_path = os.path.join(RESULTS_DIR, f"query_cache_{chunk_size}.json")
        rag_questions, rag_answers, rag_contexts, ground_truths = [], [], [], []
        grag_questions, grag_answers, grag_contexts = [], [], []
        rag_cached = False
        grag_cached = False

        if os.path.exists(query_cache_path):
            try:
                with open(query_cache_path, "r") as f:
                    qc = json.load(f)
                if qc.get("rag_answers"):
                    rag_questions = qc["rag_questions"]
                    rag_answers = qc["rag_answers"]
                    rag_contexts = qc["rag_contexts"]
                    ground_truths = qc["ground_truths"]
                    rag_cached = True
                    print(f"[INFO] Loaded cached RAG query answers.")
                if qc.get("grag_answers"):
                    grag_questions = qc["grag_questions"]
                    grag_answers = qc["grag_answers"]
                    grag_contexts = qc["grag_contexts"]
                    grag_cached = True
                    print(f"[INFO] Loaded cached GraphRAG query answers.")
            except Exception as e:
                print(f"[WARN] Failed to load query cache: {e}")

        # If RAG is already cached AND has valid scores, skip RAG queries
        need_rag_queries = not rag_cached
        # If GraphRAG had bad scores (0.0 arrays or None), force rerun
        need_grag_queries = not grag_cached
        if cached and cached.get("_rag_only"):
            need_rag_queries = False
            need_grag_queries = True
            # Clear stale GraphRAG answers from cache
            grag_questions, grag_answers, grag_contexts = [], [], []
            print("[INFO] RAG scores valid from cache. Only rerunning GraphRAG queries.")

        # Run queries that are needed
        for i, q in enumerate(queries):
            query_text = q["query"]
            query_type = q.get("type", "unknown")
            gt = q.get("ground_truth", "")
            print(f"\n--- Query {i+1}/{len(queries)} [{query_type}] (chunk={chunk_size}) ---")
            print(f"[INFO] Q: {query_text}")

            if need_rag_queries:
                try:
                    rag_result = rag.query(query_text)
                    rag_questions.append(query_text)
                    rag_answers.append(rag_result["answer"])
                    rag_contexts.append(rag_result["contexts"])
                    ground_truths.append(gt)
                except Exception as e:
                    print(f"[ERROR] Vector RAG query failed: {e}")
                    rag_questions.append(query_text)
                    rag_answers.append("ERROR")
                    rag_contexts.append([])
                    ground_truths.append(gt)

            if need_grag_queries:
                try:
                    grag_result = graphrag.query(query_text)
                    grag_questions.append(query_text)
                    grag_answers.append(grag_result["answer"])
                    grag_contexts.append(grag_result["contexts"])
                except Exception as e:
                    print(f"[ERROR] GraphRAG query failed: {e}")
                    grag_questions.append(query_text)
                    grag_answers.append("ERROR")
                    grag_contexts.append([])

        # Save query cache (always overwrite with latest)
        try:
            with open(query_cache_path, "w") as f:
                json.dump({
                    "rag_questions": rag_questions,
                    "rag_answers": rag_answers,
                    "rag_contexts": rag_contexts,
                    "ground_truths": ground_truths,
                    "grag_questions": grag_questions,
                    "grag_answers": grag_answers,
                    "grag_contexts": grag_contexts
                }, f, indent=4)
            print(f"[INFO] Saved query cache to {query_cache_path}")
        except Exception as e:
            print(f"[WARN] Failed to save query cache: {e}")

        # --- RAGAS Evaluation ---
        print("\n[INFO] --- RAGAS Evaluation ---")

        # Use cached RAG scores if available and valid
        if cached and cached.get("_rag_only") and isinstance(cached.get("rag_faithfulness"), (int, float)):
            rag_scores = {
                "faithfulness": cached["rag_faithfulness"],
                "answer_relevancy": cached["rag_answer_relevancy"],
                "context_precision": cached["rag_context_precision"]
            }
            print(f"[INFO] Using cached RAG RAGAS scores: {rag_scores}")
        else:
            rag_scores = run_ragas_evaluation(rag_questions, rag_answers, rag_contexts, ground_truths)

        grag_scores = run_ragas_evaluation(grag_questions, grag_answers, grag_contexts, ground_truths)

        # --- Record Results ---
        result_entry = {
            "chunk_size": chunk_size,
            "top_k": top_k,
            "rag_build_time_s": round(rag.build_time, 4),
            "rag_node_count": rag.node_count,
            "rag_peak_memory_mb": round(rag_mem_peak, 2),
            "graphrag_build_time_s": round(graphrag.build_time, 4),
            "graphrag_peak_memory_mb": round(graphrag_mem_peak, 2),
            "rag_faithfulness": rag_scores.get("faithfulness", None),
            "rag_answer_relevancy": rag_scores.get("answer_relevancy", None),
            "rag_context_precision": rag_scores.get("context_precision", None),
            "graphrag_faithfulness": grag_scores.get("faithfulness", None),
            "graphrag_answer_relevancy": grag_scores.get("answer_relevancy", None),
            "graphrag_context_precision": grag_scores.get("context_precision", None),
        }
        all_results.append(result_entry)

        print(f"\n[METRIC] Sweep result for chunk_size={chunk_size}:")
        for k, v in result_entry.items():
            print(f"  {k}: {v}")

        # --- Incremental Atomic Save ---
        _incremental_save(all_results, json_path)

    # =====================================================================
    # PERSIST ALL RESULTS
    # =====================================================================
    print("\n" + "=" * 70)
    print("[INFO] PERSISTING METRICS")
    print("=" * 70)

    # JSON
    json_path = os.path.join(RESULTS_DIR, "benchmark_metrics.json")
    with open(json_path, "w") as f:
        json.dump({"sweep_results": all_results}, f, indent=4)
    print(f"[INFO] Saved JSON metrics to {json_path}")

    # CSV
    if all_results:
        csv_path = os.path.join(RESULTS_DIR, "benchmark_metrics.csv")
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[INFO] Saved CSV metrics to {csv_path}")

    # --- Generate Plots ---
    print("[INFO] Generating plots...")
    try:
        generate_all_plots(all_results)
    except Exception as e:
        print(f"[WARN] Plot generation failed: {e}")

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("[INFO] BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"  Chunk sizes tested:  {CHUNK_SIZES}")
    print(f"  Queries per sweep:   {len(queries)}")
    print(f"  Total sweep runs:    {len(all_results)}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_performance()
