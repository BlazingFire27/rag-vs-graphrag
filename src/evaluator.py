import os
import time
import json
from rag_baseline import RAGBaseline
from graphrag_baseline import GraphRAGBaseline
from hybrid_rag import HybridRAG

def get_api_key():
    api_key = os.environ.get("AICREDITS_RAG_API_KEY")
    if not api_key:
        print("[ERROR] AICREDITS_RAG_API_KEY environment variable not found.")
        raise ValueError("Missing API Key")
    return api_key

def evaluate_performance():
    print("[INFO] Validating pre-execution requirements...")
    api_key = get_api_key()
    print(f"[INFO] Using API Key: {api_key[:5]}***")
    print("[INFO] Loading identical cleaned text corpus for Absolute Data Parity...")
    
    text_corpus_path = "data/supply_chain_text.txt"
    if not os.path.exists(text_corpus_path):
        print(f"[ERROR] Corpus file not found at {text_corpus_path}. Run dataset_prep.py first.")
        return
        
    with open(text_corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    print("[INFO] Starting Evaluation Suite...")
    
    db_mock = "ArcadeDB_Connection_Mock"
    
    # 1. Evaluate Vector RAG
    print("[INFO] --- Evaluating Vector RAG ---")
    start_time = time.time()
    rag = RAGBaseline(db_mock)
    rag.chunk_and_index_document(corpus)
    rag_build_time = time.time() - start_time
    print(f"[TIMING] Vector RAG Index Build Time: {rag_build_time:.4f}s")
    
    # 2. Evaluate GraphRAG
    print("[INFO] --- Evaluating GraphRAG ---")
    start_time = time.time()
    graph_rag = GraphRAGBaseline(db_mock)
    graph_rag.index_graph(corpus)
    graph_build_time = time.time() - start_time
    print(f"[TIMING] GraphRAG Index Build Time: {graph_build_time:.4f}s")
    
    # Save Metrics
    metrics = {
        "rag_build_latency_s": rag_build_time,
        "graph_build_latency_s": graph_build_time
    }
    
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/benchmark_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("[INFO] Successfully saved metrics to data/results/benchmark_metrics.json")

if __name__ == "__main__":
    evaluate_performance()
