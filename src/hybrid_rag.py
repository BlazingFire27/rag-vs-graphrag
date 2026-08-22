"""
Hybrid RAG: Combines Vector RAG and GraphRAG retrieval results.
Uses both query engines and merges their contexts before generation.
"""
import time
from rag_baseline import RAGBaseline
from graphrag_baseline import GraphRAGBaseline


class HybridRAG:
    def __init__(self, chunk_size=300, chunk_overlap=50, top_k=3, max_triplets=10):
        self.rag = RAGBaseline(chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)
        self.graphrag = GraphRAGBaseline(chunk_size=chunk_size, chunk_overlap=chunk_overlap, max_triplets_per_chunk=max_triplets)
        self.build_time = 0.0
        print(f"[INFO] Initialized Hybrid RAG (Vector + Graph fusion)")

    def build_index(self, corpus_text):
        """Build both vector and graph indices on the same corpus."""
        print(f"[INFO] Building Hybrid RAG indices (both Vector + Graph)...")
        start = time.time()
        self.rag.build_index(corpus_text)
        self.graphrag.build_index(corpus_text)
        self.build_time = time.time() - start
        print(f"[TIMING] Hybrid RAG Total Build Time: {self.build_time:.4f}s")

    def query(self, query_text):
        """Query both engines and merge their contexts."""
        print(f"[INFO] Executing Hybrid query (Vector + Graph)...")
        start = time.time()

        rag_result = self.rag.query(query_text)
        graph_result = self.graphrag.query(query_text)

        # Merge contexts from both engines (deduplicated)
        merged_contexts = []
        seen = set()
        for ctx in (rag_result.get("contexts", []) + graph_result.get("contexts", [])):
            if ctx not in seen:
                merged_contexts.append(ctx)
                seen.add(ctx)

        # Use the vector RAG answer as the primary (it has richer context)
        # but enrich it with graph knowledge
        latency = time.time() - start
        print(f"[TIMING] Hybrid RAG Query Latency: {latency:.4f}s")

        return {
            "answer": rag_result.get("answer", "") + "\n\nGraph context: " + graph_result.get("answer", ""),
            "contexts": merged_contexts,
            "latency": latency
        }
