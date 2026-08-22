"""
Vector RAG Baseline using LlamaIndex VectorStoreIndex.
Replaces the hand-rolled chunking/embedding/cosine logic.
"""
import time
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from config import EMBEDDING_MODEL, SEED


class RAGBaseline:
    def __init__(self, chunk_size=300, chunk_overlap=50, top_k=3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.index = None
        self.query_engine = None
        self.build_time = 0.0
        self.load_time = 0.0
        self.node_count = 0
        print(f"[INFO] Initialized Vector RAG Baseline (chunk_size={chunk_size}, overlap={chunk_overlap}, top_k={top_k})")

    def build_index(self, corpus_text):
        """Chunk the corpus and build a vector index."""
        import os
        import shutil
        from llama_index.core import StorageContext, load_index_from_storage
        
        persist_dir = f"data/index/rag_{self.chunk_size}"
        tmp_dir = f"{persist_dir}_tmp"
        
        if os.path.exists(persist_dir):
            print(f"[INFO] Loading Vector RAG index from {persist_dir}...")
            start = time.time()
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
            self.node_count = len(self.index.docstore.docs)
            self.load_time = time.time() - start
            print(f"[METRIC] Vector RAG Nodes Indexed: {self.node_count}")
            print(f"[TIMING] Vector RAG Index Load Time: {self.load_time:.4f}s")
            self.query_engine = self.index.as_query_engine(similarity_top_k=self.top_k)
            return

        print(f"[INFO] Building Vector RAG index...")
        start = time.time()

        # Create a LlamaIndex Document from the raw text
        documents = [Document(text=corpus_text)]

        # Use SentenceSplitter for deterministic chunking
        splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        # Build the vector index (embedding happens automatically via Settings)
        self.index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter],
            show_progress=True
        )
        
        # Save to tmp dir first, then rename atomically
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        self.index.storage_context.persist(persist_dir=tmp_dir)
        os.rename(tmp_dir, persist_dir)

        self.node_count = len(self.index.docstore.docs)
        self.build_time = time.time() - start

        print(f"[METRIC] Vector RAG Nodes Indexed: {self.node_count}")
        print(f"[TIMING] Vector RAG Index Build Time: {self.build_time:.4f}s")

        # Create query engine with the configured top_k
        self.query_engine = self.index.as_query_engine(similarity_top_k=self.top_k)

    def query(self, query_text):
        """Run a vector similarity search + LLM generation."""
        if not self.query_engine:
            print("[ERROR] Index not built. Call build_index() first.")
            return None

        print(f"[INFO] Executing vector search (top_k={self.top_k})...")
        start = time.time()
        response = self.query_engine.query(query_text)
        latency = time.time() - start
        print(f"[TIMING] Vector RAG Query Latency: {latency:.4f}s")

        # Extract the retrieved context chunks for RAGAS evaluation
        contexts = []
        if response.source_nodes:
            for node in response.source_nodes:
                contexts.append(node.node.get_content())

        return {
            "answer": str(response),
            "contexts": contexts,
            "latency": latency
        }
