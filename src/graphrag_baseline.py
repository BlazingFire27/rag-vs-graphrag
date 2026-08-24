"""
GraphRAG Baseline using LlamaIndex KnowledgeGraphIndex.
Replaces the hand-rolled LLM extraction and BFS traversal.
"""
import os
import random
import shutil
import time

import nest_asyncio
nest_asyncio.apply()

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from llama_index.core import (
    KnowledgeGraphIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.graph_stores import SimpleGraphStore

class GraphRAGBaseline:
    def __init__(self, chunk_size=300, chunk_overlap=50, max_triplets_per_chunk=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_triplets = max_triplets_per_chunk
        self.index = None
        self.query_engine = None
        self.build_time = 0.0
        self.load_time = 0.0
        self.graph_store = SimpleGraphStore()
        print(f"[INFO] Initialized GraphRAG Baseline (chunk_size={chunk_size}, max_triplets={max_triplets_per_chunk})")

    def _staggered_insert(self, node):
        # Stagger requests by 0.5s to 2.0s to avoid triggering instantaneous API limits
        time.sleep(random.uniform(0.8, 2.0))
        self.index.insert_nodes([node])

    def build_index(self, corpus_text):
        """Extract knowledge graph triplets and build a graph index."""
        
        persist_dir = f"data/index/graphrag_{self.chunk_size}"
        
        if os.path.exists(persist_dir):
            print(f"[INFO] Loading GraphRAG index from {persist_dir}...")
            start = time.time()
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.graph_store = storage_context.graph_store
            self.index = load_index_from_storage(storage_context)
            self.load_time = time.time() - start
            print(f"[TIMING] GraphRAG Index Load Time: {self.load_time:.4f}s")
            self.query_engine = self.index.as_query_engine(
                include_text=True,
                response_mode="tree_summarize",
            )
            return

        print(f"[INFO] Building GraphRAG index (LLM-based entity/relation extraction)...")
        start = time.time()

        documents = [Document(text=corpus_text)]
        splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)

        nodes = splitter.get_nodes_from_documents(documents)

        self.index = KnowledgeGraphIndex(
            [],
            storage_context=storage_context,
            max_triplets_per_chunk=self.max_triplets,
            include_embeddings=False,
        )

        print(f"[INFO] Running concurrent GraphRAG extraction across {len(nodes)} chunks (6 workers, staggered)...")
        
        # Temporarily disable auto-saving to prevent JSON serialization iteration crashes during threading
        original_add = storage_context.index_store.add_index_struct
        storage_context.index_store.add_index_struct = lambda x: None

        with ThreadPoolExecutor(max_workers=6) as executor:
            list(tqdm(executor.map(self._staggered_insert, nodes), total=len(nodes), desc="Extracting Graph"))

        # Restore and save index struct once manually
        storage_context.index_store.add_index_struct = original_add
        storage_context.index_store.add_index_struct(self.index.index_struct)
        
        # Persist to disk atomically
        tmp_dir = f"{persist_dir}_tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        storage_context.persist(persist_dir=tmp_dir)
        os.rename(tmp_dir, persist_dir)

        self.build_time = time.time() - start

        # Count graph elements
        try:
            graph_data = self.graph_store._data.graph_dict
            node_count = len(graph_data) if isinstance(graph_data, dict) else 0
        except AttributeError:
            node_count = 0
            
        print(f"[METRIC] GraphRAG Graph Subjects: {node_count}")
        print(f"[TIMING] GraphRAG Index Build Time: {self.build_time:.4f}s")

        self.query_engine = self.index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
        )

    def query(self, query_text):
        """Run a graph traversal + LLM generation."""
        if not self.query_engine:
            print("[ERROR] Index not built. Call build_index() first.")
            return None

        print(f"[INFO] Executing graph traversal query...")
        start = time.time()
        response = self.query_engine.query(query_text)
        latency = time.time() - start
        print(f"[TIMING] GraphRAG Query Latency: {latency:.4f}s")

        contexts = []
        if response.source_nodes:
            for node in response.source_nodes:
                contexts.append(node.node.get_content())

        return {
            "answer": str(response),
            "contexts": contexts,
            "latency": latency
        }
