from rag_baseline import RAGBaseline
from graphrag_baseline import GraphRAGBaseline

class HybridRAG:
    def __init__(self, db_client):
        self.rag = RAGBaseline(db_client)
        self.graph = GraphRAGBaseline(db_client)

    def query(self, query_text):
        vector_context = self.rag.query(query_text)
        graph_context = self.graph.query(query_text)
        return f"Hybrid Result combining: \n1. {vector_context}\n2. {graph_context}"
