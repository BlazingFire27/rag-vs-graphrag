## RAG vs GraphRAG Framework Research (August 2026)

### Established Frameworks for Production RAG Benchmarking

#### 1. LlamaIndex (Recommended for this project)
- **VectorStoreIndex**: Battle-tested vector RAG with built-in chunking, embedding, and retrieval.
- **KnowledgeGraphIndex / PropertyGraphIndex**: Automated entity/relation extraction and graph traversal.
- Both indices share the same `Settings` object (LLM, embedding model), ensuring Absolute Data Parity.
- Both produce `query_engine` objects with identical `.query()` interfaces, enabling Query Symmetry.
- Supports `SimpleGraphStore` (in-memory) for lightweight benchmarking without external DB dependencies.

#### 2. RAGAS (Retrieval Augmented Generation Assessment)
- Industry-standard evaluation framework for RAG pipelines.
- Provides: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
- Works as an LLM-as-a-Judge but with mathematically rigorous decomposition of claims.
- Integrates directly with LlamaIndex and LangChain outputs.
- `pip install ragas`

#### 3. nano-graphrag
- Lightweight (~1100 lines), hackable GraphRAG implementation.
- Good for prototyping but less suitable for controlled benchmarking vs. a vector baseline.

### Decision: LlamaIndex + RAGAS
- **LlamaIndex** handles both RAG architectures with minimal custom code.
- **RAGAS** replaces our hand-rolled LLM-as-a-Judge with a standardized, reproducible scoring system.
- This combination is what startups and research teams actually use in production.

### Code Structure (LlamaIndex)
```python
from llama_index.core import VectorStoreIndex, KnowledgeGraphIndex, SimpleDirectoryReader, Settings
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="meta-llama/llama-3.1-8b-instruct", api_base="https://api.aicredits.in/v1")
documents = SimpleDirectoryReader("data").load_data()

# Vector RAG
vector_index = VectorStoreIndex.from_documents(documents)
vector_engine = vector_index.as_query_engine()

# Knowledge Graph RAG
graph_store = SimpleGraphStore()
kg_index = KnowledgeGraphIndex.from_documents(documents, storage_context=...)
kg_engine = kg_index.as_query_engine()
```

Sources: LlamaIndex docs, RAGAS docs, web research August 2026
