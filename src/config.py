"""
Centralized configuration for the RAG vs GraphRAG benchmark suite.
All hyperparameters are recorded here for reproducibility (RESEARCH_RULES.md).
"""
import os

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = "https://api.aicredits.in/v1"
MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_api_key():
    api_key = os.environ.get("AICREDITS_RAG_API_KEY")
    if not api_key:
        print("[ERROR] AICREDITS_RAG_API_KEY environment variable not found.")
        raise ValueError("Missing API Key")
    return api_key

# ---------------------------------------------------------------------------
# Benchmark Sweep Parameters
# ---------------------------------------------------------------------------
CHUNK_SIZES = [200, 300, 500]
TOP_K_VALUES = [1, 3, 5]
GRAPH_MAX_HOPS = [1, 2, 3]

# ---------------------------------------------------------------------------
# Fixed Hyperparameters
# ---------------------------------------------------------------------------
CHUNK_OVERLAP = 50
TEMPERATURE = 0.0
SEED = 42

# ---------------------------------------------------------------------------
# Budget Control (RESEARCH_RULES.md)
# ---------------------------------------------------------------------------
BUDGET_CAP_INR = 20.0
# Approximate Llama 3.1 8B rates (INR per 1M tokens)
COST_PER_1M_INPUT = 1.5
COST_PER_1M_OUTPUT = 2.0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORPUS_PATH = "data/supply_chain_text.txt"
QUERIES_PATH = "data/test_queries.json"
RESULTS_DIR = "data/results"
PLOTS_DIR = "data/results/plots"
