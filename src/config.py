"""
Centralized configuration for the RAG vs GraphRAG benchmark suite.
All hyperparameters are recorded here for reproducibility (RESEARCH_RULES.md).
"""
import os

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH. Never hardcode a provider, key, or model anywhere else.
# Every module (evaluator, baselines, judge) MUST import these constants.
#
# Integrity requirement: the generation LLM and the RAGAS judge LLM are the SAME
# model on the SAME endpoint. A generator/judge mismatch invalidates cross-run
# score comparison.
#
# -------------------------- ENDPOINT CONTRACT ------------------------------
# 1. The endpoint MUST be ANTHROPIC-NATIVE (not OpenAI-compatible).
# 2. Update the API_BASE_URL placeholder below with your actual Anthropic-native
#    base URL (e.g., bare root without /v1/messages suffix, as SDKs append it).
# 3. Auth uses the "x-api-key" header automatically set by the SDKs.
# 4. If using a gateway that gates on client identity, ensure ANTHROPIC_CLIENT_HEADERS
#    are passed to make the client acceptable.
# ---------------------------------------------------------------------------
API_BASE_URL = "<YOUR_ANTHROPIC_BASE_URL>"
API_KEY_ENV_VAR = "RAG_GRAPHRAG_KEY"

# Anthropic Messages API version header.
ANTHROPIC_VERSION = "2023-06-01"

# Headers required to pass the gateway's client-identity check (point 4 above).
ANTHROPIC_CLIENT_HEADERS = {
    "user-agent": "Anthropic/Python 0.40.0",
    "x-stainless-lang": "python",
    "x-stainless-package-version": "0.40.0",
}

# Only claude-opus-5 and claude-opus-4.8 are provisioned on this key.
MODEL_NAME = "claude-opus-5"
# The RAGAS judge model. Intentionally identical to MODEL_NAME for integrity.
JUDGE_MODEL_NAME = MODEL_NAME

# Max tokens per generation call. Anthropic's Messages API requires this
# explicitly; there is no provider-side default.
MAX_TOKENS = 1024

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_api_key():
    """Fetch the API key from the system environment. Never read from a file."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        print(f"[ERROR] {API_KEY_ENV_VAR} environment variable not found.")
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
