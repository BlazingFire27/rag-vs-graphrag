"""
Visualization module for RAG vs GraphRAG benchmark results.
Auto-triggered by evaluator.py after benchmark runs.
"""
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt

PLOTS_DIR = "data/results/plots"


def generate_all_plots(results):
    """Generate all benchmark visualization plots from sweep results."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not results:
        print("[WARN] No results to plot.")
        return

    chunk_sizes = [r["chunk_size"] for r in results]

    # -----------------------------------------------------------------
    # Plot 1: Build Time vs Chunk Size (Scalability)
    # -----------------------------------------------------------------
    try:
        rag_build = [r["rag_build_time_s"] for r in results]
        grag_build = [r["graphrag_build_time_s"] for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(chunk_sizes))
        width = 0.35
        ax.bar([i - width/2 for i in x], rag_build, width, label="Vector RAG", color="#4A90D9")
        ax.bar([i + width/2 for i in x], grag_build, width, label="GraphRAG", color="#E74C3C")
        ax.set_xlabel("Chunk Size")
        ax.set_ylabel("Build Time (seconds)")
        ax.set_title("Index Build Time vs Chunk Size")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(c) for c in chunk_sizes])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "build_time_vs_chunk_size.png"), dpi=150)
        plt.close(fig)
        print("[INFO] Saved plot: build_time_vs_chunk_size.png")
    except Exception as e:
        print(f"[WARN] Failed to generate build time plot: {e}")

    # -----------------------------------------------------------------
    # Plot 2: Memory Consumption Comparison
    # -----------------------------------------------------------------
    try:
        rag_mem = [r["rag_peak_memory_mb"] for r in results]
        grag_mem = [r["graphrag_peak_memory_mb"] for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(chunk_sizes, rag_mem, "o-", label="Vector RAG", color="#4A90D9", linewidth=2)
        ax.plot(chunk_sizes, grag_mem, "s-", label="GraphRAG", color="#E74C3C", linewidth=2)
        ax.set_xlabel("Chunk Size")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Peak Memory Consumption vs Chunk Size")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "memory_vs_chunk_size.png"), dpi=150)
        plt.close(fig)
        print("[INFO] Saved plot: memory_vs_chunk_size.png")
    except Exception as e:
        print(f"[WARN] Failed to generate memory plot: {e}")

    # -----------------------------------------------------------------
    # Plot 3: RAGAS Faithfulness Comparison
    # -----------------------------------------------------------------
    try:
        rag_faith = [r.get("rag_faithfulness") or 0 for r in results]
        grag_faith = [r.get("graphrag_faithfulness") or 0 for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(chunk_sizes))
        width = 0.35
        ax.bar([i - width/2 for i in x], rag_faith, width, label="Vector RAG", color="#4A90D9")
        ax.bar([i + width/2 for i in x], grag_faith, width, label="GraphRAG", color="#E74C3C")
        ax.set_xlabel("Chunk Size")
        ax.set_ylabel("Faithfulness Score (0-1)")
        ax.set_title("RAGAS Faithfulness vs Chunk Size")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(c) for c in chunk_sizes])
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "faithfulness_vs_chunk_size.png"), dpi=150)
        plt.close(fig)
        print("[INFO] Saved plot: faithfulness_vs_chunk_size.png")
    except Exception as e:
        print(f"[WARN] Failed to generate faithfulness plot: {e}")

    # -----------------------------------------------------------------
    # Plot 4: Context Precision Comparison
    # -----------------------------------------------------------------
    try:
        rag_cp = [r.get("rag_context_precision") or 0 for r in results]
        grag_cp = [r.get("graphrag_context_precision") or 0 for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(chunk_sizes))
        width = 0.35
        ax.bar([i - width/2 for i in x], rag_cp, width, label="Vector RAG", color="#4A90D9")
        ax.bar([i + width/2 for i in x], grag_cp, width, label="GraphRAG", color="#E74C3C")
        ax.set_xlabel("Chunk Size")
        ax.set_ylabel("Context Precision (0-1)")
        ax.set_title("RAGAS Context Precision vs Chunk Size")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(c) for c in chunk_sizes])
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "context_precision_vs_chunk_size.png"), dpi=150)
        plt.close(fig)
        print("[INFO] Saved plot: context_precision_vs_chunk_size.png")
    except Exception as e:
        print(f"[WARN] Failed to generate context precision plot: {e}")

    # -----------------------------------------------------------------
    # Plot 5: Node Count vs Chunk Size
    # -----------------------------------------------------------------
    try:
        node_counts = [r["rag_node_count"] for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(chunk_sizes, node_counts, "o-", color="#2ECC71", linewidth=2, markersize=8)
        ax.set_xlabel("Chunk Size")
        ax.set_ylabel("Number of Chunks/Nodes")
        ax.set_title("Chunk Count vs Chunk Size")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "node_count_vs_chunk_size.png"), dpi=150)
        plt.close(fig)
        print("[INFO] Saved plot: node_count_vs_chunk_size.png")
    except Exception as e:
        print(f"[WARN] Failed to generate node count plot: {e}")

    print(f"[INFO] All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    import json
    results_path = os.path.join("data", "results", "benchmark_metrics.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            data = json.load(f)
        generate_all_plots(data.get("sweep_results", []))
    else:
        print("[ERROR] No benchmark results found. Run evaluator.py first.")
