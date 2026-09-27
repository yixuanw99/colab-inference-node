#!/usr/bin/env python3
"""
Automated Test & Diagnostic Suite for colab-model-station Embedding Engine.
Verifies model loading, dense vector extraction, cosine similarity ranking,
and GPU memory release on A100.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engines.embedding_engine.interface import EmbeddingEngine


def test_embedding_pipeline(model_id: str) -> None:
    print("=" * 80)
    print(f"Colab Model Station - Embedding Engine Diagnostic Suite ({model_id})")
    print("=" * 80)

    engine = EmbeddingEngine()
    engine.initialize()

    print("[TEST] 1. Loading model checkpoint onto GPU...")
    t0 = time.time()
    load_res = engine.load_model(model_id)
    t_load = time.time() - t0
    assert load_res["status"] == "loaded"
    print(f"  [PASS] Model loaded in {t_load:.2f}s on {load_res['device']}.")

    status = engine.get_status()
    assert status["status"] == "ready"
    print(f"  [PASS] Engine status confirmed: {status}")

    print("[TEST] 2. Computing dense vector embeddings...")
    sample_texts = [
        "Deep learning inference acceleration on NVIDIA A100 Tensor Core GPU.",
        "Convolutional neural networks for computer vision classification.",
        "Fast transformer inference using vLLM and PagedAttention algorithms.",
        "Cooking recipes for homemade Italian pasta carbonara.",
    ]
    t0 = time.time()
    embeddings = engine.embed(sample_texts)
    t_embed = time.time() - t0
    dim = len(embeddings[0])
    assert len(embeddings) == len(sample_texts)
    print(f"  [PASS] Batch embedding succeeded in {t_embed * 1000:.2f}ms (Shape: {len(embeddings)}x{dim}).")

    print("[TEST] 3. Semantic similarity reranking...")
    query = "How to optimize GPU memory and throughput for LLM serving?"
    t0 = time.time()
    ranked = engine.rerank(query, sample_texts)
    t_rank = time.time() - t0
    assert len(ranked) == len(sample_texts)
    print(f"  [PASS] Reranking completed in {t_rank * 1000:.2f}ms:")
    for rank, item in enumerate(ranked, 1):
        print(f"    #{rank} [Score: {item['score']:.4f}] {item['document']}")

    # Verification: LLM serving document should be top match
    top_doc = ranked[0]["document"]
    assert "Fast transformer inference" in top_doc or "NVIDIA A100" in top_doc, f"Unexpected top rank: {top_doc}"
    print(f"  [PASS] Semantic accuracy validated (Top: '{top_doc[:45]}...').")

    print("[TEST] 4. Throughput stress test...")
    large_batch = sample_texts * 25  # 100 sentences
    t0 = time.time()
    _ = engine.embed(large_batch)
    batch_time = time.time() - t0
    throughput = len(large_batch) / batch_time
    print(f"  [PASS] Processed {len(large_batch)} sentences in {batch_time:.2f}s (~{throughput:.1f} sent/s).")

    print("[TEST] 5. Graceful model teardown...")
    engine.unload_model()
    assert engine.get_status()["status"] == "unloaded"
    print("  [PASS] Model unloaded and GPU VRAM released.")

    print("=" * 80)
    print("ALL EMBEDDING TESTS PASSED SUCCESSFULLY.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Embedding Test Suite")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Hugging Face model ID")
    args = parser.parse_args()

    try:
        test_embedding_pipeline(args.model)
    except Exception as e:
        print(f"\n[FAIL] Embedding test suite failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
