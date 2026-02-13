#!/usr/bin/env python3
"""Performance-oriented integration tests for sage-anns backend."""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from sagevdb import create_database, list_sage_anns_algorithms

sage_anns = pytest.importorskip("sage_anns")

PREFERRED_ALGOS = [
    "faiss_hnsw",
    "faiss",
    "vsag_hnsw",
    "gti",
    "plsh",
    "candy_flat",
    "candy_dpg",
    "candy_nndescent",
    "candy_lshapg",
    "candy_onlinepq",
]
MAX_ALGOS = 3


def _select_algorithms() -> List[str]:
    available = list_sage_anns_algorithms() or []
    preferred = [algo for algo in PREFERRED_ALGOS if algo in available]
    if not preferred:
        pytest.skip(
            "No preferred sage-anns algorithms are available. "
            "Install isage-anns with algorithm backends enabled.",
            allow_module_level=True,
        )
    return preferred[:MAX_ALGOS]


@pytest.mark.integration
@pytest.mark.perf
@pytest.mark.slow
@pytest.mark.parametrize("algorithm", _select_algorithms())
def test_sage_anns_backend_perf(algorithm: str) -> None:
    dimension = 64
    num_vectors = 1000
    num_queries = 25
    k = 10

    rng = np.random.default_rng(42)
    vectors = rng.random((num_vectors, dimension), dtype=np.float32)
    queries = rng.random((num_queries, dimension), dtype=np.float32)

    db = create_database(
        dimension,
        backend="sage-anns",
        algorithm=algorithm,
        metric="l2",
    )

    build_start = time.perf_counter()
    db.build_index(vectors)
    build_time = time.perf_counter() - build_start

    search_start = time.perf_counter()
    results = db.batch_search(queries, k=k, include_metadata=False)
    search_time = time.perf_counter() - search_start

    assert len(results) == num_queries
    for row in results:
        assert len(row) <= k
        for item in row:
            assert 0 <= item.id < num_vectors

    avg_query_time = search_time / max(num_queries, 1)

    # Loose bounds to avoid CI noise, still catches regressions.
    assert build_time < 30.0
    assert avg_query_time < 0.5

    print(
        f"[perf] {algorithm}: build={build_time:.3f}s, "
        f"search={search_time:.3f}s, avg_query={avg_query_time:.4f}s"
    )
