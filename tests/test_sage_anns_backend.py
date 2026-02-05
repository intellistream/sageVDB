#!/usr/bin/env python3
"""Tests for the sage-anns Python backend integration."""

import numpy as np
import pytest

from sagevdb import create_database, DatabaseConfig, list_sage_anns_algorithms

sage_anns = pytest.importorskip("sage_anns")


class _DummyIndex:
    """Minimal ANNS index for testing when no algorithms are registered."""

    def __init__(self, dimension: int, metric: str = "l2", **kwargs):
        self.dimension = dimension
        self.metric = metric
        self._data = None
        self._is_built = False

    def build(self, data: np.ndarray) -> None:
        if data.ndim != 2 or data.shape[1] != self.dimension:
            raise ValueError("Data shape mismatch")
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        self._is_built = True

    def add(self, vectors: np.ndarray, ids: np.ndarray = None) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Vectors shape mismatch")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self._data is None:
            self._data = vectors
        else:
            self._data = np.vstack([self._data, vectors])
        self._is_built = True

    def search(self, queries: np.ndarray, k: int = 10, **search_params):
        if not self._is_built or self._data is None:
            raise RuntimeError("Index not built")
        if queries.ndim != 2 or queries.shape[1] != self.dimension:
            raise ValueError("Queries shape mismatch")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        diffs = queries[:, None, :] - self._data[None, :, :]
        distances = np.sum(diffs * diffs, axis=2)
        indices = np.argsort(distances, axis=1)[:, :k]
        sorted_distances = np.take_along_axis(distances, indices, axis=1)
        return sorted_distances, indices


def _ensure_algorithm() -> str:
    from sage_anns import list_algorithms, register_algorithm
    algorithms = list_algorithms()
    if algorithms:
        return algorithms[0]
    name = "dummy"
    register_algorithm(name, _DummyIndex)
    return name


def test_sage_anns_backend_basic():
    algorithm = _ensure_algorithm()
    dimension = 16
    db = create_database(
        dimension,
        backend="sage-anns",
        algorithm=algorithm,
        metric="l2",
    )

    vectors = np.random.rand(20, dimension).astype(np.float32)
    metadata = [{"tag": str(i)} for i in range(20)]
    db.build_index(vectors, metadata=metadata)

    query = vectors[:2]
    results = db.batch_search(query, k=3, include_metadata=True)

    assert len(results) == 2
    for row in results:
        assert len(row) <= 3
        for item in row:
            assert 0 <= item.id < 20
            assert isinstance(item.metadata, dict)


def test_sage_anns_backend_config_path():
    algorithm = _ensure_algorithm()
    dimension = 8
    config = DatabaseConfig(dimension)
    config.anns_algorithm = algorithm

    db = create_database(config, backend="sage-anns")
    vectors = np.random.rand(10, dimension).astype(np.float32)
    db.build_index(vectors)

    query = vectors[:1]
    results = db.search(query, k=2)

    assert len(results) <= 2
    for item in results:
        assert 0 <= item.id < 10
