"""Python adapter to use sage-anns inside the sagevdb package."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sage_anns import create_index, list_algorithms
except ImportError as exc:  # pragma: no cover - import gating
    raise ImportError(
        "sage-anns is not installed. Install with: pip install isage-anns"
    ) from exc

from ._sagevdb import MetadataStore, QueryResult


def list_sage_anns_algorithms() -> List[str]:
    """Return available sage-anns algorithm names."""
    return list_algorithms()


class SageANNSVectorStore:
    """Lightweight Python wrapper that uses sage-anns for ANN search.

    This class mirrors the core SageVDB usage pattern but is implemented in Python
    and backed by the sage-anns package. It is intended as an easy integration
    path without rebuilding the C++ core.
    """

    def __init__(
        self,
        dimension: int,
        algorithm: str,
        metric: str = "l2",
        **index_params,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = int(dimension)
        self._algorithm = algorithm
        self._metric = metric
        self._index = create_index(
            algorithm=algorithm,
            dimension=self._dimension,
            metric=metric,
            **index_params,
        )
        self._metadata_store = MetadataStore()
        self._next_id = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def build_index(
        self,
        vectors: np.ndarray,
        metadata: Optional[Sequence[dict]] = None,
    ) -> None:
        data = _ensure_2d_float32(vectors, self._dimension, "vectors")
        self._index.build(data)
        self._metadata_store.clear()
        self._next_id = int(data.shape[0])
        if metadata:
            _set_metadata_batch(self._metadata_store, range(self._next_id), metadata)

    def add(
        self,
        vector: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> int:
        data = _ensure_2d_float32(vector, self._dimension, "vector")
        vector_id = self._next_id
        self._next_id += 1
        self._index.add(data, ids=np.array([vector_id], dtype=np.int64))
        if metadata:
            self._metadata_store.set_metadata(vector_id, metadata)
        return vector_id

    def add_batch(
        self,
        vectors: np.ndarray,
        metadata: Optional[Sequence[dict]] = None,
    ) -> List[int]:
        data = _ensure_2d_float32(vectors, self._dimension, "vectors")
        count = int(data.shape[0])
        ids = np.arange(self._next_id, self._next_id + count, dtype=np.int64)
        self._next_id += count
        self._index.add(data, ids=ids)
        if metadata:
            _set_metadata_batch(self._metadata_store, ids.tolist(), metadata)
        return ids.tolist()

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        include_metadata: bool = True,
        **search_params,
    ) -> List[QueryResult]:
        queries = _ensure_2d_float32(query, self._dimension, "query")
        distances, indices = self._index.search(queries, k=k, **search_params)
        return _to_query_results(
            distances,
            indices,
            self._metadata_store,
            include_metadata,
        )[0]

    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        include_metadata: bool = True,
        **search_params,
    ) -> List[List[QueryResult]]:
        batch = _ensure_2d_float32(queries, self._dimension, "queries")
        distances, indices = self._index.search(batch, k=k, **search_params)
        return _to_query_results(distances, indices, self._metadata_store, include_metadata)

    def save(self, path: str) -> None:
        self._index.save(path)

    def load(self, path: str) -> None:
        self._index.load(path)


def _ensure_2d_float32(data: np.ndarray, dimension: int, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D array, got shape {arr.shape}")
    if arr.shape[1] != dimension:
        raise ValueError(
            f"{name} dimension mismatch: expected {dimension}, got {arr.shape[1]}"
        )
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _set_metadata_batch(
    store: MetadataStore,
    ids: Iterable[int],
    metadata: Sequence[dict],
) -> None:
    ids_list = list(ids)
    if len(ids_list) != len(metadata):
        raise ValueError("metadata size must match number of vectors")
    store.set_batch_metadata(ids_list, list(metadata))


def _to_query_results(
    distances: np.ndarray,
    indices: np.ndarray,
    store: MetadataStore,
    include_metadata: bool,
) -> List[List[QueryResult]]:
    results: List[List[QueryResult]] = []
    for row_dist, row_idx in zip(distances, indices):
        row: List[QueryResult] = []
        for dist, idx in zip(row_dist, row_idx):
            if idx < 0:
                continue
            if include_metadata:
                meta = store.get_metadata(int(idx))
                metadata = meta if meta is not None else {}
            else:
                metadata = {}
            row.append(QueryResult(int(idx), float(dist), metadata))
        results.append(row)
    return results
