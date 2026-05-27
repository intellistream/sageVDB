"""Python adapter to use sage-anns inside the sagevdb package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

try:
    from sage_anns import create_index, list_algorithms
except ImportError as exc:  # pragma: no cover - import gating
    raise ImportError(
        "sage-anns is not installed. Install with: pip install isage-anns"
    ) from exc

from ._sagevdb import MetadataStore, QueryResult


_ADAPTER_STATE_VERSION = 1


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
        self._index_params = dict(index_params)
        self._index = self._create_index_instance()
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
            self._metadata_store.set_metadata(vector_id, _normalize_metadata(metadata))
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
        return _to_query_results(
            distances, indices, self._metadata_store, include_metadata
        )

    def save(self, path: str) -> None:
        self._index.save(path)
        _save_adapter_state(
            path,
            {
                "version": _ADAPTER_STATE_VERSION,
                "dimension": self._dimension,
                "metric": self._metric,
                "algorithm": self._algorithm,
                "next_id": self._next_id,
                "metadata": _snapshot_metadata(self._metadata_store, self._next_id),
            },
        )

    def load(self, path: str) -> None:
        state = _load_adapter_state(path)
        if state is not None:
            _validate_adapter_state(state, self._dimension, self._metric, self._algorithm)

        next_index = self._create_index_instance()
        next_index.load(path)

        next_metadata_store = MetadataStore()
        if state is None:
            next_id = _infer_next_id(next_index)
        else:
            next_id = int(state["next_id"])
            _restore_metadata(next_metadata_store, state.get("metadata", {}))

        self._index = next_index
        self._metadata_store = next_metadata_store
        self._next_id = next_id

    def _create_index_instance(self) -> Any:
        return create_index(
            algorithm=self._algorithm,
            dimension=self._dimension,
            metric=self._metric,
            **self._index_params,
        )


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
    store.set_batch_metadata(ids_list, [_normalize_metadata(item) for item in metadata])


def _normalize_metadata(metadata: Mapping[Any, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in metadata.items()}


def _adapter_state_path(path: str) -> Path:
    base = Path(path)
    return base.with_name(f"{base.name}.sage_anns_adapter.json")


def _snapshot_metadata(store: MetadataStore, next_id: int) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for vector_id in range(max(0, int(next_id))):
        item = store.get_metadata(vector_id)
        if item:
            metadata[str(vector_id)] = dict(item)
    return metadata


def _save_adapter_state(path: str, state: dict[str, Any]) -> None:
    sidecar_path = _adapter_state_path(path)
    sidecar_path.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _load_adapter_state(path: str) -> dict[str, Any] | None:
    sidecar_path = _adapter_state_path(path)
    if not sidecar_path.exists():
        return None
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def _validate_adapter_state(
    state: dict[str, Any], dimension: int, metric: str, algorithm: str
) -> None:
    if int(state.get("version", -1)) != _ADAPTER_STATE_VERSION:
        raise ValueError("Unsupported sage-anns adapter state version")
    if int(state.get("dimension", -1)) != dimension:
        raise ValueError("Adapter state dimension mismatch")
    if state.get("metric") != metric:
        raise ValueError("Adapter state metric mismatch")
    if state.get("algorithm") != algorithm:
        raise ValueError("Adapter state algorithm mismatch")

    next_id = state.get("next_id")
    if not isinstance(next_id, int) or next_id < 0:
        raise ValueError("Adapter state next_id must be a non-negative integer")

    metadata = state.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Adapter state metadata must be a dictionary")


def _restore_metadata(store: MetadataStore, metadata: dict[str, Any]) -> None:
    for raw_id, item in metadata.items():
        vector_id = int(raw_id)
        if item:
            store.set_metadata(vector_id, _normalize_metadata(item))


def _infer_next_id(index: Any) -> int:
    candidates: list[int] = []

    for attr_name in ("num_vectors", "_num_vectors", "ntotal"):
        value = getattr(index, attr_name, None)
        if isinstance(value, int) and value >= 0:
            candidates.append(value)

    data = getattr(index, "_data", None)
    if isinstance(data, np.ndarray) and data.ndim >= 1:
        candidates.append(int(data.shape[0]))

    return max(candidates, default=0)


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
