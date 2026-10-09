#!/usr/bin/env python3
"""Tests for the sage-anns Python backend integration."""

import json
from pathlib import Path

import numpy as np
import pytest

from sagevdb import create_database, DatabaseConfig

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


class _PersistentDummyIndex(_DummyIndex):
    """Dummy index with simple data-only persistence."""

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            np.savez(
                handle,
                data=self._data,
                dimension=np.array([self.dimension], dtype=np.int64),
            )

    def load(self, path: str) -> None:
        with np.load(path) as payload:
            self.dimension = int(payload["dimension"][0])
            self._data = np.ascontiguousarray(payload["data"], dtype=np.float32)
            self._is_built = self._data is not None


class _RecordingDummyIndex(_DummyIndex):
    """Dummy index that records search kwargs for contract testing."""

    def __init__(self, dimension: int, metric: str = "l2", **kwargs):
        super().__init__(dimension=dimension, metric=metric, **kwargs)
        self.search_calls: list[dict[str, object]] = []

    def search(self, queries: np.ndarray, k: int = 10, **search_params):
        self.search_calls.append(dict(search_params))
        return super().search(queries, k=k, **search_params)


def _ensure_algorithm() -> str:
    from sage_anns import list_algorithms, register_algorithm

    algorithms = list_algorithms()
    name = "dummy_test_backend"
    if name in algorithms:
        return name
    try:
        register_algorithm(name, _DummyIndex)
    except ValueError:
        pass
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


def test_sage_anns_backend_stringifies_metadata_values():
    algorithm = _ensure_algorithm()
    dimension = 8
    db = create_database(
        dimension,
        backend="sage-anns",
        algorithm=algorithm,
        metric="l2",
    )

    vectors = np.random.rand(4, dimension).astype(np.float32)
    db.build_index(
        vectors,
        metadata=[{"row": index, "active": index % 2 == 0} for index in range(4)],
    )

    added_id = db.add(vectors[0], metadata={"row": 99, "active": False})

    built_result = db.search(vectors[1], k=2, include_metadata=True)[0]
    added_result = db.search(vectors[0], k=5, include_metadata=True)

    assert built_result.metadata == {"row": "1", "active": "False"}
    added_entry = next(item for item in added_result if item.id == added_id)
    assert added_entry.metadata == {"row": "99", "active": "False"}


def test_sage_anns_adapter_save_load_roundtrip_preserves_adapter_state(monkeypatch, tmp_path):
    import sagevdb.sage_anns as adapter_module

    def _fake_create_index(*, algorithm: str, dimension: int, metric: str = "l2", **kwargs):
        return _PersistentDummyIndex(dimension=dimension, metric=metric, **kwargs)

    monkeypatch.setattr(adapter_module, "create_index", _fake_create_index)

    dimension = 8
    db = create_database(dimension, backend="sage-anns", algorithm="persistent", metric="l2")

    vectors = np.random.rand(6, dimension).astype(np.float32)
    metadata = [{"tag": f"row-{index}"} for index in range(len(vectors))]
    db.build_index(vectors, metadata=metadata)

    query = vectors[0]
    before = db.search(query, k=2, include_metadata=True)
    assert before
    assert before[0].metadata

    save_path = tmp_path / "adapter-index.npz"
    db.save(str(save_path))

    loaded = create_database(dimension, backend="sage-anns", algorithm="persistent", metric="l2")
    loaded.load(str(save_path))

    after = loaded.search(query, k=2, include_metadata=True)
    assert [item.id for item in after] == [item.id for item in before]
    assert [item.metadata for item in after] == [item.metadata for item in before]

    new_vector = np.random.rand(dimension).astype(np.float32)
    new_id = loaded.add(new_vector, metadata={"tag": "new-row"})
    assert new_id == len(vectors)

    reloaded_result = loaded.search(new_vector, k=1, include_metadata=True)
    assert reloaded_result[0].id == new_id
    assert reloaded_result[0].metadata == {"tag": "new-row"}


def test_sage_anns_adapter_load_without_sidecar_infers_next_id(monkeypatch, tmp_path):
    import sagevdb.sage_anns as adapter_module

    def _fake_create_index(*, algorithm: str, dimension: int, metric: str = "l2", **kwargs):
        return _PersistentDummyIndex(dimension=dimension, metric=metric, **kwargs)

    monkeypatch.setattr(adapter_module, "create_index", _fake_create_index)

    dimension = 8
    db = create_database(dimension, backend="sage-anns", algorithm="persistent", metric="l2")

    vectors = np.random.rand(4, dimension).astype(np.float32)
    db.build_index(vectors)

    save_path = tmp_path / "legacy-index.npz"
    db._index.save(str(save_path))

    loaded = create_database(dimension, backend="sage-anns", algorithm="persistent", metric="l2")
    loaded.load(str(save_path))

    new_id = loaded.add(np.random.rand(dimension).astype(np.float32))
    assert new_id == len(vectors)


def test_sage_anns_adapter_load_invalid_sidecar_keeps_existing_state(monkeypatch, tmp_path):
    import sagevdb.sage_anns as adapter_module

    def _fake_create_index(*, algorithm: str, dimension: int, metric: str = "l2", **kwargs):
        return _PersistentDummyIndex(dimension=dimension, metric=metric, **kwargs)

    monkeypatch.setattr(adapter_module, "create_index", _fake_create_index)

    dimension = 8
    db = create_database(dimension, backend="sage-anns", algorithm="persistent", metric="l2")

    current_vectors = np.random.rand(3, dimension).astype(np.float32)
    db.build_index(current_vectors, metadata=[{"tag": f"current-{i}"} for i in range(3)])
    current_query = current_vectors[0]
    before = db.search(current_query, k=1, include_metadata=True)
    assert before[0].metadata == {"tag": "current-0"}

    donor = create_database(dimension, backend="sage-anns", algorithm="persistent", metric="l2")
    donor_vectors = np.random.rand(5, dimension).astype(np.float32)
    donor.build_index(donor_vectors, metadata=[{"tag": f"donor-{i}"} for i in range(5)])

    save_path = tmp_path / "corrupt-adapter-index.npz"
    donor.save(str(save_path))

    sidecar_path = adapter_module._adapter_state_path(str(save_path))
    state = json.loads(sidecar_path.read_text(encoding="utf-8"))
    state["dimension"] = dimension + 1
    sidecar_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="dimension mismatch"):
        db.load(str(save_path))

    after = db.search(current_query, k=1, include_metadata=True)
    assert [item.id for item in after] == [item.id for item in before]
    assert [item.metadata for item in after] == [item.metadata for item in before]

    new_id = db.add(np.random.rand(dimension).astype(np.float32), metadata={"tag": "current-new"})
    assert new_id == len(current_vectors)


def test_sage_anns_database_config_does_not_auto_forward_anns_query_params(monkeypatch):
    import sagevdb.sage_anns as adapter_module

    created: dict[str, _RecordingDummyIndex] = {}

    def _fake_create_index(*, algorithm: str, dimension: int, metric: str = "l2", **kwargs):
        index = _RecordingDummyIndex(dimension=dimension, metric=metric, **kwargs)
        created["index"] = index
        return index

    monkeypatch.setattr(adapter_module, "create_index", _fake_create_index)

    dimension = 8
    config = DatabaseConfig(dimension)
    config.anns_algorithm = "recording"
    config.anns_query_params = {"probe_budget": "7"}

    db = create_database(config, backend="sage-anns")
    vectors = np.random.rand(5, dimension).astype(np.float32)
    db.build_index(vectors)

    query = vectors[0]
    db.search(query, k=2)
    assert created["index"].search_calls[-1] == {}

    db.search(query, k=2, probe_budget="7")
    assert created["index"].search_calls[-1] == {"probe_budget": "7"}


def test_sage_anns_example_does_not_report_dimension_as_vector_count() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "sage_anns_integration_example.py"
    source = example_path.read_text(encoding="utf-8")

    assert 'Total vectors in db2: {db2.dimension}' not in source


def test_sage_vdb_backend_rejects_invalid_index_type(monkeypatch) -> None:
    import sagevdb
    from sagevdb._vdb_backend import SageVDBBackend

    monkeypatch.setattr(sagevdb, "string_to_index_type", lambda value: (_ for _ in ()).throw(ValueError("bad index")))

    with pytest.raises(ValueError, match="Unsupported index_type: BAD"):
        SageVDBBackend({"dim": 8, "index_type": "BAD"})


def test_sage_vdb_backend_rejects_invalid_metric(monkeypatch) -> None:
    import sagevdb
    from sagevdb._vdb_backend import SageVDBBackend

    original_index_type = sagevdb.string_to_index_type
    monkeypatch.setattr(sagevdb, "string_to_index_type", original_index_type)
    monkeypatch.setattr(sagevdb, "string_to_distance_metric", lambda value: (_ for _ in ()).throw(ValueError("bad metric")))

    with pytest.raises(ValueError, match="Unsupported metric: BAD"):
        SageVDBBackend({"dim": 8, "index_type": "FLAT", "metric": "BAD"})
