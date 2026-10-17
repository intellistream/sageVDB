import pytest

from sagevdb.persistence import PersistenceError, PersistentSageVDB
from test_durable_persistence import FakeDatabase, record


def index(path, namespace="knowledge"):
    return PersistentSageVDB(path, FakeDatabase, compatibility={"dim": 2},
                             source_namespace=namespace)


def embed(records):
    return [[1.0, 2.0] for _ in records]


def test_refresh_loads_other_writer_deletion_without_publishing(tmp_path):
    writer, reader = index(tmp_path), index(tmp_path)
    writer.sync([record("a", "old")], embed, source_revision=1)
    reader.refresh(minimum_source_revision=1)
    old_database = reader.database
    writer.sync([record("b", "new")], embed, source_revision=2)
    before = {p.relative_to(tmp_path): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in tmp_path.rglob("*") if p.is_file()}
    result = reader.refresh(minimum_source_revision=2)
    assert result.loaded and result.reason == "refreshed_current"
    assert reader.database is not old_database
    assert reader.source_revision == 2 and set(reader.vector_ids) == {"b"}
    after = {p.relative_to(tmp_path): (p.read_bytes(), p.stat().st_mtime_ns)
             for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_behind_and_corrupt_refresh_preserve_previous_live_handle(tmp_path):
    writer, reader = index(tmp_path), index(tmp_path)
    writer.sync([record("a", "old")], embed, source_revision=1)
    reader.refresh(minimum_source_revision=1)
    database, generation = reader.database, reader.generation
    with pytest.raises(PersistenceError, match="not_ready"):
        reader.refresh(minimum_source_revision=2)
    writer.sync([record("b", "new")], embed, source_revision=2)
    (tmp_path / f"generation-{writer.generation}" / "index.vectors").write_text("corrupt")
    with pytest.raises((PersistenceError, ValueError)):
        reader.refresh(minimum_source_revision=2)
    assert reader.database is database and reader.generation == generation
    assert reader.source_revision == 1 and set(reader.vector_ids) == {"a"}


def test_refresh_rejects_pointer_regression_and_namespace_change(tmp_path):
    writer = index(tmp_path)
    writer.sync([record("a", "old")], embed, source_revision=1)
    old = writer.generation
    writer.sync([], embed, source_revision=2)
    with pytest.raises(PersistenceError, match="namespace_conflict"):
        index(tmp_path, "other").refresh(minimum_source_revision=2)
    (tmp_path / "CURRENT").write_text(old)
    with pytest.raises(PersistenceError, match="regression"):
        writer.refresh(minimum_source_revision=1)
    assert writer.source_revision == 2 and writer.vector_ids == {}


@pytest.mark.parametrize("revision", [-1, True, 1.5, None])
def test_refresh_validates_revision(tmp_path, revision):
    with pytest.raises(ValueError, match="nonnegative_integer"):
        index(tmp_path).refresh(minimum_source_revision=revision)


def test_missing_projection_does_not_create_a_database(tmp_path):
    reader = index(tmp_path)
    with pytest.raises(PersistenceError, match="not_ready"):
        reader.refresh(minimum_source_revision=0)
    assert reader.generation is None and reader.source_revision is None
    assert not (tmp_path / "CURRENT").exists()


def test_native_reader_searches_new_generation_from_another_writer(tmp_path):
    import multiprocessing
    import numpy as np
    from sagevdb import DatabaseConfig, DistanceMetric, IndexType, SageVDB, SearchParams, search_numpy
    config = DatabaseConfig(2)
    config.index_type = IndexType.FLAT
    config.metric = DistanceMetric.L2
    def create():
        return PersistentSageVDB(tmp_path, lambda: SageVDB(config),
                                 compatibility={"dim": 2}, source_namespace="knowledge")
    writer, reader = create(), create()
    writer.sync([record("old", "old")], lambda _: [[1.0, 0.0]], source_revision=1)
    reader.refresh(minimum_source_revision=1)
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("native process refresh test requires fork")
    def publish():
        create().sync([record("new", "new")], lambda _: [[0.0, 1.0]], source_revision=2)
    process = multiprocessing.get_context("fork").Process(target=publish)
    process.start()
    process.join(10)
    try:
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
    reader.refresh(minimum_source_revision=2)
    hits = search_numpy(reader.database, np.array([0.0, 1.0], dtype=np.float32), SearchParams(k=1))
    assert len(hits) == 1 and hits[0].id == reader.vector_ids["new"]
    assert hits[0].score == pytest.approx(0.0)
    assert "old" not in reader.vector_ids


def test_load_failure_preserves_live_handle(tmp_path, monkeypatch):
    writer, reader = index(tmp_path), index(tmp_path)
    writer.sync([record("a", "old")], embed, source_revision=1)
    reader.refresh(minimum_source_revision=1)
    database = reader.database
    writer.sync([], embed, source_revision=2)
    def fail(*args):
        raise OSError("native load failed")
    monkeypatch.setattr(FakeDatabase, "load", fail)
    with pytest.raises(OSError, match="native load failed"):
        reader.refresh(minimum_source_revision=2)
    assert reader.database is database and reader.source_revision == 1
