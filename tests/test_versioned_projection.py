import pytest

from sagevdb.persistence import PersistenceError, PersistentSageVDB
from test_durable_persistence import FakeDatabase, record


def open_index(path, namespace="knowledge"):
    result = PersistentSageVDB(path, FakeDatabase, compatibility={"dim": 2},
                               source_namespace=namespace)
    result.open()
    return result


def embed(records):
    return [[1.0, 2.0] for _ in records]


def no_embed(_):
    pytest.fail("replay, stale or metadata-only snapshot must not embed")


def test_stale_replay_conflict_and_higher_unchanged_revision(tmp_path):
    index = open_index(tmp_path)
    index.sync([record("a", "one")], embed, source_revision=1)
    first = index.generation
    result = index.sync([record("a", "one")], no_embed, source_revision=2)
    assert result.committed and result.embedded == 0
    assert index.source_revision == 2 and index.generation != first
    assert not index.sync([record("a", "one")], no_embed, source_revision=2).committed
    with pytest.raises(PersistenceError, match="identity_conflict"):
        index.sync([record("a", "two")], no_embed, source_revision=2)
    with pytest.raises(PersistenceError, match="stale_source_revision"):
        index.sync([], no_embed, source_revision=1)
    reopened = open_index(tmp_path)
    assert reopened.source_revision == 2
    assert set(reopened.vector_ids) == {"a"}


def test_independent_instances_reload_fence_and_tombstones(tmp_path):
    older = open_index(tmp_path)
    newer = open_index(tmp_path)
    older.sync([record("a", "one")], embed, source_revision=1)
    newer.sync([], embed, source_revision=3)
    with pytest.raises(PersistenceError, match="stale_source_revision"):
        older.sync([record("a", "resurrected")], no_embed, source_revision=2)
    assert older.source_revision == 3
    assert older.vector_ids == {}


def test_failures_restore_matching_revision_and_ack_loss_keeps_commit(tmp_path, monkeypatch):
    index = open_index(tmp_path)
    index.sync([record("a", "one")], embed, source_revision=1)
    with monkeypatch.context() as patch:
        def fail():
            raise OSError("save failed")
        patch.setattr(index, "_commit_generation", fail)
        with pytest.raises(OSError, match="save failed"):
            index.sync([record("b", "two")], embed, source_revision=2)
    assert index.source_revision == 1
    assert set(index.vector_ids) == {"a"}
    with monkeypatch.context() as patch:
        def lost_ack():
            raise OSError("cleanup failed after commit")
        patch.setattr(index, "_prune_generations", lost_ack)
        with pytest.raises(OSError, match="after commit"):
            index.sync([record("b", "two")], embed, source_revision=2)
    assert index.source_revision == 2
    assert set(index.vector_ids) == {"b"}
    assert not index.sync([record("b", "two")], no_embed, source_revision=2).committed


@pytest.mark.parametrize("damage", ["corrupt", "missing_pointer"])
def test_damaged_versioned_current_never_recovers_older_fence(tmp_path, damage):
    index = open_index(tmp_path)
    index.sync([record("a", "old")], embed, source_revision=1)
    index.sync([record("b", "new")], embed, source_revision=2)
    if damage == "corrupt":
        (tmp_path / f"generation-{index.generation}" / "index.vectors").write_text("corrupt")
        with pytest.raises(PersistenceError):
            index.sync([record("b", "new")], no_embed, source_revision=2)
    else:
        (tmp_path / "CURRENT").unlink()
    with pytest.raises(PersistenceError, match="requires_canonical_rebuild"):
        open_index(tmp_path)


def test_cannot_drop_or_change_namespace(tmp_path):
    index = open_index(tmp_path)
    index.sync([], embed, source_revision=0)
    for namespace in (None, "other"):
        with pytest.raises(PersistenceError, match="source_namespace_conflict"):
            open_index(tmp_path, namespace)
    with pytest.raises(ValueError, match="source_revision"):
        index.sync([], embed)
    with pytest.raises(ValueError, match="source_revision"):
        index.sync([], embed, source_revision=True)


def test_embedding_callback_cannot_change_saved_metadata(tmp_path):
    index = open_index(tmp_path)
    item = record("a", "one", title="original")
    def mutate(records):
        records[0].metadata["title"] = "changed"
        return embed(records)
    index.sync([item], mutate, source_revision=1)
    assert not index.sync([item], no_embed, source_revision=1).committed


def test_process_writers_cannot_finish_with_older_revision(tmp_path):
    import multiprocessing
    import os
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("process test requires fork")
    index = open_index(tmp_path)
    index.sync([], embed, source_revision=1)

    def write(revision):
        child = open_index(tmp_path)
        try:
            child.sync([record(str(revision), str(revision))], embed, source_revision=revision)
        except PersistenceError as exc:
            os._exit(23 if str(exc) == "stale_source_revision" else 24)
        os._exit(0)

    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=write, args=(revision,)) for revision in (2, 3)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
        assert processes[0].exitcode in (0, 23)
        assert processes[1].exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    reopened = open_index(tmp_path)
    assert reopened.source_revision == 3
    assert set(reopened.vector_ids) == {"3"}


def test_native_index_reopen_search_respects_latest_projection(tmp_path):
    import numpy as np
    from sagevdb import DatabaseConfig, DistanceMetric, IndexType, SageVDB, SearchParams, search_numpy
    config = DatabaseConfig(2)
    config.index_type = IndexType.FLAT
    config.metric = DistanceMetric.L2
    def create():
        return PersistentSageVDB(tmp_path, lambda: SageVDB(config), compatibility={"dim": 2},
                                 source_namespace="knowledge")
    index = create()
    index.sync([record("old", "old")], lambda _: [[1.0, 0.0]], source_revision=1)
    index.sync([record("new", "new")], lambda _: [[0.0, 1.0]], source_revision=2)
    reopened = create()
    assert reopened.open().loaded
    with pytest.raises(PersistenceError, match="stale_source_revision"):
        reopened.sync([record("old", "old")], no_embed, source_revision=1)
    hits = search_numpy(reopened.database, np.array([0.0, 1.0], dtype=np.float32), SearchParams(k=1))
    assert len(hits) == 1
    assert hits[0].id == reopened.vector_ids["new"]
    assert reopened.source_revision == 2
    assert "old" not in reopened.vector_ids
