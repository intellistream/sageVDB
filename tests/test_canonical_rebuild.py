import pytest

from sagevdb.document_catalog import VersionedDocumentCatalog
from sagevdb.persistence import PersistenceError, PersistentRecord, PersistentSageVDB, stable_content_hash
from test_document_catalog import AtomicMemory
from test_durable_persistence import FakeDatabase


def setup(path):
    catalog = VersionedDocumentCatalog(AtomicMemory(), namespace="knowledge")
    index = PersistentSageVDB(path, FakeDatabase, compatibility={"dim": 2},
                              source_namespace="knowledge")
    return catalog, index


def build(key, document):
    return PersistentRecord(key, stable_content_hash(document["text"]), {"text": document["text"]})


def embed(records):
    return [[1.0, 2.0] for _ in records]


@pytest.mark.parametrize("damage", ["vectors", "pointer", "manifest", "pointer_encoding"])
def test_rebuild_uses_current_canonical_documents_and_tombstones(tmp_path, damage):
    catalog, index = setup(tmp_path)
    catalog.commit("one", {"a": {"text": "old"}}, expected_revision=0)
    index.rebuild_from_catalog(catalog, build, embed)
    generation = index.generation
    if damage == "pointer":
        (tmp_path / "CURRENT").unlink()
    elif damage == "pointer_encoding":
        (tmp_path / "CURRENT").write_bytes(b"\xffinvalid")
    else:
        filename = "index.vectors" if damage == "vectors" else "manifest.json"
        (tmp_path / f"generation-{generation}" / filename).write_text("corrupt")
    catalog.commit("two", {"a": None, "b": {"text": "new"}}, expected_revision=1)
    _, replacement = setup(tmp_path)
    with pytest.raises(PersistenceError):
        replacement.open()
    result = replacement.rebuild_from_catalog(catalog, build, embed)
    assert result.committed
    assert replacement.source_revision == 2
    assert set(replacement.vector_ids) == {"b"}
    _, reopened = setup(tmp_path)
    assert reopened.open().loaded
    assert reopened.source_revision == 2
    assert set(reopened.vector_ids) == {"b"}


def test_rebuild_failure_does_not_publish_and_catalog_change_is_retryable(tmp_path, monkeypatch):
    catalog, index = setup(tmp_path)
    catalog.commit("one", {"a": {"text": "old"}}, expected_revision=0)
    index.rebuild_from_catalog(catalog, build, embed)
    generation = index.generation
    catalog.commit("two", {"a": None}, expected_revision=1)
    with monkeypatch.context() as patch:
        def fail():
            raise OSError("save failed")
        patch.setattr(index, "_commit_generation", fail)
        with pytest.raises(OSError, match="save failed"):
            index.rebuild_from_catalog(catalog, build, embed)
    assert index.generation == generation
    assert index.source_revision == 1
    assert (tmp_path / "CURRENT").read_text().strip() == generation
    index.rebuild_from_catalog(catalog, build, embed)
    assert index.source_revision == 2 and index.vector_ids == {}
    catalog.commit("three", {"b": {"text": "three"}}, expected_revision=2)
    def changed(records):
        catalog.commit("four", {"b": {"text": "four"}}, expected_revision=3)
        return embed(records)
    with pytest.raises(PersistenceError, match="changed_during_rebuild"):
        index.rebuild_from_catalog(catalog, build, changed)
    assert index.source_revision == 2
    index.rebuild_from_catalog(catalog, build, embed)
    assert index.source_revision == 4


def test_namespace_regression_and_equal_revision_identity_are_rejected(tmp_path):
    catalog, index = setup(tmp_path)
    catalog.commit("one", {"a": {"text": "one"}}, expected_revision=0)
    index.rebuild_from_catalog(catalog, build, embed)
    wrong = VersionedDocumentCatalog(AtomicMemory(), namespace="other")
    with pytest.raises(PersistenceError, match="namespace_conflict"):
        index.rebuild_from_catalog(wrong, build, embed)
    with pytest.raises(TypeError, match="requires_versioned"):
        index.rebuild_from_catalog(catalog.snapshot(), build, embed)
    with pytest.raises(PersistenceError, match="identity_conflict"):
        index.rebuild_from_catalog(catalog, lambda key, doc: build(key, {"text": "changed"}), embed)
    catalog.commit("two", {"a": None}, expected_revision=1)
    index.rebuild_from_catalog(catalog, build, embed)
    old, _ = setup(tmp_path)
    old.commit("one", {"a": {"text": "one"}}, expected_revision=0)
    with pytest.raises(PersistenceError, match="older_than_index"):
        index.rebuild_from_catalog(old, build, embed)


def test_post_commit_failure_restores_committed_generation(tmp_path, monkeypatch):
    catalog, index = setup(tmp_path)
    catalog.commit("one", {"a": {"text": "one"}}, expected_revision=0)
    def fail():
        raise OSError("ack lost")
    with monkeypatch.context() as patch:
        patch.setattr(index, "_prune_generations", fail)
        with pytest.raises(OSError, match="ack lost"):
            index.rebuild_from_catalog(catalog, build, embed)
    assert index.source_revision == 1
    assert set(index.vector_ids) == {"a"}
    _, reopened = setup(tmp_path)
    assert reopened.open().loaded
    assert reopened.source_revision == 1


def test_native_rebuild_repairs_search_from_current_body(tmp_path):
    import numpy as np
    from sagevdb import DatabaseConfig, DistanceMetric, IndexType, SageVDB, SearchParams, search_numpy
    config = DatabaseConfig(2)
    config.index_type = IndexType.FLAT
    config.metric = DistanceMetric.L2
    def create():
        return PersistentSageVDB(tmp_path, lambda: SageVDB(config), compatibility={"dim": 2},
                                 source_namespace="knowledge")
    catalog, _ = setup(tmp_path)
    catalog.commit("one", {"old": {"text": "old"}}, expected_revision=0)
    index = create()
    index.rebuild_from_catalog(catalog, build, embed)
    (tmp_path / f"generation-{index.generation}" / "index.vectors").write_text("corrupt")
    catalog.commit("two", {"old": None, "new": {"text": "new"}}, expected_revision=1)
    recovered = create()
    with pytest.raises(PersistenceError):
        recovered.open()
    recovered.rebuild_from_catalog(catalog, build, embed)
    reopened = create()
    assert reopened.open().loaded
    hits = search_numpy(reopened.database, np.array([1.0, 2.0], dtype=np.float32), SearchParams(k=1))
    assert len(hits) == 1 and hits[0].id == reopened.vector_ids["new"]
    assert "old" not in reopened.vector_ids
    assert reopened.source_revision == 2


@pytest.mark.parametrize("after_publication", [False, True])
def test_process_exit_before_and_after_publication_recovers(tmp_path, after_publication):
    import multiprocessing
    import os
    module = pytest.importorskip("sage.runtime.flownet.runtime.operator_runtime.stateful")
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("process test requires fork")
    path = str(tmp_path / "canonical.db")
    root = tmp_path / "index"
    state = module.StatefulProcessRuntime(backend="sqlite", sqlite_path=path)
    catalog = VersionedDocumentCatalog(state, namespace="knowledge")
    _, index = setup(root)
    catalog.commit("one", {"a": {"text": "old"}}, expected_revision=0)
    index.rebuild_from_catalog(catalog, build, embed)
    catalog.commit("two", {"a": None, "b": {"text": "new"}}, expected_revision=1)
    state.close()

    def child():
        child_state = module.StatefulProcessRuntime(backend="sqlite", sqlite_path=path)
        child_catalog = VersionedDocumentCatalog(child_state, namespace="knowledge")
        _, child_index = setup(root)
        def stop(*args):
            os._exit(18 if after_publication else 17)
        if after_publication:
            child_index._prune_generations = stop
        else:
            child_index._replace_current = stop
        child_index.rebuild_from_catalog(child_catalog, build, embed)

    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    process.join(10)
    try:
        assert process.exitcode == (18 if after_publication else 17)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
    state = module.StatefulProcessRuntime(backend="sqlite", sqlite_path=path)
    try:
        catalog = VersionedDocumentCatalog(state, namespace="knowledge")
        _, recovered = setup(root)
        assert recovered.open().loaded
        assert recovered.source_revision == (2 if after_publication else 1)
        recovered.rebuild_from_catalog(catalog, build, embed)
        assert recovered.source_revision == 2
        assert set(recovered.vector_ids) == {"b"}
    finally:
        state.close()
