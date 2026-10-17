import pytest

from sagevdb.persistence import PersistentSageVDB, PersistenceError
from sagevdb.document_catalog import VersionedDocumentCatalog
from test_document_catalog import AtomicMemory
from test_durable_persistence import FakeDatabase, record


class SnapshotDatabase(FakeDatabase):
    def build_snapshot(self, vectors, metadata):
        self.items = {i: {"vector": list(vector), "metadata": dict(meta)}
                      for i, (vector, meta) in enumerate(zip(vectors, metadata))}
        self.next_id = len(vectors)
        return list(self.items)

    def add(self, *args):
        pytest.fail("snapshot lifecycle must not call mutable add")

    remove = set_metadata = build_index = add


def create(path, factory=SnapshotDatabase):
    return PersistentSageVDB(path, factory, compatibility={"dim": 2},
                             source_namespace="knowledge", build_mode="snapshot")


def embed(records):
    return [[1.0, 2.0] for _ in records]


def forbidden(_):
    pytest.fail("unchanged revision must not embed")


def test_snapshot_updates_deletions_empty_and_replay_share_version_fence(tmp_path):
    first = create(tmp_path)
    initial = first.sync([record("a", "one"), record("b", "two")], embed, source_revision=1)
    assert initial.embedded == 2
    assert not first.sync([record("a", "one"), record("b", "two")], forbidden,
                          source_revision=1).committed
    advanced = first.sync([record("a", "one"), record("b", "two")], forbidden, source_revision=2)
    assert advanced.committed and advanced.embedded == 0
    changed = first.sync([record("b", "changed")], embed, source_revision=3)
    assert (changed.updated, changed.deleted, changed.embedded) == (1, 1, 1)
    reader = create(tmp_path)
    reader.refresh(minimum_source_revision=3)
    assert set(reader.vector_ids) == {"b"}
    with pytest.raises(PersistenceError, match="stale_source_revision"):
        reader.sync([record("a", "resurrect")], forbidden, source_revision=2)
    empty = first.sync([], forbidden, source_revision=4)
    assert empty.deleted == 1 and empty.embedded == 0
    reader.refresh(minimum_source_revision=4)
    assert reader.vector_ids == {} and reader.database.items == {}


@pytest.mark.parametrize("failure", ["embed", "build", "save", "after_commit"])
def test_snapshot_failures_preserve_committed_generation(tmp_path, monkeypatch, failure):
    index = create(tmp_path)
    index.sync([record("a", "one")], embed, source_revision=1)
    before = index.generation
    def fail(*args):
        raise OSError("injected failure")
    provider = fail if failure == "embed" else embed
    if failure == "build":
        monkeypatch.setattr(SnapshotDatabase, "build_snapshot", fail)
    if failure == "save":
        monkeypatch.setattr(SnapshotDatabase, "save", fail)
    if failure == "after_commit":
        monkeypatch.setattr(index, "_prune_generations", fail)
    with pytest.raises(OSError):
        index.sync([record("b", "two")], provider, source_revision=2)
    assert index.source_revision == (2 if failure == "after_commit" else 1)
    assert set(index.vector_ids) == ({"b"} if failure == "after_commit" else {"a"})
    assert (index.generation != before) == (failure == "after_commit")


def test_snapshot_canonical_recovery_and_build_mode_cannot_be_reinterpreted(tmp_path):
    catalog = VersionedDocumentCatalog(AtomicMemory(), namespace="knowledge")
    catalog.commit("one", {"old": {"text": "old"}}, expected_revision=0)
    index = create(tmp_path)
    def build(key, document):
        return record(key, document["text"])
    index.rebuild_from_catalog(catalog, build, embed)
    (tmp_path / f"generation-{index.generation}" / "index.vectors").write_text("corrupt")
    catalog.commit("two", {"old": None, "new": {"text": "new"}}, expected_revision=1)
    index.rebuild_from_catalog(catalog, build, embed)
    assert index.source_revision == 2 and set(index.vector_ids) == {"new"}
    wrong = PersistentSageVDB(tmp_path, FakeDatabase, compatibility={"dim": 2},
                              source_namespace="knowledge")
    with pytest.raises(PersistenceError):
        wrong.open()
    with pytest.raises(PersistenceError, match="build mode"):
        wrong.rebuild_from_catalog(catalog, build, embed)


def test_real_anns_snapshot_restart_search_and_empty_deletion(tmp_path):
    pytest.importorskip("sage_anns")
    from sagevdb.sage_anns import SageANNSVectorStore
    def factory():
        return SageANNSVectorStore(2, "faiss_hnsw", metric="l2")
    writer = create(tmp_path, factory)
    writer.sync([record("a", "one")], embed, source_revision=1)
    reader = create(tmp_path, factory)
    reader.refresh(minimum_source_revision=1)
    hits = reader.database.search([1.0, 2.0], k=1)
    assert hits and hits[0].metadata["__sagevdb_record_key__"] == "a"
    writer.sync([], forbidden, source_revision=2)
    reader.refresh(minimum_source_revision=2)
    assert reader.database.search([1.0, 2.0]) == []
    writer.sync([record("b", "new")], embed, source_revision=3)
    reader.refresh(minimum_source_revision=3)
    assert reader.database.search([1.0, 2.0], k=1)[0].metadata["__sagevdb_record_key__"] == "b"
