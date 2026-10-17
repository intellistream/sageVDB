import pytest

from sagevdb.persistence import PersistentSageVDB, PersistenceError
from sagevdb.document_catalog import VersionedDocumentCatalog
from test_document_catalog import AtomicMemory
from test_durable_persistence import FakeDatabase, record


class RecordDatabase(FakeDatabase):
    def build_records(self, payloads, metadata):
        self.items = {i: {"payload": payload, "metadata": meta}
                      for i, (payload, meta) in enumerate(zip(payloads, metadata))}
        self.next_id = len(payloads)
        return list(self.items)

    def add(self, *args):
        pytest.fail("record projection cannot use vector mutation")

    remove = set_metadata = build_index = add


def create(path):
    return PersistentSageVDB(path, RecordDatabase, compatibility={"kind": "text"},
                             source_namespace="knowledge", build_mode="records")


def payloads(records):
    return [{"text": item.key, "nested": {"tags": ["research"]}} for item in records]


def forbidden(_):
    pytest.fail("unchanged record projection should not rebuild")


def test_record_projection_restarts_refreshes_and_fences_without_fake_vectors(tmp_path):
    index = create(tmp_path)
    result = index.sync([record("a", "one")], payloads, source_revision=1)
    assert (result.projected, result.embedded) == (1, 0)
    assert index.database.items[0]["payload"]["text"] == "a"
    assert not index.sync([record("a", "one")], forbidden, source_revision=1).committed
    advanced = index.sync([record("a", "one")], forbidden, source_revision=2)
    assert advanced.committed and advanced.projected == advanced.embedded == 0
    reader = create(tmp_path)
    assert reader.open().loaded
    index.sync([record("b", "two")], payloads, source_revision=3)
    reader.refresh(minimum_source_revision=3)
    assert reader.database.items[0]["payload"]["text"] == "b"
    with pytest.raises(PersistenceError, match="stale_source_revision"):
        reader.sync([record("a", "old")], forbidden, source_revision=2)
    index.sync([], forbidden, source_revision=4)
    assert create(tmp_path).open().loaded
    reader.refresh(minimum_source_revision=4)
    assert reader.database.items == {}


@pytest.mark.parametrize("failure", ["provider", "build", "save", "after_commit"])
def test_record_projection_failure_preserves_committed_state(tmp_path, monkeypatch, failure):
    index = create(tmp_path)
    index.sync([record("a", "one")], payloads, source_revision=1)
    def fail(*args):
        raise OSError("injected")
    provider = fail if failure == "provider" else payloads
    if failure == "build":
        monkeypatch.setattr(RecordDatabase, "build_records", fail)
    if failure == "save":
        monkeypatch.setattr(RecordDatabase, "save", fail)
    if failure == "after_commit":
        monkeypatch.setattr(index, "_prune_generations", fail)
    with pytest.raises(OSError):
        index.sync([record("b", "two")], provider, source_revision=2)
    assert index.source_revision == (2 if failure == "after_commit" else 1)
    assert set(index.vector_ids) == ({"b"} if failure == "after_commit" else {"a"})


def test_record_projection_repairs_from_current_catalog_and_rejects_mode_conflict(tmp_path):
    catalog = VersionedDocumentCatalog(AtomicMemory(), namespace="knowledge")
    catalog.commit("one", {"old": {"text": "old"}}, expected_revision=0)
    index = create(tmp_path)
    def build(key, document):
        return record(key, document["text"])
    index.rebuild_from_catalog(catalog, build, payloads)
    (tmp_path / "CURRENT").write_bytes(b"\xffbroken")
    catalog.commit("two", {"old": None, "new": {"text": "new"}}, expected_revision=1)
    repaired = index.rebuild_from_catalog(catalog, build, payloads)
    assert repaired.projected == 1 and repaired.embedded == 0
    assert index.source_revision == 2 and set(index.vector_ids) == {"new"}
    wrong = PersistentSageVDB(tmp_path, RecordDatabase, compatibility={"kind": "text"},
                              source_namespace="knowledge", build_mode="snapshot")
    with pytest.raises(PersistenceError):
        wrong.open()


def test_record_payloads_are_validated_and_detached(tmp_path):
    index = create(tmp_path)
    external = {"text": "body", "nested": {"tags": []}}
    index.sync([record("a", "one")], lambda _: [external], source_revision=1)
    external["nested"]["tags"].append("changed")
    assert index.database.items[0]["payload"]["nested"]["tags"] == []
    with pytest.raises(ValueError, match="must_be_records"):
        index.sync([record("b", "two")], lambda _: [[1.0, 2.0]], source_revision=2)
    assert index.source_revision == 1
