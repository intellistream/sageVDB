from concurrent.futures import ThreadPoolExecutor

import pytest

from sagevdb.document_catalog import DocumentConflict, VersionedDocumentCatalog


class AtomicMemory:
    """Detached transactional test backend, without an optional SAGE dependency."""
    def __init__(self):
        self.values = {}

    def peek(self, slot):
        import copy
        return copy.deepcopy(self.values.get(slot))

    def update(self, slot, transform):
        import copy
        value = transform(self.peek(slot))
        self.values[slot] = copy.deepcopy(value)
        return copy.deepcopy(value)


def test_replay_keeps_newer_versions_and_tombstones():
    store = VersionedDocumentCatalog(AtomicMemory(), namespace="knowledge")
    assert store.commit("first", {"a": {"text": "one"}}, expected_revision=0) == 1
    assert store.commit("second", {"a": None, "b": {"text": "two"}}, expected_revision=1) == 2
    assert store.commit("first", {"a": {"text": "one"}}, expected_revision=0) == 1
    assert store.snapshot() == {"revision": 2, "records": {
        "a": {"revision": 2, "document": None},
        "b": {"revision": 2, "document": {"text": "two"}},
    }}
    with pytest.raises(DocumentConflict, match="identity_conflict"):
        store.commit("first", {"a": {"text": "changed"}}, expected_revision=0)
    with pytest.raises(DocumentConflict, match="revision_conflict"):
        store.commit("stale", {"a": {"text": "resurrection"}}, expected_revision=1)


def test_prepared_input_and_read_snapshots_are_detached():
    state = AtomicMemory()
    store = VersionedDocumentCatalog(state, namespace="knowledge")
    changes = {"a": {"text": ["original"]}}
    slot, transform = store.prepare("one", changes, expected_revision=0)
    changes["a"]["text"].append("late mutation")
    state.update(slot, transform)
    snapshot = store.snapshot()
    snapshot["records"]["a"]["document"]["text"].append("read mutation")
    assert store.snapshot()["records"]["a"]["document"] == {"text": ["original"]}
    assert VersionedDocumentCatalog(state, namespace="other").snapshot()["revision"] == 0


@pytest.mark.parametrize("changes,revision", [({}, 0), ({"": {}}, 0), ({"a": []}, 0),
                                             ({"a": {}}, -1), ({"a": {}}, True),
                                             ({"a": {"x": float("nan")}}, 0)])
def test_invalid_batch_never_changes_state(changes, revision):
    store = VersionedDocumentCatalog(AtomicMemory(), namespace="knowledge")
    with pytest.raises(ValueError):
        store.commit("op", changes, expected_revision=revision)
    assert store.snapshot() == {"revision": 0, "records": {}}


def test_sqlite_reopen_ack_loss_and_competing_revisions(tmp_path, monkeypatch):
    module = pytest.importorskip("sage.runtime.flownet.runtime.operator_runtime.stateful")
    first = module.StatefulProcessRuntime(backend="sqlite", sqlite_path=str(tmp_path / "state.db"))
    second = module.StatefulProcessRuntime(backend="sqlite", sqlite_path=str(tmp_path / "state.db"))
    a = VersionedDocumentCatalog(first, namespace="knowledge")
    b = VersionedDocumentCatalog(second, namespace="knowledge")
    original = first.update
    try:
        with monkeypatch.context() as patch:
            def lost_ack(*args):
                original(*args)
                raise OSError("lost ack")
            patch.setattr(first, "update", lost_ack)
            with pytest.raises(OSError, match="lost ack"):
                a.commit("one", {"a": {"text": "saved"}}, expected_revision=0)
        assert b.commit("one", {"a": {"text": "saved"}}, expected_revision=0) == 1

        def competing(item):
            key, catalog = item
            try:
                return catalog.commit(key, {key: {"text": key}}, expected_revision=1)
            except DocumentConflict:
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(competing, [("b", a), ("c", b)]))
        assert sorted(map(str, results)) == ["2", "conflict"]
        assert a.snapshot() == b.snapshot()
        assert len(a.snapshot()["records"]) == 2
    finally:
        first.close()
        second.close()


def test_document_and_work_intent_share_transaction(tmp_path):
    module = pytest.importorskip("sage.runtime.flownet.runtime.operator_runtime.stateful")
    journal_module = pytest.importorskip("sage.runtime.work_journal")
    state = module.StatefulProcessRuntime(backend="sqlite", sqlite_path=str(tmp_path / "state.db"))
    catalog = VersionedDocumentCatalog(state, namespace="knowledge")
    journal = journal_module.WorkJournal(state, namespace="indexing")
    try:
        slot, transform = catalog.prepare("one", {"a": {"text": "saved"}}, expected_revision=0)
        journal.commit("one", result={"revision": 1}, tasks={"index": {"revision": 1}},
                       mutations={slot: transform})
        before = catalog.snapshot()
        slot, transform = catalog.prepare("conflict", {"a": None}, expected_revision=1)
        with pytest.raises(ValueError):
            journal.commit("one", result={"revision": 2}, tasks={"index": {"revision": 2}},
                           mutations={slot: transform})
        assert catalog.snapshot() == before
        # Unserializable second slot fails after the earlier SQL write.
        other = ("zz", "failure", "slot")
        with pytest.raises((TypeError, AttributeError)):
            state.update_many([slot, other], lambda values: {
                slot: transform(values[slot]), other: lambda: None})
        assert catalog.snapshot() == before
    finally:
        state.close()


@pytest.mark.parametrize("after_commit", [False, True])
def test_process_exit_preserves_atomic_document_and_task(tmp_path, after_commit):
    import multiprocessing
    import os
    module = pytest.importorskip("sage.runtime.flownet.runtime.operator_runtime.stateful")
    journal_module = pytest.importorskip("sage.runtime.work_journal")
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("process-exit test requires fork")
    path = str(tmp_path / "crash.db")

    def child():
        state = module.StatefulProcessRuntime(backend="sqlite", sqlite_path=path)
        catalog = VersionedDocumentCatalog(state, namespace="knowledge")
        journal = journal_module.WorkJournal(state, namespace="indexing")
        slot, transform = catalog.prepare("one", {"a": {"text": "durable"}}, expected_revision=0)
        def interrupted(value):
            result = transform(value)
            if not after_commit:
                os._exit(17)
            return result
        journal.commit("one", result={"revision": 1}, tasks={"index": {"revision": 1}},
                       mutations={slot: interrupted})
        os._exit(18)  # No close, finalizer, worker start or response delivery.

    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    process.join(10)
    try:
        assert process.exitcode == (18 if after_commit else 17)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
    state = module.StatefulProcessRuntime(backend="sqlite", sqlite_path=path)
    try:
        catalog = VersionedDocumentCatalog(state, namespace="knowledge")
        journal = journal_module.WorkJournal(state, namespace="indexing")
        assert catalog.snapshot()["revision"] == (1 if after_commit else 0)
        assert (journal.read("one") is not None) is after_commit
    finally:
        state.close()
