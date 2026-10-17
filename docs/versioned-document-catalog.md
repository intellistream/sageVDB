# Canonical documents and derived indexes

`VersionedDocumentCatalog` owns canonical JSON document versions, deletion
tombstones and operation receipts. `PersistentSageVDB` owns derived index
generations. Neither should silently treat an index failure as document loss.

The catalog accepts an injected atomic state backend. It creates no executor,
database or scheduling loop. In SAGE, use the same `StatefulProcessRuntime`
instance for the catalog and `WorkJournal`:

```python
from sagevdb import VersionedDocumentCatalog
from sage.runtime.work_journal import WorkJournal

catalog = VersionedDocumentCatalog(state, namespace="knowledge")
journal = WorkJournal(state, namespace="knowledge-indexing")
snapshot = catalog.snapshot()
slot, transform = catalog.prepare(
    "ingest-operation-123",
    {"document-123": {"content": "Full permitted evidence", "source": "example"}},
    expected_revision=snapshot["revision"],
)
journal.commit(
    "ingest-operation-123",
    result={"revision": snapshot["revision"] + 1},
    tasks={"index": {"revision": snapshot["revision"] + 1}},
    mutations={slot: transform},
)
```

`prepare` freezes input but does not persist it. The combined commit atomically
admits the canonical mutation and indexing intent. An identical retry does not
reapply documents, even after later revisions; reuse of an operation ID with
different content or expected revision raises `DocumentConflict`. Concurrent
changes require explicit reconciliation with a fresh revision, not blind retry
with a new revision. A `None` document is a retained tombstone. Reads are detached.

The component does not authorize callers, infer ownership, publish documents,
truncate content or delete old receipts. Applications must validate source
authority before preparing a batch. State backends must provide detached,
transactional `peek`/`update`; composing with a work journal additionally requires
the same backend and atomic multi-slot updates. No cross-database atomicity is
claimed. Crash tests cover process exit, not power-loss guarantees.

## Remaining integration gates

This is the canonical commit primitive, **not a completed indexing pipeline**.
Index workers must reject stale projections, recover incomplete generations,
and acknowledge only after publishing a valid index. Consumers must not let
stale index metadata bypass current canonical authorization or tombstones.
Catalog snapshots and receipt retention are currently unbounded; capacity,
compaction policy and indexed enumeration require separate qualification.
Twin's legacy JSON import, index-worker integration, dependency packaging and
production cutover are not implied by these component tests.

Tests in `test_document_catalog.py` exercise an independent atomic-memory backend;
when SAGE is installed, they also exercise SQLite concurrency, serialization
rollback, a shared work-journal transaction and real subprocess exit before/after
commit. SAGE is optional for standalone SageVDB, so CI without it must not claim
those integration tests passed merely because they skipped.
