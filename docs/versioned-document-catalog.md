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

### Version-fenced index generations

Create `PersistentSageVDB(..., source_namespace="knowledge")` on a fresh index
root and supply `source_revision=snapshot["revision"]` to `sync`. The namespace
is fixed for that root, and the revision is saved in the same generation manifest
as the records/index. Each writer rechecks CURRENT under the inter-process lock.
Older revisions fail before embedding; an equal revision is an idempotent no-op
only when all keys, content hashes and metadata match exactly. A higher revision
is persisted even if vectors are unchanged. `source_revision` reports the loaded
revision, including after restart or a lost post-commit acknowledgement.

Do not derive this revision from task arrival time. Use the canonical catalog
snapshot revision and ensure all records belong to that same snapshot. This fence
prevents index rollback, but it does not prove the index has caught up with newer
canonical changes. Reads must still apply canonical deletion/authorization checks.

Versioned roots fail closed when CURRENT is missing/corrupt or its generation is
invalid: loading an older generation would also lose the high-water mark and let
an old task resurrect deleted data. Explicit rebuilding from trusted canonical
state and atomic replacement of such a root is still an integration gate; do not
retry as unversioned or serve an older generation as current. Unversioned users
retain their existing last-known-good recovery semantics. There is no automatic
in-place conversion of an existing unversioned root to a versioned namespace.

### Application cutover

This is the canonical commit primitive, **not a completed indexing pipeline**.
Index workers must use the version fence, recover incomplete generations,
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
