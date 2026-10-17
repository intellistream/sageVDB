"""Versioned canonical documents, independent of derived vector generations.

The injected state backend owns transactions. ``prepare`` returns a pure slot
mutation that can join a SAGE work-journal transaction; no scheduler or second
database is created here. Index failures must not discard canonical documents.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


class DocumentConflict(ValueError):
    """The caller's revision or operation identity does not match durable state."""


class VersionedDocumentCatalog:
    """CAS updates, tombstones and replay receipts over atomic state slots.

    ``state`` must implement detached ``peek(slot)`` and transactional
    ``update(slot, pure_transform)``. In SAGE, use StatefulProcessRuntime with
    the same backend as WorkJournal when composing a multi-slot commit.
    Receipts and tombstones are retained; retention is deliberately not implicit.
    """

    def __init__(self, state, *, namespace: str):
        if not isinstance(namespace, str) or not namespace.strip() or namespace != namespace.strip():
            raise ValueError("document_namespace_must_be_nonempty_and_trimmed")
        self.state = state
        self.slot = ("sagevdb-documents-v1", namespace, "catalog")

    @staticmethod
    def _collection(value):
        if value is None:
            return {"version": 1, "revision": 0, "records": {}, "receipts": {}}
        if value.get("version") != 1:
            raise ValueError("unsupported_document_catalog_version")
        return value

    def snapshot(self) -> dict[str, Any]:
        """Detached canonical state; tombstones prevent accidental resurrection."""
        state = self._collection(self.state.peek(self.slot))
        return _json_copy({"revision": state["revision"], "records": state["records"]})

    def prepare(self, operation_id: str, changes: Mapping[str, dict | None], *, expected_revision: int):
        """Freeze a batch; return (slot, pure transform) for one atomic commit.

        A document is opaque JSON. None means deletion. A successful nonempty
        batch increments the catalog revision once. An identical replay returns
        its original receipt even after later changes; conflicting reuse fails.
        """
        if not isinstance(operation_id, str) or not operation_id.strip() or operation_id != operation_id.strip():
            raise ValueError("document_operation_id_must_be_nonempty_and_trimmed")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("document_revision_must_be_nonnegative_integer")
        if not isinstance(changes, Mapping) or not changes:
            raise ValueError("document_changes_must_be_nonempty_mapping")
        for key, document in changes.items():
            if not isinstance(key, str) or not key.strip() or key != key.strip():
                raise ValueError("document_key_must_be_nonempty_and_trimmed")
            if document is not None and not isinstance(document, dict):
                raise ValueError("document_must_be_json_object_or_tombstone")
        encoded = json.dumps({"expected_revision": expected_revision, "changes": dict(changes)},
                             sort_keys=True, ensure_ascii=False, allow_nan=False)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()

        def apply(value):
            state = self._collection(value)
            receipt = state["receipts"].get(operation_id)
            if receipt is not None:
                if receipt["fingerprint"] != fingerprint:
                    raise DocumentConflict("document_operation_identity_conflict")
                return state
            if state["revision"] != expected_revision:
                raise DocumentConflict("document_revision_conflict")
            revision = expected_revision + 1
            # Decode anew on each invocation: a failed/retried transform never
            # shares mutable application objects with another transaction.
            for key, document in json.loads(encoded)["changes"].items():
                state["records"][key] = {"revision": revision, "document": document}
            state["revision"] = revision
            state["receipts"][operation_id] = {"fingerprint": fingerprint, "revision": revision}
            return state

        return self.slot, apply

    def commit(self, operation_id: str, changes: Mapping[str, dict | None], *, expected_revision: int) -> int:
        slot, mutation = self.prepare(operation_id, changes, expected_revision=expected_revision)
        state = self.state.update(slot, mutation)
        return int(state["receipts"][operation_id]["revision"])
