"""SageVDB adapter for the sage.libs.vdb.VDBBackend interface.

Registers ``SageVDBBackend`` under the name ``"sagedb"`` so that L4 packages
(e.g. ``isage-neuromem``) can obtain an instance via::

    from sage.libs.vdb import create_backend
    backend = create_backend("sagedb", {"dim": 768})

This module is imported at the bottom of ``sagevdb/__init__.py`` so the
registration happens automatically when ``import sagevdb`` is executed.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from sage.libs.vdb import VDBBackend, register_backend


@register_backend("sagedb")
class SageVDBBackend(VDBBackend):
    """``VDBBackend`` adapter wrapping the C++ ``SageVDB`` engine.

    ``SageVDB`` auto-assigns integer IDs on ``db.add(vector)``.  This
    adapter maintains an in-memory bidirectional mapping between the
    **string** IDs expected by the ``VDBBackend`` contract and the
    **integer** IDs used internally by ``SageVDB``.

    Config keys
    -----------
    dim : int
        Vector dimensionality (default 768).
    index_type : str
        ANNS index type recognised by ``sagevdb.IndexType``
        (default ``"FLAT"``).
    metric : str
        Distance metric recognised by ``sagevdb.DistanceMetric``
        (default ``"L2"``).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        from sagevdb import (
            DatabaseConfig,
            SageVDB,
            string_to_index_type,
            string_to_distance_metric,
        )

        dim: int = int(config.get("dim", 768))
        self._dim = dim

        cfg = DatabaseConfig(dim)

        # Optional index type override
        index_type_str: str = str(config.get("index_type", "FLAT"))
        try:
            cfg.index_type = string_to_index_type(index_type_str)
        except Exception:
            pass  # leave as SageVDB default

        # Optional metric override
        metric_str: str = str(config.get("metric", "L2"))
        try:
            cfg.metric = string_to_distance_metric(metric_str)
        except Exception:
            pass  # leave as SageVDB default

        self._cfg = cfg
        self._db = SageVDB(cfg)
        self._db.build_index()

        # Bidirectional ID mapping: str_id ↔ int_id
        self._str_to_int: dict[str, int] = {}
        self._int_to_str: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        if len(ids) != len(vectors) or len(ids) != len(metadata):
            raise ValueError(
                f"ids, vectors, metadata must have the same length; "
                f"got {len(ids)}, {len(vectors)}, {len(metadata)}"
            )

        for str_id, vector, meta in zip(ids, vectors, metadata):
            # Insert vector  →  C++ returns auto-assigned int ID
            int_id: int = self._db.add(
                [float(v) for v in vector] if not isinstance(vector, list) else vector
            )
            # Persist mapping
            self._str_to_int[str_id] = int_id
            self._int_to_str[int_id] = str_id
            # Persist all user metadata + our reserved __str_id__ key
            self._db.set_metadata(int_id, {"__str_id__": str_id, **meta})

        # Keep index fresh
        self._db.build_index()

    def delete(self, ids: list[str]) -> bool:
        deleted_any = False
        for str_id in ids:
            int_id = self._str_to_int.get(str_id)
            if int_id is None:
                continue
            ok: bool = self._db.remove(int_id)
            if ok:
                self._str_to_int.pop(str_id, None)
                self._int_to_str.pop(int_id, None)
                deleted_any = True
        return deleted_any

    def clear(self) -> bool:
        from sagevdb import SageVDB

        self._db = SageVDB(self._cfg)
        self._db.build_index()
        self._str_to_int.clear()
        self._int_to_str.clear()
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        filter_metadata: dict[str, Any] | None = None,
        top_k: int = 10,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        from sagevdb import SearchParams, search_numpy

        if query_vector is not None:
            # Vector similarity search
            q_np = np.array(query_vector, dtype=np.float32)
            params = SearchParams(k=min(top_k, max(self._db.size(), 1)))
            raw_results = search_numpy(self._db, q_np, params)

            results: list[dict[str, Any]] = []
            for r in raw_results:
                int_id = r.id
                str_id = self._int_to_str.get(int_id)
                if str_id is None:
                    continue  # ID already deleted
                meta = dict(self._db.get_metadata(int_id))
                meta.pop("__str_id__", None)

                # Apply metadata filter if requested
                if filter_metadata and not _matches(meta, filter_metadata):
                    continue

                results.append({"id": str_id, "score": r.score, "metadata": meta})
                if len(results) >= top_k:
                    break
            return results

        # Metadata-only filter (no query vector)
        results = []
        for str_id, int_id in list(self._str_to_int.items()):
            meta = dict(self._db.get_metadata(int_id))
            meta.pop("__str_id__", None)
            if filter_metadata and not _matches(meta, filter_metadata):
                continue
            results.append({"id": str_id, "metadata": meta})
            if len(results) >= top_k:
                break
        return results

    def get_all_ids(self) -> list[str]:
        return list(self._str_to_int.keys())

    def count(self) -> int:
        return len(self._str_to_int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches(meta: dict[str, Any], filter_metadata: dict[str, Any]) -> bool:
    """Return True if all key/value pairs in *filter_metadata* are present in *meta*."""
    return all(str(meta.get(k)) == str(v) for k, v in filter_metadata.items())
