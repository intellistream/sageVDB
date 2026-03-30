#!/usr/bin/env python3
"""Issue #29 contract tests for persistence path behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sagevdb import (
    DatabaseConfig,
    DistanceMetric,
    IndexType,
    SageVDB,
    SearchParams,
    add_numpy,
    search_numpy,
)


class TestIssue29PersistenceContract:
    """Contract checks for save/load path and sidecar persistence."""

    def test_save_creates_required_sidecar_files(self, tmp_path: Path) -> None:
        config = DatabaseConfig(32)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.L2

        db = SageVDB(config)
        vectors = np.random.default_rng(2026).random((20, 32), dtype=np.float32)
        add_numpy(db, vectors)
        db.build_index()

        base_path = tmp_path / "issue29_contract_db"
        db.save(str(base_path))

        assert (tmp_path / "issue29_contract_db.vectors").exists()
        assert (tmp_path / "issue29_contract_db.metadata").exists()
        assert (tmp_path / "issue29_contract_db.config").exists()

    def test_load_roundtrip_preserves_search_contract(self, tmp_path: Path) -> None:
        config = DatabaseConfig(16)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.COSINE

        db1 = SageVDB(config)
        vectors = np.random.default_rng(29).random((50, 16), dtype=np.float32)
        add_numpy(db1, vectors)
        db1.build_index()

        query = np.random.default_rng(290).random(16, dtype=np.float32)
        params = SearchParams(k=5)
        results_before = search_numpy(db1, query, params)

        base_path = tmp_path / "issue29_roundtrip"
        db1.save(str(base_path))

        # Start with intentionally different config to validate persisted config reload.
        config2 = DatabaseConfig(4)
        config2.index_type = IndexType.HNSW
        config2.metric = DistanceMetric.L2
        db2 = SageVDB(config2)
        db2.load(str(base_path))

        assert db2.dimension() == 16
        assert db2.index_type() == IndexType.FLAT
        assert db2.size() == 50

        results_after = search_numpy(db2, query, params)
        assert len(results_before) == len(results_after)

        for left, right in zip(results_before, results_after):
            assert left.id == right.id
            assert abs(left.score - right.score) < 1e-5

    def test_load_requires_base_path_not_sidecar_path(self, tmp_path: Path) -> None:
        config = DatabaseConfig(8)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.L2

        db = SageVDB(config)
        vectors = np.random.default_rng(99).random((8, 8), dtype=np.float32)
        add_numpy(db, vectors)
        db.build_index()

        base_path = tmp_path / "issue29_sidecar_contract"
        db.save(str(base_path))

        db2 = SageVDB(config)
        with pytest.raises(Exception):
            db2.load(str(base_path) + ".vectors")
