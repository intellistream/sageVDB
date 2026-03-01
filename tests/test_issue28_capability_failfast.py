"""Regression tests for issue #28: no implicit capability fallback."""

from pathlib import Path

import pytest


def test_issue28_unknown_algorithm_fails_fast() -> None:
    """Creating a DB with unknown algorithm must fail instead of falling back."""
    sagevdb = pytest.importorskip("sagevdb")

    config = sagevdb.DatabaseConfig(16)
    config.anns_algorithm = "definitely_unknown_algo"

    with pytest.raises(Exception, match="not registered|algorithm"):
        sagevdb.SageVDB(config)


def test_issue28_no_hidden_bruteforce_reassignment_in_vector_store() -> None:
    """VectorStore implementation must not reassign unknown algorithms to brute_force."""
    repo_root = Path(__file__).parent.parent
    content = (repo_root / "src" / "vector_store.cpp").read_text(encoding="utf-8")

    assert 'algorithm_name_ = "brute_force";' not in content
    assert "is not registered" in content
