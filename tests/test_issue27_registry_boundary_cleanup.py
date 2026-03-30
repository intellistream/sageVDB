"""Regression checks for issue #27: VectorStore/ANNSRegistry boundary cleanup."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue27_public_include_layer_has_no_plugin_headers() -> None:
    """Implementation plugin headers must not be exposed from public include layer."""
    assert not (REPO_ROOT / "include/sage_vdb/anns/brute_force_plugin.h").exists()
    assert not (REPO_ROOT / "include/sage_vdb/anns/faiss_plugin.h").exists()


def test_issue27_vector_store_no_plugin_header_dependency() -> None:
    """VectorStore should depend on ANNSRegistry abstraction only."""
    content = _read("src/vector_store.cpp")

    assert '#include "sage_vdb/anns/brute_force_plugin.h"' not in content
    assert '#include "sage_vdb/anns/faiss_plugin.h"' not in content
    assert "register_builtin_algorithms" in content


def test_issue27_no_hidden_algorithm_fallback_in_initialize() -> None:
    """Unknown algorithm should fail fast instead of implicit brute-force fallback."""
    content = _read("src/vector_store.cpp")

    assert 'algorithm_name_ = "brute_force";' not in content
    assert "is not registered" in content
