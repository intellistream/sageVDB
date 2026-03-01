# ADR 0001: Converge VectorStore and ANNSRegistry boundary

- Date: 2026-03-01
- Status: Accepted
- Issue: https://github.com/intellistream/sageVDB/issues/27

## Context

The public include layer exposed plugin implementation headers (`brute_force_plugin.h`,
`faiss_plugin.h`). `VectorStore` also directly included plugin headers, which coupled public
orchestration code with concrete backend implementations.

## Decision

1. Keep public ANNS surface limited to `include/sage_vdb/anns/anns_interface.h`.
2. Move plugin headers to internal source layer (`src/anns/*.h`).
3. Remove direct plugin-header dependency from `src/vector_store.cpp`.
4. Add built-in registry bootstrap (`src/anns/register_builtin_algorithms.cpp`) and fail fast
   when requested algorithm is not registered.

## Consequences

- Public API no longer leaks FAISS/brute-force implementation details.
- `VectorStore` depends on ANNS abstraction and registration boundary only.
- No shim/re-export/fallback compatibility path is introduced.

## Validation

- `ruff check tests/test_issue27_registry_boundary_cleanup.py`
- `pytest -q tests/test_issue27_registry_boundary_cleanup.py`
