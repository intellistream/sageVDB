# ADR 0001: Persistence and multimodal contract tests

- Date: 2026-03-01
- Status: Accepted
- Issue: https://github.com/intellistream/sageVDB/issues/29

## Context

Issue #29 requires explicit contract coverage for:

1. `SageVDB::save/load` persistence path behavior.
2. Multimodal fusion/search interface stability.

Existing tests covered happy-path functionality, but lacked focused contract checks for:
- sidecar artifact contract (`.vectors`, `.metadata`, `.config`)
- base-path load contract
- multimodal boundary behavior (`empty/overflow modalities`)
- default supported fusion strategy surface and query-path API stability

## Decision

1. Add Python contract tests in `tests/test_issue29_persistence_contract.py` to enforce:
   - save sidecar artifacts are created
   - load roundtrip preserves retrieval behavior and persisted configuration
   - load expects base path contract (reject sidecar path as root)
2. Extend `tests/test_multimodal.cpp` with interface contract checks:
   - expected default fusion strategies remain available
   - `update_fusion_params/get_fusion_params` contract remains stable
   - boundary exceptions for empty and overflow modality payloads
   - `search_multimodal` API returns bounded results and metadata path is valid

## Consequences

- Persistence sidecar and load-path contracts are now explicit and regression guarded.
- Multimodal interface behavior is verified at boundary and API-surface levels.
- No compatibility shim/fallback/re-export path is introduced.

## Validation

- `pytest -q tests/test_issue29_persistence_contract.py tests/test_persistence.py`
- `cmake -S . -B build -DBUILD_TESTS=ON -DENABLE_MULTIMODAL=ON -DBUILD_PYTHON_BINDINGS=OFF`
- `cmake --build build --target test_multimodal --parallel 2`
- `ctest --test-dir build -R test_multimodal --output-on-failure`
