# Copilot Instructions for sageDB

These guardrails keep completions consistent with the repo goal: FAISS-style API with a modular, pluggable ANNS core and multimodal fusion.

## Design North Star
- Mirror FAISS user ergonomics: `IndexType`, `DistanceMetric`, `DatabaseConfig`, `train_index()`, `build_index()`, `search()`, and IVF/HNSW params match FAISS naming where possible.
- Everything runs through `VectorStore`/`QueryEngine` with algorithm lookup via `anns::ANNSRegistry`. New algorithms must register factories with `REGISTER_ANNS_ALGORITHM` and avoid leaking implementation headers.
- Multimodal stays additive: `MultimodalSageDB` composes `ModalityManager` and `FusionEngine` (concat/weighted/attention/etc.) without coupling to specific embedders.

## Preferred APIs
- For vector DB entry points use `SageDB` and `create_database()`; validate dimensions with `DatabaseConfig` rather than ad-hoc checks.
- Use `SearchParams` for k/nprobe/radius/metadata flags; use `anns::QueryConfig` for algorithm-specific knobs; keep `anns_build_params`/`anns_query_params` stringly-typed like FAISS CLI.
- Metadata flows through `MetadataStore` and `QueryResult.metadata`; keep it optional but stable.
- Persistence goes through `VectorStore::save/load` and mirrors FAISS index IO semantics; avoid one-off serializers.

## Implementation Rules
- Respect capabilities: if an algorithm cannot update/delete/range-search, throw `std::runtime_error` with the algorithm name (see `anns::ANNSAlgorithm` defaults).
- Keep thread safety: `VectorStore` uses shared_mutex for read-heavy paths; do not expose raw internals that bypass locking.
- Favor C++20 standard library; avoid introducing new deps unless aligned with cmake options (`ENABLE_FAISS`, `ENABLE_MULTIMODAL`, etc.).
- Keep code pluggable: new ANNS implementations live under `src/anns/` + public headers under `include/sage_db/anns/`; register via factory macro in a `.cpp` file only.

## Testing & Build
- Default build: `./build.sh`; CI builds with `cmake -B build ...` and optional `-DFAISS_ROOT` when FAISS is present.
- Tests: from `build/`, run `ctest --verbose` or `./test_sage_db`, `./test_multimodal`.

## Python Binding Notes
- `python/bindings.cpp` exposes the C++ core; keep interface names and docstrings aligned with the C++ API (FAISS-like naming, e.g., `search`, `add`, `train_index`).
- When adding params, thread them through `DatabaseConfig` -> `VectorStore` -> bindings; keep defaults backward compatible.

## Documentation Expectations
- Update `README.md` for core vector/ANNS changes; use `docs/guides/README_Multimodal.md` and `docs/USAGE_MODES.md` when changing multimodal flows.
- Keep option names and examples FAISS-compatible to ease user migration.

## Publishing

**⚠️ IMPORTANT: Publishing requires explicit version update and manual action**

### Publishing Workflow
1. **Update version** in `pyproject.toml` (current: 0.1.5)
   - Bug fixes: increment patch (0.1.5 → 0.1.6)
   - New features: increment minor (0.1.5 → 0.2.0)
   - Breaking changes: increment major (0.1.5 → 1.0.0)

2. **Build and publish** using `sage-pypi-publisher`:
   ```bash
   # Option 1: Automated (recommended)
   sage-pypi-publisher publish  # Interactive: asks for version update and PyPI upload
   
   # Option 2: Manual steps
   sage-pypi-publisher build-manylinux     # Build wheels
   sage-pypi-publisher upload wheelhouse/*.whl  # Upload to PyPI
   ```

3. **Git hook behavior**: The pre-push hook will:
   - Warn if version wasn't updated
   - Ask [u/y/n]: update now / continue / cancel
   - Choosing 'y' pushes to GitHub but does NOT publish to PyPI

### Notes
- Use latest `isage-pypi-publisher` from `sage` conda env
- PyPI token must be in `~/.pypirc`
- Push to GitHub and publish to PyPI are separate steps
- See `docs/ops/RELEASE.md` and `docs/ops/DEPLOYMENT.md` for details
