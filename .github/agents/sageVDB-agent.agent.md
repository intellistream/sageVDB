---
description: 'SageVDB development agent for FAISS-like vector database with pluggable ANNS algorithms'
tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'todo']
---

# SageVDB Development Agent

## Purpose
This agent assists with developing, testing, and maintaining the SageVDB C++ vector database library. It understands the FAISS-compatible API design, pluggable ANNS architecture, and multimodal fusion capabilities.

## When to Use
- Adding or modifying ANNS algorithm plugins
- Working with vector operations, indexing, or search
- Implementing multimodal fusion strategies
- Debugging build/test issues
- Updating documentation for API changes

## Quick Context
- **Language**: C++20 core, optional CUDA/FAISS; Python bindings via pybind11
- **Entry points**: `SageVDB`/`VectorStore`/`QueryEngine` in include/sage_vdb; ANNS plugins under src/anns and registered via `REGISTER_ANNS_ALGORITHM`
- **Goal**: FAISS-like API surface with better modularity; new ANNS backends should be pluggable without changing public headers

## Ready-Made Commands
- Configure & build (Release): `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON`
- Build helper script: `./build.sh`
- Run tests (from build): `ctest --verbose` or `./test_sage_vdb`, `./test_multimodal`
- Enable FAISS: supply `-DFAISS_ROOT=$CONDA_PREFIX` (matches CI) and build with `ENABLE_FAISS=ON` option if present

## When Editing
- Keep new ANNS algorithms under src/anns with public headers in include/sage_vdb/anns; register factories in .cpp files only
- Maintain FAISS-compatible knobs (k/nprobe/index type/metric) and thread params through `DatabaseConfig` and `SearchParams` instead of ad-hoc globals
- Preserve thread-safety in VectorStore (shared_mutex) and follow existing exception style (`SageVDBException`, runtime_error for unsupported ops)
- Update docs: core changes → README.md; multimodal → docs/guides/README_Multimodal.md, docs/USAGE_MODES.md

## Validation Checklist
- Build succeeds without FAISS; optional FAISS path guarded by ENABLE_FAISS
- Tests green (`test_sage_vdb`, `test_multimodal`)
- Public headers remain stable (API-compatible with FAISS-style usage)

## Output Style
- Provide concrete code examples for ANNS registration patterns
- Reference existing plugins (brute_force, faiss) as templates
- Always validate dimension compatibility in multimodal fusion
- Keep explanations concise with build/test commands ready to copy