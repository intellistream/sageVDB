# SageVDB + sage-anns Integration: Multi-Agent Prompts

Below are prompts you can copy into separate agents to parallelize the integration plan and implementation.

## Agent 1: Architecture and API Mapping
Prompt:
You are integrating sage-anns into sageVDB. Study the sageVDB public API (VectorStore, QueryEngine, ANNSRegistry, DatabaseConfig, SearchParams) and design how sage-anns algorithms should appear as pluggable ANNS backends. Identify which sage-anns indices map to FAISS-like IndexType and DistanceMetric, and propose a minimal adapter layer in C++20. Produce a short design doc with class names, method signatures, and a registration strategy using REGISTER_ANNS_ALGORITHM. Do not write code; focus on architecture and API mapping.

## Agent 2: Build System and Dependency Wiring
Prompt:
Review sageVDB CMakeLists.txt and sage-anns build structure. Propose how to include sage-anns as an optional dependency (e.g., ENABLE_SAGE_ANNS) without breaking current builds. Provide a CMake integration plan: add_subdirectory vs. ExternalProject, include paths, library targets, and compile definitions. Identify any required changes to pyproject.toml or packaging so wheels can bundle sage-anns if enabled. Output a step-by-step build plan and potential pitfalls.

## Agent 3: Adapter Implementation Plan
Prompt:
Based on the sageVDB ANNS plugin interface, propose a concrete adapter implementation that wraps a sage-anns index. Identify minimal methods to implement (train, add, search, save/load) and how to translate SearchParams and anns_query_params into sage-anns config. Produce a C++ class skeleton (no full code) and list the files/paths where it should live (include/sage_vdb/anns and src/anns). Include notes about thread-safety and unsupported ops handling.

## Agent 4: Python Binding Surface
Prompt:
Assess how the new sage-anns backend should be exposed via python/bindings.cpp. Propose any new enums or parameters that need to surface in Python (IndexType names, query/build params). Ensure naming matches FAISS-like conventions and defaults stay backward compatible. Provide a minimal change plan and docstring updates, no code.

## Agent 5: Tests and Docs
Prompt:
Create a test and documentation plan for sage-anns integration in sageVDB. Identify existing tests to extend, new tests to add (C++ and Python), and docs to update (README.md, docs/USAGE_MODES.md, docs/guides/README_Multimodal.md if relevant). Provide a checklist and expected commands to run (ctest, test_sage_vdb, test_multimodal).
