# Changelog

All notable changes to isage-vdb will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.10] - 2026-02-14

### Fixed
- **CRITICAL**: Added `faiss-cpu>=1.7.0` to core dependencies in pyproject.toml
  - libsage_vdb.so is compiled with FAISS support and requires libfaiss.so at runtime
  - Previously faiss-cpu was optional, causing OSError when importing sagevdb
- Improved error message when libfaiss.so is missing during module import

### Changed
- Removed redundant `faiss` from optional dependencies (now in core)

## [0.1.9] - 2026-02-13

### Added
- Initial release with FAISS-compatible API
- Support for multiple index types (FLAT, IVF, HNSW)
- Metadata filtering and hybrid search
- Persistent storage (save/load)
