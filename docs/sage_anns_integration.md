# sageVDB + sage-anns Integration

## Overview

This document describes the current Python adapter integration between **sageVDB** (vector database) and **sage-anns** (ANN algorithms library). Today, `sage-anns` is not a native sageVDB C++ plugin: sageVDB's native core uses its own `ANNSRegistry` boundary, while `backend="sage-anns"` selects a separate Python adapter path.

## Architecture

```
┌─────────────────────────────────────────┐
│         SageVDB Python API              │
│    (create_database, SageVDB, etc.)     │
└──────────────┬──────────────────────────┘
               │
               ├─── backend="cpp" (default)
               │    └──> C++ SageVDB core via ANNSRegistry
               │         ├─ brute_force (baseline)
               │         └─ faiss (optional)
               │
               └─── backend="sage-anns"
                  └──> Python adapter backend (SageANNSVectorStore)
                         └──> sage-anns library
                        ├─ FAISS / FAISS HNSW
                              ├─ VSAG HNSW
                              ├─ GTI
                              ├─ PLSH
                        └─ other wrappers exposed by the installed sage-anns package
```

   ## Boundary Today

   - The native sageVDB plugin system is the C++ `ANNSRegistry` path used by `VectorStore`.
   - The `backend="sage-anns"` route bypasses that native registry and instantiates `SageANNSVectorStore` in Python.
   - This means `sage-anns` currently behaves as an optional algorithm provider consumed through a Python adapter, not as a native registered C++ plugin.

## Design Rationale

### Why Python-level integration?

1. **Fast iteration**: New ANNS algorithms can be added to `sage-anns` without rebuilding sageVDB's C++ core
2. **Ease of deployment**: No CMake/compiler dependencies for algorithm updates
3. **Compatibility**: Works with pre-built sageVDB wheels from PyPI
4. **Flexibility**: C++ core remains stable while algorithm library evolves

### When to use which backend?

- **C++ backend** (`backend="cpp"`): Default native core, production-ready, performance-critical applications
- **sage-anns backend** (`backend="sage-anns"`): Optional Python adapter for prototyping, evaluation, and wider algorithm access without rebuilding the C++ core

## Installation

```bash
# Install sageVDB (includes C++ backend)
pip install isage-vdb

# Install sage-anns integration (optional)
pip install 'isage-vdb[sage-anns]'

# Or install sage-anns separately
pip install isage-anns
```

## Usage

### Basic Usage

```python
from sagevdb import create_database
import numpy as np

# Create database with sage-anns backend
db = create_database(
    128,  # dimension
    backend="sage-anns",
    algorithm="faiss_hnsw",  # any sage-anns algorithm
    metric="l2",
    M=32,
    ef_construction=200,
)

# Add vectors
vectors = np.random.rand(1000, 128).astype('float32')
db.build_index(vectors)

# Search
query = np.random.rand(128).astype('float32')
results = db.search(query, k=10)
```

### Using DatabaseConfig

```python
from sagevdb import create_database, DatabaseConfig, DistanceMetric

config = DatabaseConfig(128)
config.metric = DistanceMetric.L2
config.anns_algorithm = "faiss_hnsw"
config.anns_build_params = {"M": "32", "ef_construction": "200"}

db = create_database(config, backend="sage-anns")
```

### List Available Algorithms

```python
from sagevdb import list_sage_anns_algorithms

algorithms = list_sage_anns_algorithms()
print(f"Available: {algorithms}")
```

## API Compatibility

The `sage-anns` backend provides a subset of the full SageVDB API:

| Feature | Supported | Notes |
|---------|-----------|-------|
| `build_index()` | ✅ | Main indexing method |
| `add()` | ✅ | Incremental insertion |
| `add_batch()` | ✅ | Batch insertion |
| `search()` | ✅ | k-NN search |
| `batch_search()` | ✅ | Batch queries |
| Metadata | ✅ | In-memory only via SageVDB's `MetadataStore` adapter wrapper; adapter normalizes metadata keys and values to strings before storing them |
| `save()`/`load()` | ✅ | Delegates to the external ANN index only; metadata and local ID state are not fully persisted by the adapter |
| `anns_query_params` from `DatabaseConfig` | ❌ | Not auto-forwarded into search calls; pass keyword args to `search()` / `batch_search()` directly |
| `remove()` | ❌ | Not yet implemented |
| `update()` | ❌ | Not yet implemented |
| `size()` / native object graph accessors | ❌ | The adapter exposes `dimension`, `metric`, and `algorithm`, but not the native `SageVDB` introspection surface |

## Implementation Details

### Components

1. **`SageANNSVectorStore`** (`sagevdb/sage_anns.py`): Python adapter class
   - Wraps `sage_anns.create_index()`
   - Manages metadata via `MetadataStore`
   - Translates a VectorStore-like subset of the sageVDB API to the sage-anns API

2. **`create_database()` factory** (`sagevdb/__init__.py`): Backend router
   - `backend="cpp"` → C++ `SageVDB`
   - `backend="sage-anns"` → `SageANNSVectorStore`

3. **Python bindings update** (`python/bindings.cpp`): Exposed config params
   - `DatabaseConfig.anns_algorithm`
   - `DatabaseConfig.anns_build_params`
   - Native backend also exposes `DatabaseConfig.anns_query_params`, but the current sage-anns adapter does not consume them automatically

The important boundary is that this routing happens in the Python factory and adapter layer, not in `VectorStore`'s native plugin loading path.

### Parameter Mapping

The current adapter maps `DatabaseConfig` to sage-anns like this:

```python
config.anns_algorithm        → algorithm name
config.anns_build_params     → index construction params
config.metric                → distance metric
```

`config.anns_query_params` are part of the native SageVDB configuration surface, but they are not automatically forwarded by the current Python adapter. For search-time controls on the adapter path, pass keyword arguments directly to `search()` or `batch_search()`.

Metadata on the adapter path is stored through SageVDB's `MetadataStore`, which currently exposes a string-based mapping contract in Python. The `sage-anns` adapter therefore normalizes metadata keys and values to strings during `build_index()`, `add()`, `add_batch()`, and metadata restore during `load()`. If you pass integers, booleans, or other scalar values, expect them to be returned as strings in query results.

The adapter also does not expose the native `SageVDB.size()`, `vector_store()`, `query_engine()`, or `metadata_store()` accessors. Examples that need a vector count should track inserted items explicitly instead of reading `dimension`.

## Testing

Run integration tests:

```bash
# Requires sage-anns installed
LD_LIBRARY_PATH=/path/to/sagevdb/sagevdb:$LD_LIBRARY_PATH pytest tests/test_sage_anns_backend.py -v
```

## Future Enhancements

### Short-term
- Add `remove()` and `update()` support
- Expose more algorithm-specific parameters
- Add more tests for different algorithms

### Long-term (optional)
- Native C++ adapter for sage-anns through SageVDB's ANNSRegistry boundary
- Unified configuration schema
- Performance benchmarking suite

## See Also

- [sage-anns README](https://github.com/intellistream/sage-anns)
- [sageVDB ANNS Plugin Guide](./anns_plugin_guide.md)
- [Example: sage_anns_integration_example.py](../examples/sage_anns_integration_example.py)
