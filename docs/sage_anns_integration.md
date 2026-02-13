# sageVDB + sage-anns Integration

## Overview

This document describes the Python-level integration between **sageVDB** (vector database) and **sage-anns** (ANNS algorithms library).

## Architecture

```
┌─────────────────────────────────────────┐
│         SageVDB Python API              │
│    (create_database, SageVDB, etc.)     │
└──────────────┬──────────────────────────┘
               │
               ├─── backend="cpp" (default)
               │    └──> C++ SageVDB core
               │         ├─ brute_force (baseline)
               │         └─ faiss (optional)
               │
               └─── backend="sage-anns"
                    └──> Python SageANNSVectorStore
                         └──> sage-anns library
                              ├─ FAISS HNSW
                              ├─ VSAG HNSW
                              ├─ GTI
                              ├─ PLSH
                              ├─ DiskANN
                              └─ CANDY (various)
```

## Design Rationale

### Why Python-level integration?

1. **Fast iteration**: New ANNS algorithms can be added to `sage-anns` without rebuilding sageVDB's C++ core
2. **Ease of deployment**: No CMake/compiler dependencies for algorithm updates
3. **Compatibility**: Works with pre-built sageVDB wheels from PyPI
4. **Flexibility**: C++ core remains stable while algorithm library evolves

### When to use which backend?

- **C++ backend** (`backend="cpp"`): Default, production-ready, performance-critical applications
- **sage-anns backend** (`backend="sage-anns"`): New algorithms, prototyping, algorithm comparison

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
| Metadata | ✅ | Via MetadataStore |
| `save()`/`load()` | ✅ | If algorithm supports |
| `remove()` | ❌ | Not yet implemented |
| `update()` | ❌ | Not yet implemented |

## Implementation Details

### Components

1. **`SageANNSVectorStore`** (`sagevdb/sage_anns.py`): Python adapter class
   - Wraps `sage_anns.create_index()`
   - Manages metadata via `MetadataStore`
   - Translates sageVDB API to sage-anns API

2. **`create_database()` factory** (`sagevdb/__init__.py`): Backend router
   - `backend="cpp"` → C++ `SageVDB`
   - `backend="sage-anns"` → `SageANNSVectorStore`

3. **Python bindings update** (`python/bindings.cpp`): Exposed config params
   - `DatabaseConfig.anns_algorithm`
   - `DatabaseConfig.anns_build_params`
   - `DatabaseConfig.anns_query_params`

### Parameter Mapping

FAISS-style parameters in `DatabaseConfig` map to sage-anns:

```python
config.anns_algorithm        → algorithm name
config.anns_build_params     → index construction params
config.anns_query_params     → search-time params
config.metric                → distance metric
```

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
- C++ adapter for sage-anns (deeper integration)
- Unified configuration schema
- Performance benchmarking suite

## See Also

- [sage-anns README](https://github.com/intellistream/sage-anns)
- [sageVDB ANNS Plugin Guide](./anns_plugin_guide.md)
- [Example: sage_anns_integration_example.py](../examples/sage_anns_integration_example.py)
