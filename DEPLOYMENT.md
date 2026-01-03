# sageDB Deployment Strategies

## Overview

sageDB supports **two deployment modes** that serve different use cases:

### 1. Embedded C++ Component (SAGE Framework)
**Used by:** SAGE middleware internal components

**Location:** `SAGE/packages/sage-middleware/src/sage/middleware/components/sage_db/`

**How it works:**
- C++ source code is compiled directly as part of SAGE build
- Python bindings (`_sage_db.so`) are generated via pybind11
- Imported as: `from sage.middleware.components.sage_db import SageDB`
- Version managed in component's `__init__.py`

**Advantages:**
- ✅ Zero installation overhead (built with SAGE)
- ✅ Full control over compilation flags and optimization
- ✅ Tight integration with SAGE middleware
- ✅ Can customize C++ code for SAGE-specific needs

**Update process:**
```bash
# In sageDB repo - push changes
cd ~/sageDB
git push origin main

# In SAGE repo - update submodule or copy files
cd ~/SAGE/packages/sage-middleware/src/sage/middleware/components/sage_db/
# ... update files ...
```

---

### 2. PyPI Package (Standalone Use)
**Used by:** External projects, quick prototyping, independent tools

**Installation:** `pip install sagedb`

**How it works:**
- Source distribution uploaded to PyPI
- Users compile from source during `pip install`
- C++ code built automatically via scikit-build-core
- Imported as: `import sagedb` or `from sagedb import SageDB`

**Advantages:**
- ✅ Easy installation for external users
- ✅ Semantic versioning and release tracking
- ✅ Can be used independently of SAGE
- ✅ Standard Python packaging workflow

**Update process:**
```bash
# Bump version in __init__.py
vim __init__.py  # __version__ = "0.1.1"

# Commit and push (pre-push hook handles PyPI upload)
git add __init__.py
git commit -m "Bump version to 0.1.1"
git push origin main  # Hook asks to upload to PyPI
```

---

## When to Use Which?

### Use Embedded Component When:
- Developing SAGE middleware features
- Need tight integration with SAGE internals
- Want to customize C++ implementation
- Building SAGE from source anyway

### Use PyPI Package When:
- Using sageDB in a standalone Python project
- Quick prototyping without SAGE framework
- Teaching/demo scenarios
- Contributing example scripts or tutorials
- Other projects depend on sageDB functionality

---

## Version Synchronization

Both deployment modes should maintain **the same version number** for consistency:

| Location | File | Current Version |
|----------|------|----------------|
| PyPI Package | `sageDB/__init__.py` | 0.1.0 |
| SAGE Component | `sage/middleware/components/sage_db/__init__.py` | 0.1.0 |
| SAGE Component | `sage/middleware/components/sage_db/python/__init__.py` | 0.1.0 |

**To update versions:**
```bash
# 1. Update sageDB standalone
cd ~/sageDB
vim __init__.py  # Change version
git commit -am "Bump to 0.1.1"
git push  # Pre-push hook uploads to PyPI

# 2. Update SAGE embedded
cd ~/SAGE
sed -i 's/__version__ = "0.1.0"/__version__ = "0.1.1"/g' \
  packages/sage-middleware/src/sage/middleware/components/sage_db/__init__.py \
  packages/sage-middleware/src/sage/middleware/components/sage_db/python/__init__.py
git commit -am "Update sageDB to 0.1.1"
```

---

## Why Both?

**No Conflict:** These are complementary, not competing:
- PyPI package enables **ecosystem growth** (external tools can depend on sageDB)
- Embedded component ensures **SAGE independence** (no external dependency for core features)
- Users choose based on their use case

**Example Use Cases:**

1. **External Researcher** wants to benchmark ANNS algorithms:
   ```bash
   pip install sagedb  # Quick setup
   ```

2. **SAGE Developer** adding multimodal fusion:
   ```python
   from sage.middleware.components.sage_db import MultimodalSageDB
   # Already available, no pip install needed
   ```

3. **Tutorial Author** writing sageDB examples:
   ```bash
   pip install sagedb  # Readers can easily reproduce
   ```

---

## Technical Notes

### Build Differences
- **Embedded:** Uses SAGE's CMake configuration, may have SAGE-specific flags
- **PyPI:** Uses `pyproject.toml` + scikit-build-core, generic build

### Import Paths
- **Embedded:** `sage.middleware.components.sage_db.*`
- **PyPI:** `sagedb.*`

### Dependencies
- **Embedded:** Inherits from SAGE's dependency tree
- **PyPI:** Minimal dependencies (numpy, optional faiss-cpu)

---

## Future Considerations

As sageDB matures, consider:
- Providing **pre-built wheels** (manylinux) to speed up pip installation
- Adding **conda-forge** package for scientific computing users
- Creating **Docker images** with sageDB pre-installed
- Publishing to **other language package managers** (if bindings added)

All while maintaining the embedded component for SAGE's internal use.

---

## ✅ CORRECT: PyPI Dependency Model (Current)

**As of 2026-01-04**, sageDB follows standard Python packaging:

```toml
# In SAGE's sage-middleware/pyproject.toml
dependencies = [
    "sagedb>=0.1.0,<0.2.0",
    ...
]
```

```python
# In SAGE code
from sagedb import SageDB, DatabaseConfig
```

**No more:**
- ❌ Copying sageDB source into SAGE repo
- ❌ Building C++ extensions within SAGE
- ❌ Manual version synchronization
- ❌ Code duplication

**Workflow:**
1. Develop sageDB independently
2. Release to PyPI when ready
3. SAGE updates dependency version
4. Users get new features via `pip install --upgrade sagedb`
