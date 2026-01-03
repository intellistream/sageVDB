# PyPI Release Guide for sageDB

## Quick Release Checklist

### 1. Update Version
Edit `__init__.py` and bump the version:
```python
__version__ = "0.1.1"  # Update this
```

### 2. Commit Changes
```bash
git add __init__.py
git commit -m "Bump version to 0.1.1"
```

### 3. Push (Automatic Upload)
```bash
git push origin main
```
The pre-push hook will:
- Detect version change
- Ask if you want to upload to PyPI
- Build and upload automatically if you confirm

### Manual Upload (Alternative)
If you skipped automatic upload or want to upload manually:
```bash
# Clean old builds
rm -rf dist/

# Build source distribution
python -m build --sdist

# Upload to PyPI
python -m twine upload dist/sagedb-0.1.1.tar.gz
```

## Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

Examples:
- `0.1.0` → `0.1.1` (bug fix)
- `0.1.1` → `0.2.0` (new feature)
- `0.9.0` → `1.0.0` (stable release)

## Pre-push Hook Details

The `.git/hooks/pre-push` script:
1. Checks if `__version__` in `__init__.py` was updated
2. If updated: offers to build and upload to PyPI
3. If not updated: warns and asks for confirmation to proceed

## PyPI Token Configuration

Ensure your `~/.pypirc` is configured:
```ini
[pypi]
username = __token__
password = <your-pypi-token>
```

## Testing Before Release

```bash
# Build locally
python -m build --sdist

# Test installation
pip install dist/sagedb-0.1.1.tar.gz

# Run tests
pytest tests/
```

## View Published Package

After upload, visit:
https://pypi.org/project/sagedb/

Install with:
```bash
pip install sagedb
```
