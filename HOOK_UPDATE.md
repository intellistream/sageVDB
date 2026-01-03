# Git Hook Update

The pre-push hook has been updated with the following improvements:

## New Features

1. **Smart Version Detection**: Checks last 5 commits for version updates
2. **Interactive Version Update**: Allows updating version directly when prompted
3. **Better User Experience**: Clear prompts with colored output
4. **Automatic PyPI Upload**: Uses sage-pypi-publisher for building and uploading

## User Options When Version Not Updated

- `[u]` Update version now and retry push
- `[y]` Continue pushing without version update  
- `[n]` Cancel push

## User Options When Version Was Updated

- `[y]` Build and upload to PyPI, then push
- `[n]` Skip PyPI upload, just push to GitHub
- `[c]` Cancel push

The hook automatically detects C/C++ extensions and builds manylinux wheels.
