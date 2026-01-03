# Pre-Push Hook Usage Guide

The git pre-push hook has been enhanced for better user experience.

## When Version IS Updated

Hook detects version change and asks:
```
✓ Version updated to 0.1.3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Build and upload version 0.1.3 to PyPI?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [y] Yes, build and upload to PyPI
  [n] No, just push to GitHub
  [c] Cancel push
```

## When Version NOT Updated

Interactive version update:
```
⚠  WARNING: Version not updated!
📌 Current version: 0.1.3

What would you like to do?
  [u] Update version now
  [y] Continue without version update
  [n] Cancel push
Your choice [u/y/n]: u

Enter new version (e.g., 0.1.4):
New version: 0.1.4

✓ 0.1.3 → 0.1.4

📦 Build and upload to PyPI? [y/n]: y
```

Hook automatically:
- Updates pyproject.toml
- Commits the change
- Builds manylinux wheel (for C++ extensions)
- Uploads to PyPI
- Pushes to GitHub

## Example Workflow

```bash
# Make changes
git commit -m "feat: add new feature"

# Push (hook will prompt for version)
git push

# Choose [u] and enter new version
# Hook does the rest automatically!
```
