#!/usr/bin/env bash
# link_shared_libs.sh — Symlink compiled C-extension .so files from the PyPI
# install (isage-vdb) into the source checkout so that sageVDB can be placed
# on PYTHONPATH for development without triggering a silent ImportError.
#
# Background
# ----------
# The sagevdb package ships a compiled C extension (_sagevdb.cpython-*.so) and
# a native shared library (libsage_vdb.so).  A bare git checkout does NOT
# contain these build artefacts, so when the source directory is on PYTHONPATH,
# sagevdb/__init__.py catches the ImportError from the missing .so, sets
# __all__ = [], and exports like DatabaseConfig / DistanceMetric become
# unavailable — a very confusing failure mode.
#
# This script creates symlinks from the installed PyPI package into the source
# tree, giving developers live Python source changes while reusing the
# pre-compiled C extension.  Idempotent: safe to run repeatedly.
#
# Usage
# -----
#   bash scripts/link_shared_libs.sh
#   PYTHON_BIN=/path/to/python bash scripts/link_shared_libs.sh

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
src_pkg="$repo_root/sagevdb"

# --- Locate Python interpreter ------------------------------------------------
resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
        printf '%s\n' "$PYTHON_BIN"; return 0
    fi
    # Prefer the conda env recorded by sage-faculty-twin's marker file.
    local marker="$repo_root/../sage-faculty-twin/.python-bin"
    if [[ -f "$marker" ]]; then
        local candidate
        candidate=$(sed -n '1p' "$marker" | tr -d '\r')
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"; return 0
        fi
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3; return 0
    fi
    if command -v python  >/dev/null 2>&1; then
        command -v python;  return 0
    fi
    return 1
}

python_bin=$(resolve_python) || {
    echo "ERROR: Cannot find a usable Python interpreter." >&2
    echo "  Set PYTHON_BIN=/path/to/python and retry." >&2
    exit 1
}

echo "Using Python: $python_bin ($($python_bin --version 2>&1))"

# --- Locate installed sagevdb package ----------------------------------------
installed_pkg=$("$python_bin" -c "
import importlib.util, os, sys
spec = importlib.util.find_spec('sagevdb')
if spec is None or spec.origin is None:
    sys.exit(1)
print(os.path.dirname(spec.origin))
" 2>/dev/null) || {
    echo "ERROR: sagevdb is not installed in the current Python environment." >&2
    echo "  Install it first:  pip install isage-vdb" >&2
    exit 1
}

echo "Installed sagevdb: $installed_pkg"

if [[ "$installed_pkg" == "$src_pkg" ]]; then
    echo "Installed path matches source path — no symlinking needed."
    exit 0
fi

# --- Symlink .so files --------------------------------------------------------
so_count=0
for so_file in "$installed_pkg"/*.so; do
    [[ -f "$so_file" ]] || continue
    base=$(basename "$so_file")
    target="$src_pkg/$base"
    # Skip if it already points to the right place.
    if [[ -L "$target" && "$(readlink -f "$target")" == "$(readlink -f "$so_file")" ]]; then
        echo "  [ok]      $base -> $so_file"
    else
        ln -sf "$so_file" "$target"
        echo "  [linked]  $base -> $so_file"
    fi
    ((so_count++)) || true
done

if [[ $so_count -eq 0 ]]; then
    echo "WARNING: No .so files found in $installed_pkg" >&2
    echo "  The isage-vdb wheel may not include compiled extensions for this platform." >&2
    exit 1
fi

echo ""
echo "=== sageVDB shared libraries linked ($so_count file(s)) ==="
echo "Source checkout at $src_pkg can now be used on PYTHONPATH."
