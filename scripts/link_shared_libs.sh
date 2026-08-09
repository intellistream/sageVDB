#!/usr/bin/env bash
# link_shared_libs.sh - Symlink compiled C-extension .so files from the PyPI
# install (isage-vdb) into the source checkout so that sageVDB can be placed
# on PYTHONPATH for development without triggering a silent ImportError.
#
# Background
# ----------
# The sagevdb package ships a compiled C extension (_sagevdb.cpython-*.so) and
# a native shared library (libsage_vdb.so).  A bare git checkout does NOT
# contain these build artifacts, so when the source directory is on PYTHONPATH,
# native API exports like DatabaseConfig / DistanceMetric are unavailable.
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

ext_suffix=$("$python_bin" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')")
if [[ -z "$ext_suffix" ]]; then
    echo "ERROR: Could not determine Python extension suffix for $python_bin." >&2
    exit 1
fi
echo "Expected extension suffix: $ext_suffix"

# --- Locate installed sagevdb package ----------------------------------------
installed_pkg=$("$python_bin" - "$repo_root" "$src_pkg" <<'PY'
import importlib.util
import os
import sys

repo_root = os.path.abspath(sys.argv[1])
src_pkg = os.path.abspath(sys.argv[2])
source_paths = {repo_root, src_pkg, ""}
sys.path[:] = [
    path for path in sys.path
    if os.path.abspath(path or os.getcwd()) not in source_paths
]

spec = importlib.util.find_spec('sagevdb')
if spec is None or spec.origin is None:
    sys.exit(1)
pkg_dir = os.path.abspath(os.path.dirname(spec.origin))
if pkg_dir == src_pkg or pkg_dir.startswith(repo_root + os.sep):
    sys.exit(2)
print(pkg_dir)
PY
) || {
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
ext_count=0

for so_file in "$installed_pkg"/*.so; do
    [[ -f "$so_file" ]] || continue
    base=$(basename "$so_file")

    if [[ "$base" == _sagevdb*.so && "$base" != *"$ext_suffix" ]]; then
        echo "  [skip]    $base (does not match $ext_suffix)"
        continue
    fi

    if [[ "$base" != "libsage_vdb.so" && "$base" != _sagevdb*.so ]]; then
        echo "  [skip]    $base (not a sageVDB shared library)"
        continue
    fi

    if [[ "$base" == _sagevdb*.so ]]; then
        ((ext_count++)) || true
    fi

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
    echo "WARNING: No compatible sageVDB shared libraries found in $installed_pkg" >&2
    echo "  Expected _sagevdb*$ext_suffix for this interpreter." >&2
    echo "  The isage-vdb wheel may not include compiled extensions for this Python ABI." >&2
    exit 1
fi

if [[ $ext_count -eq 0 ]]; then
    echo "ERROR: No compatible sageVDB Python extension found in $installed_pkg" >&2
    echo "  Expected _sagevdb*$ext_suffix for this interpreter." >&2
    echo "  libsage_vdb.so alone is not enough for 'import sagevdb' native APIs." >&2
    exit 1
fi

echo ""
echo "=== sageVDB shared libraries linked ($so_count file(s)) ==="
echo "Source checkout at $src_pkg can now be used on PYTHONPATH."
