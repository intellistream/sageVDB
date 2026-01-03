#!/bin/bash
# Script to build manylinux wheel for sageDB

set -e

echo "🔧 Building sageDB with manylinux tags..."

# Clean previous builds
rm -rf dist/ wheelhouse/

# Build the wheel
python -m build --wheel

# Check if we have a wheel
WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo "❌ Wheel build failed"
    exit 1
fi

echo "📦 Processing wheel: $WHEEL_FILE"

# Create wheelhouse directory
mkdir -p wheelhouse

# Extract wheel
TEMP_DIR=$(mktemp -d)
python -m wheel unpack "$WHEEL_FILE" -d "$TEMP_DIR"

# Get the unpacked directory
UNPACKED=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d)
cd "$UNPACKED"

# Modify WHEEL file to set platform tag
WHEEL_INFO=$(find . -name "WHEEL" -path "*.dist-info/WHEEL")
if [ -f "$WHEEL_INFO" ]; then
    echo "📝 Updating WHEEL metadata..."
    # Replace any linux_x86_64 or abi3 tags with manylinux and cp311
    sed -i 's/Tag: cp[0-9]*-abi3-linux_x86_64/Tag: cp311-cp311-manylinux_2_34_x86_64/' "$WHEEL_INFO"
    sed -i 's/Tag: cp[0-9]*-cp[0-9]*-linux_x86_64/Tag: cp311-cp311-manylinux_2_34_x86_64/' "$WHEEL_INFO"
    echo "Updated WHEEL file:"
    cat "$WHEEL_INFO"
fi

# Repack the wheel
cd - > /dev/null
# Create the proper filename
BASE_NAME=$(basename "$WHEEL_FILE" .whl)
NEW_WHEEL_NAME=$(echo "$BASE_NAME" | sed -E 's/-(cp[0-9]+)-(abi3|cp[0-9]+)-linux_x86_64/-cp311-cp311-manylinux_2_34_x86_64/').whl
python -m wheel pack "$UNPACKED" -d wheelhouse/

# Rename if necessary
PACKED_WHEEL=$(ls wheelhouse/*.whl)
if [ "$(basename $PACKED_WHEEL)" != "$NEW_WHEEL_NAME" ]; then
    mv "$PACKED_WHEEL" "wheelhouse/$NEW_WHEEL_NAME"
fi

# Clean up
rm -rf "$TEMP_DIR"

echo ""
echo "✅ Created manylinux wheel: wheelhouse/$NEW_WHEEL_NAME"
echo ""
echo "To upload to PyPI, run:"
echo "  python -m twine upload wheelhouse/*.whl"
