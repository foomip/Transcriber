#!/usr/bin/env bash
# install_ctranslate2_rocm.sh — install a ROCm/HIP-enabled CTranslate2 build.
#
# Run this AFTER the normal project setup:
#
#   python3 -m venv whisper_env
#   source whisper_env/bin/activate
#   pip install -r requirements.txt
#   bash scripts/install_ctranslate2_rocm.sh
#
# What it does
# ------------
# There are no pre-built CTranslate2 ROCm/HIP wheels on PyPI or any wheel index.
# The only way to get GPU transcription with Faster-Whisper on AMD is to build
# the C++ library from source and then compile the Python wrapper against it.
# This script automates that two-stage build.
#
# ROCm device string note
# -----------------------
# CTranslate2 ROCm builds still use device="cuda" in the Python API, which is
# why Faster-Whisper works without any additional changes: detect_device() in
# lib/transcription.py returns ("cuda", <compute_type>) for both NVIDIA and AMD
# GPUs when a GPU is detected.
#
# Troubleshooting: RDNA2 allocator quirk
# ---------------------------------------
# Some RDNA2 cards (RX 6000-series) report illegal memory access errors when
# CTranslate2 loads a model.  If you hit this, try:
#
#   export CT2_CUDA_ALLOCATOR=cub_caching
#   python transcribe.py <audio.wav>
#
# You can add that export to your .envrc to make it permanent for the project.

set -euo pipefail

# ---------------------------------------------------------------------------
# Check that we are inside a virtual environment.
# ---------------------------------------------------------------------------

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "⚠️  No active virtual environment detected." >&2
    echo "   Activate one first:  source whisper_env/bin/activate" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Detect ROCm GPU architecture (gfx target).
# ---------------------------------------------------------------------------

HIP_ARCH="${HIP_ARCH:-}"
if [ -z "$HIP_ARCH" ] && command -v rocminfo >/dev/null 2>&1; then
    HIP_ARCH="$(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]*gfx[0-9a-z]+/ { print $2 }' | sort -u | paste -sd';' -)"
fi

if [ -n "$HIP_ARCH" ]; then
    echo "   HIP architectures: ${HIP_ARCH}"
else
    echo "⚠️  Could not detect GPU architecture from rocminfo."
    echo "   Build will use the default set."
    HIP_ARCH="gfx1030"
fi

# ---------------------------------------------------------------------------
# Build directory (outside the venv).
# ---------------------------------------------------------------------------

CTR2_BUILD_DIR="${HOME}/.cache/ctranslate2-build"
INSTALL_PREFIX="${VIRTUAL_ENV}/opt/ctranslate2"
CTR2_VERSION="${CTR2_VERSION:-v4.7.1}"

echo "▶ Building CTranslate2 ${CTR2_VERSION} for ROCm/HIP"
echo "   Source tag   : ${CTR2_VERSION}"
echo "   Build dir    : ${CTR2_BUILD_DIR}"
echo "   Install to   : ${INSTALL_PREFIX}"
echo ""

# ---------------------------------------------------------------------------
# Prerequisites check.
# ---------------------------------------------------------------------------

for cmd in git cmake amdclang amdclang++; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "❌  Missing required tool: ${cmd}" >&2
        echo "    On Ubuntu/Debian: sudo apt install git cmake" >&2
        echo "    amdclang is part of the ROCm installation." >&2
        exit 1
    fi
done

if ! ldconfig -p 2>/dev/null | grep -q libopenblas; then
    echo "⚠️  libopenblas not found in ldconfig cache." >&2
    echo "    Install it:  sudo apt install libopenblas-dev" >&2
fi

# ---------------------------------------------------------------------------
# Clone or update the CTranslate2 source.
# ---------------------------------------------------------------------------

rm -rf "${CTR2_BUILD_DIR}"
mkdir -p "${CTR2_BUILD_DIR}"

echo "▶ Cloning CTranslate2 ${CTR2_VERSION}..."
git clone --branch "${CTR2_VERSION}" --depth 1 --recurse-submodules \
    https://github.com/OpenNMT/CTranslate2.git \
    "${CTR2_BUILD_DIR}"

cd "${CTR2_BUILD_DIR}"

# ---------------------------------------------------------------------------
# Build the C++ library.
# ---------------------------------------------------------------------------

echo ""
echo "▶ Configuring CTranslate2 with -DWITH_HIP=ON..."
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_C_COMPILER=amdclang \
    -DCMAKE_CXX_COMPILER=amdclang++ \
    -DWITH_HIP=ON \
    -DWITH_MKL=OFF \
    -DWITH_DNNL=OFF \
    -DWITH_CUDNN=OFF \
    -DWITH_OPENBLAS=ON \
    -DOPENMP_RUNTIME=COMP \
    -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CLI=OFF \
    -DBUILD_TESTS=OFF

echo ""
echo "▶ Building CTranslate2 (this may take a while)..."
cmake --build build --parallel "$(nproc)"

echo ""
echo "▶ Installing CTranslate2 to ${INSTALL_PREFIX}..."
cmake --install build
sudo ldconfig 2>/dev/null || true

# ---------------------------------------------------------------------------
# Build and install the Python wrapper.
# ---------------------------------------------------------------------------

export CTRANSLATE2_ROOT="${INSTALL_PREFIX}"

echo ""
echo "▶ Building Python wrapper..."
cd python
"${VIRTUAL_ENV}/bin/pip" install -r install_requirements.txt
python setup.py bdist_wheel
"${VIRTUAL_ENV}/bin/pip" install --no-cache-dir dist/*.whl

# ---------------------------------------------------------------------------
# Write an activation hook so LD_LIBRARY_PATH is set automatically.
# ---------------------------------------------------------------------------

HOOK_FILE="${VIRTUAL_ENV}/bin/activate_ctranslate2_rocm"
cat > "${HOOK_FILE}" << 'SCRIPT_EOF'
# Auto-generated by scripts/install_ctranslate2_rocm.sh
CTR2_LIB="${VIRTUAL_ENV}/opt/ctranslate2/lib"
if [ -d "${CTR2_LIB}" ]; then
    export LD_LIBRARY_PATH="${CTR2_LIB}:${LD_LIBRARY_PATH}"
fi
SCRIPT_EOF

# Append a source call to the venv activate script if not already present.
ACTIVATE_SCRIPT="${VIRTUAL_ENV}/bin/activate"
if ! grep -q "activate_ctranslate2_rocm" "${ACTIVATE_SCRIPT}" 2>/dev/null; then
    echo "" >> "${ACTIVATE_SCRIPT}"
    echo "# Added by install_ctranslate2_rocm.sh" >> "${ACTIVATE_SCRIPT}"
    echo "source \"\${VIRTUAL_ENV:-\$(dirname \$(dirname \$(readlink -f \$0)))}\"/bin/activate_ctranslate2_rocm" >> "${ACTIVATE_SCRIPT}"
fi

# ---------------------------------------------------------------------------
# Smoke test.
# ---------------------------------------------------------------------------

echo ""
echo "▶ Smoke-testing CTranslate2 GPU visibility..."

LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
"${VIRTUAL_ENV}/bin/python" - <<'PYEOF'
import ctranslate2, sys

count_fn = getattr(ctranslate2, "get_cuda_device_count", None)
supported_fn = getattr(ctranslate2, "get_supported_compute_types", None)

if count_fn is None:
    print("  ⚠️  get_cuda_device_count not available — very old CTranslate2 build?")
    sys.exit(0)

count = count_fn()
print(f"  Visible CUDA/ROCm devices : {count}")

if supported_fn and count > 0:
    types = supported_fn("cuda")
    print(f"  Supported compute types   : {types}")
elif count == 0:
    print("  ℹ️  No GPU visible to CTranslate2.")
    print("     Verify ROCm is working:  rocminfo | grep -i gfx")
PYEOF

echo ""
echo "✅  Done.  If count > 0 above, AMD GPU transcription should work."
echo "   If count = 0, check your ROCm installation and kernel module setup."
