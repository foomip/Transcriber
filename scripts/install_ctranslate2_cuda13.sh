#!/usr/bin/env bash
# install_ctranslate2_cuda13.sh — install a CUDA 13-enabled CTranslate2 build.
#
# Run this AFTER the normal project setup:
#
#   python3 -m venv whisper_env
#   source whisper_env/bin/activate
#   pip install -r requirements.txt
#   bash scripts/install_ctranslate2_cuda13.sh
#
# Why
# ---
# PyPI CTranslate2 wheels (including 4.8.2) are built against CUDA 12 and
# hard-require libcublas.so.12. On hosts with a CUDA 13-only driver/runtime
# (e.g. NVIDIA driver 580.x), GPU transcription fails with:
#
#   RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
#
# This script assembles a venv-local CUDA 13 toolkit from NVIDIA's pip
# packages (no system CUDA install, no sudo required), builds CTranslate2
# from source against it, and installs the Python wrapper. The built
# library links libcublas.so.13.
#
# What it does
# ------------
#   1. Pins the CUDA 13 toolchain packages into the active venv
#   2. Adds the layout symlinks nvcc / FindCUDA expect (lib64, libcublas.so)
#   3. Builds CTranslate2 ${CTR2_VERSION} with -DWITH_CUDA=ON
#   4. Installs it to $VIRTUAL_ENV/opt/ctranslate2
#   5. Builds + installs the Python wrapper wheel
#   6. Patches RPATHs so bare `whisper_env/bin/python` needs no env vars
#   7. Appends an LD_LIBRARY_PATH hook to the venv activate script
#   8. Smoke-tests GPU visibility (and transcription if ffmpeg is present)
#
# Re-running the script is safe: it rebuilds and reinstalls the same pins.

set -euo pipefail

# CDPATH makes `cd` print the target directory, which corrupts command
# substitutions like $(cd ... && pwd). Neutralise it for this script.
unset CDPATH

# ---------------------------------------------------------------------------
# Check that we are inside a virtual environment.
# ---------------------------------------------------------------------------

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "⚠️  No active virtual environment detected." >&2
    echo "   Activate one first:  source whisper_env/bin/activate" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Pinned versions (verified together on 2026-09-10, RTX 3060 / driver 580).
# ---------------------------------------------------------------------------

CTR2_VERSION="${CTR2_VERSION:-v4.8.2}"
CMAKE_PIN="cmake==3.31.10"
NVCC_PIN="nvidia-cuda-nvcc==13.0.88"
NVVM_PIN="nvidia-nvvm==13.0.88"
CRT_PIN="nvidia-cuda-crt==13.0.48"
CCCL_PIN="nvidia-cuda-cccl==13.0.85"
RUNTIME_PIN="nvidia-cuda-runtime==13.0.96"
CUBLAS_PIN="nvidia-cublas==13.1.1.3"

CTR2_BUILD_DIR="${HOME}/.cache/ctranslate2-build"
INSTALL_PREFIX="${VIRTUAL_ENV}/opt/ctranslate2"
SITE="$(ls -d "${VIRTUAL_ENV}"/lib/python*/site-packages 2>/dev/null | head -1 || true)"
CU13_ROOT="${SITE}/nvidia/cu13"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${SITE}" ] || [ ! -d "${CU13_ROOT}" ]; then
    echo "❌  Could not locate the venv site-packages / nvidia/cu13 layout." >&2
    exit 1
fi

echo "▶ CTranslate2 ${CTR2_VERSION} for CUDA 13"
echo "   Install to   : ${INSTALL_PREFIX}"
echo "   CUDA toolkit : ${CU13_ROOT} (venv-local pip packages)"
echo ""

# ---------------------------------------------------------------------------
# Prerequisites check.
# ---------------------------------------------------------------------------

if ! command -v git >/dev/null 2>&1; then
    echo "❌  Missing required tool: git" >&2
    echo "    On Ubuntu/Debian: sudo apt install git" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Pin the CUDA 13 toolchain into the venv.
# ---------------------------------------------------------------------------

echo "▶ Installing CUDA 13 toolchain packages (pinned)..."
"${VIRTUAL_ENV}/bin/pip" install --quiet \
    "${CMAKE_PIN}" "${NVCC_PIN}" "${NVVM_PIN}" "${CRT_PIN}" \
    "${CCCL_PIN}" "${RUNTIME_PIN}" "${CUBLAS_PIN}"

# ---------------------------------------------------------------------------
# 2. Layout symlinks expected by nvcc / FindCUDA.
# ---------------------------------------------------------------------------

echo "▶ Fixing toolkit layout symlinks..."
ln -sfn lib "${CU13_ROOT}/lib64"
ln -sfn libcublas.so.13 "${CU13_ROOT}/lib/libcublas.so"

# ---------------------------------------------------------------------------
# 3. Clone the CTranslate2 source.
# ---------------------------------------------------------------------------

rm -rf "${CTR2_BUILD_DIR}"
mkdir -p "${CTR2_BUILD_DIR}"

echo "▶ Cloning CTranslate2 ${CTR2_VERSION}..."
git clone --branch "${CTR2_VERSION}" --depth 1 --recurse-submodules \
    --shallow-submodules https://github.com/OpenNMT/CTranslate2.git \
    "${CTR2_BUILD_DIR}"

# ---------------------------------------------------------------------------
# 4. Configure + build the C++ library.
# ---------------------------------------------------------------------------

export PATH="${VIRTUAL_ENV}/bin:${CU13_ROOT}/bin:${PATH}"

echo ""
echo "▶ Configuring CTranslate2 with -DWITH_CUDA=ON..."
cmake -S "${CTR2_BUILD_DIR}" -B "${CTR2_BUILD_DIR}/build" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=OFF \
    -DWITH_MKL=OFF \
    -DWITH_DNNL=OFF \
    -DWITH_RUY=ON \
    -DOPENMP_RUNTIME=COMP \
    -DCUDA_TOOLKIT_ROOT_DIR="${CU13_ROOT}" \
    -DBUILD_CLI=OFF \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release

echo ""
echo "▶ Building CTranslate2 (this may take a while)..."
cmake --build "${CTR2_BUILD_DIR}/build" --parallel "$(nproc)"

echo ""
echo "▶ Installing CTranslate2 to ${INSTALL_PREFIX}..."
cmake --install "${CTR2_BUILD_DIR}/build"

# ---------------------------------------------------------------------------
# 5. Build and install the Python wrapper.
# ---------------------------------------------------------------------------

echo ""
echo "▶ Building Python wrapper..."
cd "${CTR2_BUILD_DIR}/python"
"${VIRTUAL_ENV}/bin/pip" install --quiet -r install_requirements.txt
CTRANSLATE2_ROOT="${INSTALL_PREFIX}" \
    "${VIRTUAL_ENV}/bin/python" setup.py bdist_wheel
"${VIRTUAL_ENV}/bin/pip" install --quiet --force-reinstall --no-deps \
    dist/ctranslate2-*.whl

# Leave the build tree: its python/ directory contains a ctranslate2/ source
# package that would shadow the installed one for any relative imports.
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# 6. Patch RPATHs so the venv works with a bare interpreter (no env vars).
# ---------------------------------------------------------------------------

echo ""
echo "▶ Patching RPATHs (patchelf)..."
"${VIRTUAL_ENV}/bin/pip" install --quiet patchelf
PATCHELF="${VIRTUAL_ENV}/bin/patchelf"

REAL_LIB="$(readlink -f "${INSTALL_PREFIX}/lib/libctranslate2.so")"
"${PATCHELF}" --set-rpath "${CU13_ROOT}/lib" "${REAL_LIB}"

EXT_SO=""
for candidate in "${SITE}"/ctranslate2/_ext*.so; do
    if [ -f "${candidate}" ]; then
        EXT_SO="${candidate}"
        break
    fi
done
if [ -z "${EXT_SO}" ]; then
    echo "❌  Could not find the ctranslate2 _ext shared object." >&2
    exit 1
fi
"${PATCHELF}" --set-rpath "\$ORIGIN/../ctranslate2.libs:${INSTALL_PREFIX}/lib" "${EXT_SO}"

# ---------------------------------------------------------------------------
# 7. Write an activation hook so LD_LIBRARY_PATH is set automatically.
# ---------------------------------------------------------------------------

HOOK_FILE="${VIRTUAL_ENV}/bin/activate_ctranslate2_cuda13"
cat > "${HOOK_FILE}" << 'SCRIPT_EOF'
# Auto-generated by scripts/install_ctranslate2_cuda13.sh
_CTR2_SITE="$(ls -d "${VIRTUAL_ENV}"/lib/python*/site-packages 2>/dev/null | head -1)"
_CTR2_CU13_LIB="${_CTR2_SITE}/nvidia/cu13/lib"
_CTR2_PREFIX_LIB="${VIRTUAL_ENV}/opt/ctranslate2/lib"
for _dir in "${_CTR2_PREFIX_LIB}" "${_CTR2_CU13_LIB}"; do
    if [ -d "${_dir}" ] && [[ ":${LD_LIBRARY_PATH:-}:" != *":${_dir}:"* ]]; then
        LD_LIBRARY_PATH="${_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done
export LD_LIBRARY_PATH
unset _CTR2_SITE _CTR2_CU13_LIB _CTR2_PREFIX_LIB _dir
SCRIPT_EOF

ACTIVATE_SCRIPT="${VIRTUAL_ENV}/bin/activate"
if ! grep -q "activate_ctranslate2_cuda13" "${ACTIVATE_SCRIPT}" 2>/dev/null; then
    echo "" >> "${ACTIVATE_SCRIPT}"
    echo "# Added by install_ctranslate2_cuda13.sh" >> "${ACTIVATE_SCRIPT}"
    echo "source \"\${VIRTUAL_ENV:-\$(dirname \$(dirname \$(readlink -f \$0)))/bin/activate_ctranslate2_cuda13\"" >> "${ACTIVATE_SCRIPT}"
fi

# ---------------------------------------------------------------------------
# 8. Smoke test.
# ---------------------------------------------------------------------------

echo ""
echo "▶ Smoke-testing CTranslate2 GPU visibility..."
"${VIRTUAL_ENV}/bin/python" - <<'PYEOF'
import ctranslate2

count = ctranslate2.get_cuda_device_count()
print(f"  ctranslate2 version     : {ctranslate2.__version__}")
print(f"  Visible CUDA devices    : {count}")
if count > 0:
    print(f"  Supported compute types : {ctranslate2.get_supported_compute_types('cuda')}")
else:
    raise SystemExit("  ❌ No GPU visible to CTranslate2 — check nvidia-smi.")
PYEOF

if command -v ffmpeg >/dev/null 2>&1; then
    echo ""
    echo "▶ Smoke-testing GPU transcription on a synthetic tone..."
    TONE="$(mktemp --suffix=.wav)"
    ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 "${TONE}" >/dev/null 2>&1
    (cd "${PROJECT_ROOT}" && "${VIRTUAL_ENV}/bin/python" - "${TONE}" <<'PYEOF'
import sys
from lib.transcription import transcribe_audio

result = transcribe_audio(sys.argv[1], "cuda", "float16", language="en")
print(f"  ✅ GPU transcription completed ({len(result.lines)} segment(s))")
PYEOF
    )
    rm -f "${TONE}"
fi

echo ""
echo "✅  Done. CUDA 13 GPU transcription should now work."
echo "   Re-run your pipeline:  python transcribe.py <audio.wav>"
