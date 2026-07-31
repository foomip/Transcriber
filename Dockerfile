# syntax=docker/dockerfile:1.7
# x86-64-only runtime, pinned to the current Python 3.14 slim manifest.
ARG PYTHON_IMAGE=python:3.14-slim@sha256:d4fea6e20c09820028eea3f5c17f5b8ebd2ecb9c2bf28e561681a74a96090e4f

FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS python-deps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY requirements-docker.lock /tmp/requirements-docker.lock
RUN grep -v '^llama-cpp-python==' /tmp/requirements-docker.lock > /tmp/requirements-no-llama.lock && \
    python -m pip install --no-cache-dir -r /tmp/requirements-no-llama.lock

FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS cpu-build
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*
COPY requirements-docker.lock /tmp/requirements-docker.lock
RUN LLAMA_VERSION="$(awk -F== '/^llama-cpp-python==/ {print $2}' /tmp/requirements-docker.lock)" && \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    CMAKE_ARGS="-DCMAKE_CXX_COMPILER=/usr/bin/g++ -DGGML_NATIVE=OFF" FORCE_CMAKE=1 \
    python -m pip wheel --no-cache-dir --no-deps --no-binary llama-cpp-python \
        --wheel-dir /wheels "llama-cpp-python==${LLAMA_VERSION}"

FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS vulkan-build
ARG WHISPER_CPP_COMMIT=2ca53bb45e38748d07b310eeb36245a7157ac882
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git pkg-config \
        libvulkan-dev glslc spirv-headers && \
    rm -rf /var/lib/apt/lists/*
COPY requirements-docker.lock /tmp/requirements-docker.lock
RUN LLAMA_VERSION="$(awk -F== '/^llama-cpp-python==/ {print $2}' /tmp/requirements-docker.lock)" && \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    CMAKE_ARGS="-DCMAKE_CXX_COMPILER=/usr/bin/g++ -DGGML_VULKAN=ON -DGGML_NATIVE=OFF" FORCE_CMAKE=1 \
    python -m pip wheel --no-cache-dir --no-deps --no-binary llama-cpp-python \
        --wheel-dir /wheels "llama-cpp-python==${LLAMA_VERSION}"
RUN git clone --filter=blob:none https://github.com/ggml-org/whisper.cpp.git /tmp/whisper.cpp && \
    git -C /tmp/whisper.cpp checkout --detach "${WHISPER_CPP_COMMIT}"
COPY docker/patches/whisper-cli-language-probability.patch /tmp/whisper-language.patch
RUN git -C /tmp/whisper.cpp apply --check /tmp/whisper-language.patch && \
    git -C /tmp/whisper.cpp apply /tmp/whisper-language.patch && \
    cmake -S /tmp/whisper.cpp -B /tmp/whisper.cpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
        -DGGML_VULKAN=ON \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF && \
    cmake --build /tmp/whisper.cpp/build --target whisper-cli && \
    install -Dm755 /tmp/whisper.cpp/build/bin/whisper-cli /artifacts/whisper-cli
COPY docker/vulkan_probe.c /tmp/vulkan_probe.c
RUN cc -O2 -std=c11 -Wall -Wextra -Werror \
        /tmp/vulkan_probe.c -lvulkan -o /artifacts/transcriber-vulkan-probe

FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS runtime-common
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache \
    TRANSCRIBER_GGUF_CACHE_DIR=/cache/transcriber/gguf \
    TRANSCRIBER_WHISPER_CPP_CACHE_DIR=/cache/transcriber/whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg libgomp1 libsndfile1 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=python-deps /usr/local /usr/local
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid "${GROUP_ID}" appgroup && \
    useradd --uid "${USER_ID}" --gid appgroup --create-home appuser && \
    mkdir -p /app /cache/huggingface /cache/transcriber/gguf /cache/transcriber/whisper && \
    chown -R appuser:appgroup /app /cache
WORKDIR /app
COPY --chown=appuser:appgroup . /app
ENTRYPOINT ["python", "transcribe.py"]

FROM runtime-common AS cpu
ENV TRANSCRIBER_RUNTIME_PROFILE=cpu \
    TRANSCRIBER_TRANSCRIPTION_BACKEND=faster_whisper
COPY --from=cpu-build /wheels /tmp/wheels
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels
USER appuser

FROM runtime-common AS vulkan
ENV TRANSCRIBER_RUNTIME_PROFILE=vulkan \
    TRANSCRIBER_TRANSCRIPTION_BACKEND=auto
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libvulkan1 mesa-vulkan-drivers && \
    rm -rf /var/lib/apt/lists/*
COPY --from=vulkan-build /wheels /tmp/wheels
COPY --from=vulkan-build /artifacts/whisper-cli /usr/local/bin/whisper-cli
COPY --from=vulkan-build /artifacts/transcriber-vulkan-probe /usr/local/bin/transcriber-vulkan-probe
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels
USER appuser
