# llmdev Dev Container
# Base: CUDA 12.8 (REQUIRED for RTX 5060 Ti — Blackwell sm_120 is not supported by CUDA < 12.8).
# `devel` flavor ships nvcc + headers so custom C++/CUDA kernels can be compiled in-container.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG NODE_MAJOR=22
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# ── System toolchain ──────────────────────────────────────────────────────────
# build-essential + cmake + ninja: native Node addons (node-gyp) & custom kernels.
# python3 + pip + venv: fallback bindings (PyTorch interop, HF scripts) later.
# Every apt layer is cleaned in the same RUN to keep the image slim (disk budget).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git git-lfs \
        build-essential cmake ninja-build pkg-config \
        python3 python3-pip python3-venv \
        htop jq unzip \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# ── Node.js 22 (NodeSource) ───────────────────────────────────────────────────
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && npm install -g typescript tsx --no-fund --no-audit \
    && npm cache clean --force

# ── Non-root user (uid 1000 to match the Mint host user) ─────────────────────
# Ubuntu 24.04 images ship a default 'ubuntu' user at uid 1000 — replace it.
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd -g 1000 dev \
    && useradd -m -u 1000 -g 1000 -s /bin/bash dev

# ── Disk policy: pre-create cache/artifact mountpoints with correct ownership ─
RUN mkdir -p /cache/hf /cache/npm /cache/pip /artifacts \
    && chown -R dev:dev /cache /artifacts

USER dev
WORKDIR /workspaces/llmdev

# Belt-and-braces cache redirection even if env vars are stripped.
RUN npm config set cache /cache/npm --global 2>/dev/null || true

CMD ["sleep", "infinity"]
