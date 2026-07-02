#!/usr/bin/env bash
# Runs once after the dev container is created.
set -euo pipefail

echo "──────────────────────────────────────────────"
echo " llmdev post-create"
echo "──────────────────────────────────────────────"

# Volume-mounted node_modules are owned by root on first creation — fix that.
sudo chown -R dev:dev node_modules webapp/node_modules 2>/dev/null || true

echo "→ Installing backend deps (cache → /cache/npm)"
npm ci --no-fund --no-audit 2>/dev/null || npm install --no-fund --no-audit

echo "→ Installing webapp deps"
npm --prefix webapp install --no-fund --no-audit

echo "→ GPU sanity check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "  WARNING: nvidia-smi not found — GPU passthrough inactive."
  echo "  On the Mint host run: sudo apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

echo "→ Disk budget snapshot"
bash scripts/disk-guard.sh || true

echo "Done. Try:  npm run poc"
