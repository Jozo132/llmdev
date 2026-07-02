#!/usr/bin/env bash
# disk-guard.sh — enforce the 100GB total budget. Warns at 80%, fails at 95%.
set -uo pipefail

BUDGET_GB="${LLMDEV_DISK_BUDGET_GB:-100}"

usage_gb() { du -s --block-size=1G "$1" 2>/dev/null | cut -f1; }

paths=("/cache" "/artifacts" "$(pwd)")
total=0
printf "%-40s %8s\n" "PATH" "SIZE(GB)"
for p in "${paths[@]}"; do
  [[ -d "$p" ]] || continue
  sz=$(usage_gb "$p"); sz=${sz:-0}
  printf "%-40s %8s\n" "$p" "$sz"
  total=$((total + sz))
done

echo "──────────────────────────────────────"
echo "TOTAL: ${total}GB / budget ${BUDGET_GB}GB"

warn=$((BUDGET_GB * 80 / 100))
crit=$((BUDGET_GB * 95 / 100))

if (( total >= crit )); then
  echo "CRITICAL: ≥95% of disk budget. Prune now:"
  echo "  docker volume rm llmdev-cache   (safe — caches only)"
  echo "  rm -rf /artifacts/tokens/*.bin  (re-streamable from HF)"
  exit 2
elif (( total >= warn )); then
  echo "WARNING: ≥80% of disk budget consumed."
  exit 1
fi
echo "OK"
