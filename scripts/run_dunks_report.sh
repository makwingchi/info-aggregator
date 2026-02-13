#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/reports"

# Report name uses today's date in Asia/Shanghai; --date is one day before.
REPORT_DATE=$(TZ=Asia/Shanghai date +%Y%m%d)
NY_DATE=$(TZ=Asia/Shanghai date -v-1d +%Y-%m-%d)

mkdir -p "$OUTPUT_DIR"

echo "Report date (Asia/Shanghai): ${REPORT_DATE}"
echo "Data date (one day before): ${NY_DATE}"

poetry run python3 "${ROOT_DIR}/dunks_report.py" \
  --output "${OUTPUT_DIR}/report_${REPORT_DATE}.md" \
  --date "${NY_DATE}"
