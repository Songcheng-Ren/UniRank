#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="${SCRIPT_DIR}/KuaiRec.zip"
DOWNLOAD_URL="https://zenodo.org/records/18164998/files/KuaiRec.zip"

if [[ ! -f "${ARCHIVE}" ]]; then
    wget -O "${ARCHIVE}" "${DOWNLOAD_URL}"
fi

unzip -o "${ARCHIVE}" -d "${SCRIPT_DIR}/raw"

echo "KuaiRec downloaded to ${SCRIPT_DIR}/raw"
