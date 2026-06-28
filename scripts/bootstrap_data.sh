#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f /tmp/geoai_s3_env && -z "${GEOAI_DATA_S3_URI:-}" ]]; then
    # Local developer convenience; this file contains no secret values.
    source /tmp/geoai_s3_env
fi

python scripts/geoai_s3_manifest.py download "$@"
