#!/usr/bin/env python3
"""Verify local data assets declared in data_manifest/s3_assets.json."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geoai_s3_manifest import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(["verify", *sys.argv[1:]]))
