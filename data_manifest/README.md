# Data Manifests

This directory contains file-level manifests for data assets that are intentionally
kept out of git. The manifests map repo-relative paths to canonical keys under
`GEOAI_DATA_S3_URI`.

Common commands:

```bash
# Configure AWS first, then:
source /tmp/geoai_s3_env

# See what the default bundle contains.
python scripts/geoai_s3_manifest.py list

# Upload selected assets from this machine.
bash scripts/export_data_to_s3.sh

# Hydrate a fresh clone.
bash scripts/bootstrap_data.sh

# Verify local files by size and SHA256.
python scripts/verify_data_manifest.py
```

Use `--bundle <name>` repeatedly to select additional bundles, or `--all` to
select every manifest asset.
