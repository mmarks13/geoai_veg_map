#!/usr/bin/env python3
"""Upload, download, and verify repo data assets from a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_MANIFEST = "data_manifest/s3_assets.json"
DEFAULT_BUFFER_SIZE = 8 * 1024 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open() as handle:
        manifest = json.load(handle)
    if "assets" not in manifest or not isinstance(manifest["assets"], list):
        raise ValueError(f"Manifest {path} must contain an 'assets' list.")
    return manifest


def normalize_s3_root(raw: str | None) -> str:
    s3_root = raw or os.environ.get("GEOAI_DATA_S3_URI")
    if not s3_root:
        raise ValueError(
            "S3 root is required. Set GEOAI_DATA_S3_URI or pass --s3-root."
        )
    if not s3_root.startswith("s3://"):
        raise ValueError(f"S3 root must start with s3://, got: {s3_root}")
    return s3_root.rstrip("/")


def selected_assets(
    assets: list[dict],
    bundles: Iterable[str] | None,
    all_assets: bool,
) -> list[dict]:
    if all_assets:
        return assets
    wanted = set(bundles or ["default"])
    return [
        asset for asset in assets
        if wanted.intersection(set(asset.get("bundles", [])))
    ]


def asset_local_path(root: Path, asset: dict) -> Path:
    local_path = asset.get("local_path")
    if not local_path:
        raise ValueError(f"Manifest asset missing local_path: {asset}")
    path = Path(local_path)
    if path.is_absolute():
        raise ValueError(f"local_path must be repo-relative: {local_path}")
    return root / path


def asset_s3_uri(s3_root: str, asset: dict) -> str:
    key = asset.get("s3_key")
    if not key:
        raise ValueError(f"Manifest asset missing s3_key: {asset}")
    if key.startswith("/") or key.startswith("s3://"):
        raise ValueError(f"s3_key must be relative to the S3 root: {key}")
    return f"{s3_root}/{key}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DEFAULT_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(root: Path, asset: dict, quiet: bool = False) -> bool:
    path = asset_local_path(root, asset)
    expected_size = int(asset["size"])
    expected_hash = asset["sha256"]

    if not path.exists():
        if not quiet:
            print(f"MISS {asset['local_path']}")
        return False

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        if not quiet:
            print(
                f"SIZE {asset['local_path']} expected={expected_size} actual={actual_size}"
            )
        return False

    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        if not quiet:
            print(f"HASH {asset['local_path']} expected={expected_hash} actual={actual_hash}")
        return False

    if not quiet:
        print(f"OK   {asset['local_path']}")
    return True


def require_aws() -> None:
    if shutil.which("aws") is None:
        raise RuntimeError(
            "aws CLI not found on PATH. Activate the project environment or install awscli."
        )


def run_aws(args: list[str], dry_run: bool) -> None:
    print(" ".join(args))
    if dry_run:
        return
    subprocess.run(args, check=True)


def command_list(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    assets = selected_assets(manifest["assets"], args.bundle, args.all)
    total_size = sum(int(asset["size"]) for asset in assets)
    print(f"Manifest: {args.manifest}")
    print(f"Repo: {manifest.get('repo', '<unknown>')}")
    print(f"Assets: {len(assets)}")
    print(f"Bytes: {total_size}")
    print(f"GiB: {total_size / (1024 ** 3):.2f}")
    for asset in assets:
        bundles = ",".join(asset.get("bundles", []))
        print(f"{asset['local_path']} -> {asset['s3_key']} [{bundles}]")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    assets = selected_assets(manifest["assets"], args.bundle, args.all)
    failures = 0
    for asset in assets:
        if not verify_asset(args.root, asset):
            failures += 1
    if failures:
        print(f"FAILED: {failures} asset(s) did not verify.")
        return 1
    print(f"Verified {len(assets)} asset(s).")
    return 0


def command_upload(args: argparse.Namespace) -> int:
    require_aws()
    manifest = load_manifest(args.manifest)
    assets = selected_assets(manifest["assets"], args.bundle, args.all)
    s3_root = normalize_s3_root(args.s3_root)

    for asset in assets:
        local_path = asset_local_path(args.root, asset)
        if not verify_asset(args.root, asset, quiet=True):
            raise RuntimeError(f"Local asset failed verification before upload: {asset['local_path']}")
        source_path = str(local_path.resolve())
        run_aws(
            ["aws", "s3", "cp", "--only-show-errors", source_path, asset_s3_uri(s3_root, asset)],
            args.dry_run,
        )
    print(f"Uploaded {len(assets)} asset(s).")
    return 0


def command_download(args: argparse.Namespace) -> int:
    require_aws()
    manifest = load_manifest(args.manifest)
    assets = selected_assets(manifest["assets"], args.bundle, args.all)
    s3_root = normalize_s3_root(args.s3_root)

    for asset in assets:
        local_path = asset_local_path(args.root, asset)
        if local_path.exists() and not args.force and verify_asset(args.root, asset, quiet=True):
            print(f"SKIP {asset['local_path']}")
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        run_aws(
            ["aws", "s3", "cp", "--only-show-errors", asset_s3_uri(s3_root, asset), str(local_path)],
            args.dry_run,
        )
        if not args.dry_run and not verify_asset(args.root, asset, quiet=True):
            raise RuntimeError(f"Downloaded asset failed verification: {asset['local_path']}")
    print(f"Downloaded {len(assets)} asset(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root() / DEFAULT_MANIFEST,
        help="Manifest JSON path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root(),
        help="Repo root for local_path resolution.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_selection_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--bundle",
            action="append",
            default=None,
            help="Bundle to select; may be repeated. Defaults to 'default' when omitted.",
        )
        subparser.add_argument(
            "--all",
            action="store_true",
            help="Select every asset in the manifest.",
        )

    list_parser = subparsers.add_parser("list", help="List selected manifest assets.")
    add_selection_flags(list_parser)
    list_parser.set_defaults(func=command_list)

    verify_parser = subparsers.add_parser("verify", help="Verify selected local assets.")
    add_selection_flags(verify_parser)
    verify_parser.set_defaults(func=command_verify)

    upload_parser = subparsers.add_parser("upload", help="Upload selected assets to S3.")
    add_selection_flags(upload_parser)
    upload_parser.add_argument("--s3-root", default=None, help="S3 root URI.")
    upload_parser.add_argument("--dry-run", action="store_true", help="Print AWS commands only.")
    upload_parser.set_defaults(func=command_upload)

    download_parser = subparsers.add_parser("download", help="Download selected assets from S3.")
    add_selection_flags(download_parser)
    download_parser.add_argument("--s3-root", default=None, help="S3 root URI.")
    download_parser.add_argument("--dry-run", action="store_true", help="Print AWS commands only.")
    download_parser.add_argument("--force", action="store_true", help="Download even if local file verifies.")
    download_parser.set_defaults(func=command_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
