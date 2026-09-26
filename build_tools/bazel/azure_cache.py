#!/usr/bin/env python3

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Upload/download/prune a Bazel disk cache blob on Azure Blob Storage.

Usage:
  python3 azure_cache.py download /tmp/cache.tar.gz
  python3 azure_cache.py upload   /tmp/cache.tar.gz
  python3 azure_cache.py prune    /tmp/bazel-disk-cache --max-bytes 10737418240
  python3 azure_cache.py save     /tmp/bazel-disk-cache --max-bytes 10737418240

Commands:
  download   Download the cache blob to a local file.
  upload     Upload a local file to the cache blob.
  prune      Delete oldest-accessed files in a directory until it is under
             --max-bytes (default 10 GiB).  Uses access-time (atime) LRU.
  save       Prune, pack (tar.gz), and upload in one step.

Required environment variables (for download/upload/save):
  AZURE_CONNECTION_STRING  Azure Storage connection string.
  AZURE_CONTAINER          Container name (default: ccache-container).
  AZURE_CACHE_BLOB_NAME    Blob name (default: bazel-disk-cache.tar.gz).
"""

import argparse
import os
import subprocess
import sys

from azure.storage.blob import ContainerClient

DEFAULT_MAX_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB


def _get_container() -> ContainerClient | None:
    conn_str = os.environ.get("AZURE_CONNECTION_STRING", "")
    if not conn_str:
        print("AZURE_CONNECTION_STRING not set, skipping cache", file=sys.stderr)
        return None
    container_name = os.environ.get("AZURE_CONTAINER", "ccache-container")
    return ContainerClient.from_connection_string(conn_str, container_name)


def _blob_name() -> str:
    return os.environ.get("AZURE_CACHE_BLOB_NAME", "bazel-disk-cache.tar.gz")


def download(path: str) -> None:
    container = _get_container()
    if container is None:
        return
    blob = _blob_name()
    try:
        with open(path, "wb") as f:
            container.download_blob(blob).readinto(f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Downloaded {blob} -> {path} ({size_mb:.1f} MB)")
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        print(f"No cache found ({e}), starting fresh")


def upload(path: str) -> None:
    container = _get_container()
    if container is None:
        return
    blob = _blob_name()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    with open(path, "rb") as f:
        container.upload_blob(blob, f, overwrite=True)
    print(f"Uploaded {path} -> {blob} ({size_mb:.1f} MB)")


def prune(cache_dir: str, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
    """Prune *cache_dir* to *max_bytes* using LRU (oldest access time first).

    We can't rely on ``--experimental_disk_cache_gc_max_size`` because that
    requires ``bazel shutdown`` which stalls and sometimes leaves Bazel in a
    bad state.  So we evict manually.
    """
    total = 0
    entries: list[tuple[float, int, str]] = []
    for dirpath, _dirnames, filenames in os.walk(cache_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            total += st.st_size
            entries.append((st.st_atime, st.st_size, fpath))

    if total <= max_bytes:
        print(
            f"Cache size {total / (1024**3):.2f} GiB is within "
            f"{max_bytes / (1024**3):.0f} GiB limit, nothing to prune"
        )
        return

    overshoot = total - max_bytes
    print(
        f"Cache size {total / (1024**3):.2f} GiB exceeds "
        f"{max_bytes / (1024**3):.0f} GiB limit by "
        f"{overshoot / (1024**3):.2f} GiB, pruning..."
    )

    entries.sort(key=lambda e: e[0])
    freed = 0
    removed = 0
    for _atime, size, fpath in entries:
        try:
            os.remove(fpath)
            freed += size
            removed += 1
        except OSError:
            continue
        if freed >= overshoot:
            break

    print(f"Pruned {removed} files, freed {freed / (1024**3):.2f} GiB")


def save(cache_dir: str, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
    """Prune, pack, and upload the cache in one step."""
    prune(cache_dir, max_bytes)

    tarball = "/tmp/cache.tar.gz"
    print(f"Packing {cache_dir} -> {tarball} ...")
    subprocess.check_call(["tar", "-czf", tarball, "-C", cache_dir, "."])

    upload(tarball)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)

    dl = sub.add_parser("download", help="Download cache blob")
    dl.add_argument("path", help="Local file path")

    ul = sub.add_parser("upload", help="Upload cache blob")
    ul.add_argument("path", help="Local file path")

    pr = sub.add_parser("prune", help="Prune cache directory by LRU")
    pr.add_argument("cache_dir", help="Path to the Bazel disk cache directory")
    pr.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Maximum cache size in bytes (default: 10 GiB)",
    )

    sv = sub.add_parser("save", help="Prune + pack + upload")
    sv.add_argument("cache_dir", help="Path to the Bazel disk cache directory")
    sv.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Maximum cache size in bytes (default: 10 GiB)",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.action == "download":
        download(args.path)
    elif args.action == "upload":
        upload(args.path)
    elif args.action == "prune":
        prune(args.cache_dir, args.max_bytes)
    elif args.action == "save":
        save(args.cache_dir, args.max_bytes)


if __name__ == "__main__":
    main()
