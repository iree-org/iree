#!/usr/bin/env python3

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Upload/download a single blob to/from Azure Blob Storage.

Usage:
  python3 azure_cache.py download /tmp/cache.tar.zst
  python3 azure_cache.py upload /tmp/cache.tar.zst

Required environment variables:
  AZURE_CONNECTION_STRING  Azure Storage connection string.
  AZURE_CONTAINER          Container name (default: ccache-container).
  AZURE_CACHE_BLOB_NAME    Blob name (default: bazel-disk-cache.tar.zst).
"""

import os
import sys

from azure.storage.blob import ContainerClient


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("download", "upload"):
        print(f"Usage: {sys.argv[0]} <download|upload> <local-path>", file=sys.stderr)
        sys.exit(1)

    action, path = sys.argv[1], sys.argv[2]

    conn_str = os.environ.get("AZURE_CONNECTION_STRING", "")
    if not conn_str:
        print("AZURE_CONNECTION_STRING not set, skipping cache", file=sys.stderr)
        sys.exit(0)

    container_name = os.environ.get("AZURE_CONTAINER", "ccache-container")
    blob_name = os.environ.get("AZURE_CACHE_BLOB_NAME", "bazel-disk-cache.tar.zst")

    container = ContainerClient.from_connection_string(conn_str, container_name)

    if action == "download":
        try:
            with open(path, "wb") as f:
                container.download_blob(blob_name).readinto(f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"Downloaded {blob_name} -> {path} ({size_mb:.1f} MB)")
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            print(f"No cache found ({e}), starting fresh")
    elif action == "upload":
        size_mb = os.path.getsize(path) / (1024 * 1024)
        with open(path, "rb") as f:
            container.upload_blob(blob_name, f, overwrite=True)
        print(f"Uploaded {path} -> {blob_name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
