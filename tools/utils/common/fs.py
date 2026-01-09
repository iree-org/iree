# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Filesystem utilities for safe atomic file operations."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path


def safe_write_text(
    path: Path | str,
    data: str,
    *,
    encoding: str = "utf-8",
    mkdir: bool = True,
    atomic: bool = True,
    backup: bool = False,
) -> None:
    """Write text to file with optional atomic write and backup.

    Args:
        path: Destination file path.
        data: Text content to write.
        encoding: Text encoding (default: utf-8).
        mkdir: Create parent directories if needed (default: True).
        atomic: Use atomic write via temp file (default: True).
        backup: Create .bak file before overwriting (default: False).
    """
    p = Path(path)
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    if not atomic:
        with open(p, "w", encoding=encoding, newline="\n") as f:
            f.write(data)
        return
    # Atomic via write to a temp file in the same dir and rename
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".tmp-", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(data)
        if backup and p.exists():
            p.replace(p.with_suffix(p.suffix + ".bak"))
        os.replace(tmp_name, p)
    except Exception:
        with suppress(Exception):
            os.unlink(tmp_name)
        raise
