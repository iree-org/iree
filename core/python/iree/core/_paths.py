# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers for the paths held by configuration and result objects."""

from pathlib import Path


def collect_paths(data: object) -> tuple[Path, ...]:
    """Collect every ``Path`` in a nested structure of dicts, lists and tuples.

    Meant for the result of ``dataclasses.asdict``.
    """
    if isinstance(data, Path):
        return (data,)
    if isinstance(data, dict):
        return tuple(path for value in data.values() for path in collect_paths(value))
    if isinstance(data, (list, tuple)):
        return tuple(path for item in data for path in collect_paths(item))
    return ()
