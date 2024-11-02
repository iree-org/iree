# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

import argparse
import functools

from iree.build.args import cl_arg  # Export as part of the public API
from iree.build.executor import Entrypoint

__all__ = [
    "cl_arg",
    "entrypoint",
]


def entrypoint(
    f=None,
    *,
    description: str | None = None,
):
    """Function decorator to turn it into a build entrypoint."""
    if f is None:
        return functools.partial(entrypoint, description=description)
    target = Entrypoint(f.__name__, f, description=description)
    functools.wraps(target, f)
    return target
