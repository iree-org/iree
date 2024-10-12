# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

from iree.build.executor import Entrypoint

__all__ = [
    "entrypoint",
]


def entrypoint(f=None, **kwargs):
    """Function decorator to turn it into a build entrypoint."""
    if f is None:
        return functools.partial(target, **kwargs)
    target = Entrypoint(f.__name__, f, **kwargs)
    functools.wraps(target, f)
    return target
