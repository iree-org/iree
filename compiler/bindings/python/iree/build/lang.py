# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

import argparse
import functools

from iree.build.executor import ClArg, Entrypoint

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


def cl_arg(name: str, *, action=None, default=None, type=None, help=None):
    """Used to define or reference a command-line argument from within actions
    and entry-points.

    Keywords have the same interpretation as `ArgumentParser.add_argument()`.

    Any ClArg set as a default value for an argument to an `entrypoint` will be
    added to the global argument parser. Any particular argument name can only be
    registered once and must not conflict with a built-in command line option.
    The implication of this is that for single-use arguments, the `=cl_arg(...)`
    can just be added as a default argument. Otherwise, for shared arguments,
    it should be created at the module level and referenced.

    When called, any entrypoint arguments that do not have an explicit keyword
    set will get their value from the command line environment.
    """
    if name.startswith("-"):
        raise ValueError("cl_arg name must not be prefixed with dashes")
    dest = name.replace("-", "_")
    return ClArg(name, action=action, default=default, type=type, dest=dest, help=help)
