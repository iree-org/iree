# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A minimal subset of Bazel genrule for common use cases"""

def iree_genrule(
        name,
        srcs,
        outs,
        cmd,
        **kwargs):
    """A minimal subset of Bazel genrule for common use cases.

    Args:
        name: Name of the target.
        srcs: Source files, including any script run in the command.
              Unlike Bazel's genrule, we do not try to distinguish between the
              two. The distinction is needed when tools need to be compiled for
              host, but that doesn't concern us if we only need to run python
              scripts.
        outs: Files generated by the command.
        cmd: The command to be executed. The only supported special Bazel
             genrule syntax is:
               * "$(rootpath x)", which expands to the path to a file in the
                 source tree.
               * "$(execpath x)", which expands to the path to a file in the
                 directory where Bazel runs the build action.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    native.genrule(
        name = name,
        srcs = srcs,
        outs = outs,
        cmd = cmd,
        **kwargs
    )
