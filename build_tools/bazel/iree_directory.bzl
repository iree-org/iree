# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for bundling files into a directory (TreeArtifact).

Bazel operates on individual files, but some tools require a directory
path rather than individual file paths. iree_directory bridges this gap:
it copies source files into a declared directory, producing a single
TreeArtifact output whose path can be used as a directory reference.

Usage:

    load("//build_tools/bazel:iree_directory.bzl", "iree_directory")

    iree_directory(
        name = "my_data_dir",
        srcs = glob(["*.dat"]),
    )

When used as a dependency in iree_hal_executable's flag_values, the
placeholder resolves to the directory path (since TreeArtifacts produce
a single output whose path is the directory itself).
"""

def _iree_directory_impl(ctx):
    """Copies source files into a declared directory."""
    srcs = ctx.files.srcs
    if not srcs:
        fail("iree_directory requires at least one source file")
    directory = ctx.actions.declare_directory(ctx.attr.name)
    args = ctx.actions.args()
    args.add(directory.path)
    args.add_all(srcs)
    ctx.actions.run_shell(
        inputs = srcs,
        outputs = [directory],
        arguments = [args],
        command = 'dest="$1"; shift; mkdir -p "$dest" && cp "$@" "$dest"',
    )
    return [DefaultInfo(files = depset([directory]))]

iree_directory = rule(
    implementation = _iree_directory_impl,
    attrs = {
        "srcs": attr.label_list(mandatory = True, allow_files = True),
    },
)
