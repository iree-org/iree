# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Adds a local dependency on xla."""

local_repository(
    name = "xla",
    path = "../xla",
)

local_repository(
    name = "iree_core",
    path = "../iree",
)

# Import all of the xla dependencies.
load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

# Setup IREE dependencies.
load("@iree_core//build_tools/bazel:workspace.bzl", "configure_iree_submodule_deps", "configure_iree_cuda_deps")

# TODO: Path hard-coding is... not great. Oh bazel.
configure_iree_submodule_deps(
    iree_repo_alias = "@iree_core",
    iree_path = "../iree",
)

configure_iree_cuda_deps()
