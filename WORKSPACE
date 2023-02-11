# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Adds a local dependency on tensorflow."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

local_repository(
    name = "org_tensorflow",
    path = "external/tensorflow",
)

local_repository(
    name = "iree_core",
    path = "external/iree",
)

# Import all of the tensorflow dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

# Setup IREE dependencies.
load("@iree_core//build_tools/bazel:workspace.bzl", "configure_iree_submodule_deps", "configure_iree_cuda_deps")

# TODO: Path hard-coding is... not great. Oh bazel.
configure_iree_submodule_deps(
    iree_repo_alias = "@iree_core",
    iree_path = "external/iree",
)

configure_iree_cuda_deps()
