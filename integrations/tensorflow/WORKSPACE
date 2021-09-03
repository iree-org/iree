# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Configures a separate workspace for IREE's TF integrations.

This is siloed to limit TF complexity from leaking into the main project. It
adds the core IREE project and its submodules as repo aliases based on relative
directory paths."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

IREE_PATH = "../.."

#################################### Skylib ####################################
http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

load("@bazel_skylib//lib:paths.bzl", "paths")
################################################################################

##################################### IREE #####################################
# We need a shim here to make this use the version of mlir-hlo present in
# TensorFlow. This shim is just alias rules that forward to the TF rule. Note
# that because configure_iree_submodule_deps uses `maybe` and we define this
# first we will use this definition instead of the one there.
maybe(
    local_repository,
    name = "mlir-hlo",
    path = "build_tools/overlay/mlir-hlo",
)

maybe(
    local_repository,
    name = "iree",
    path = IREE_PATH,
)

load("@iree//build_tools/bazel:workspace.bzl", "configure_iree_submodule_deps")

configure_iree_submodule_deps(
    iree_path = IREE_PATH,
    iree_repo_alias = "@iree",
)

new_local_repository(
    name = "llvm-raw",
    build_file_content = "# empty",
    path = paths.join(IREE_PATH, "third_party/llvm-project"),
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

llvm_configure(name = "llvm-project")

llvm_disable_optional_support_deps()

################################################################################

################################## TensorFlow ##################################
maybe(
    local_repository,
    name = "org_tensorflow",
    path = paths.join(IREE_PATH, "third_party/tensorflow"),
)

# Import all of the tensorflow dependencies. Note that any repository previously
# defined (e.g. from IREE submodules) will be skipped.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()
################################################################################
