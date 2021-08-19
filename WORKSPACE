# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Workspace file for the IREE project.
# buildozer: disable=positional-args

workspace(name = "iree_core")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

###############################################################################
# Skylib
http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()
###############################################################################

###############################################################################
# llvm-project

new_local_repository(
    name = "llvm-raw",
    build_file_content = "# empty",
    path = "third_party/llvm-project",
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

llvm_configure(name = "llvm-project")

llvm_disable_optional_support_deps()

###############################################################################

###############################################################################
# Find and configure the Vulkan SDK, if installed.
load("//build_tools/third_party/vulkan_sdk:repo.bzl", "vulkan_sdk_setup")

maybe(
    vulkan_sdk_setup,
    name = "vulkan_sdk",
)
###############################################################################
# All other IREE submodule dependencies

load("//build_tools/bazel:workspace.bzl", "configure_iree_submodule_deps")

configure_iree_submodule_deps()

###############################################################################
# bazel toolchains rules for remote execution (https://releases.bazel.build/bazel-toolchains.html).
http_archive(
    name = "bazel_toolchains",
    sha256 = "8c9728dc1bb3e8356b344088dfd10038984be74e1c8d6e92dbb05f21cabbb8e4",
    strip_prefix = "bazel-toolchains-3.7.1",
    urls = [
        "https://github.com/bazelbuild/bazel-toolchains/releases/download/3.7.1/bazel-toolchains-3.7.1.tar.gz",
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/releases/download/3.7.1/bazel-toolchains-3.7.1.tar.gz",
    ],
)

load("@bazel_toolchains//rules:rbe_repo.bzl", "rbe_autoconfig")

rbe_autoconfig(
    name = "rbe_default",
    base_container_digest = "sha256:1a8ed713f40267bb51fe17de012fa631a20c52df818ccb317aaed2ee068dfc61",
    digest = "sha256:62b161e79413f0f59ae3845c377b10e60a4a639f3d32569a82b620f017837a68",
    registry = "gcr.io",
    repository = "iree-oss/rbe-toolchain",
    use_checked_in_confs = "Force",
)

###############################################################################
