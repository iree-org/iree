# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Adds a local dependency on xla."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

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


load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",

    # Replace the commit hash in both places (below) with the latest, rather
    # than using the stale one here.  Even better, set up Renovate and let it do
    # the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/ed994039a951b736091776d677f324b3903ef939.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-ed994039a951b736091776d677f324b3903ef939",

    # When you first run this tool, it'll recommend a sha256 hash to put here
    # with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that
    # a canonical reproducible form can be obtained by modifying arguments
    # sha256 = ..."
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()


###############################################################################
# Additional hacks to appease the XLA/LLVM bazel setup.
###############################################################################

# Taken from the LLVM utils/bazel/WORKSPACE
maybe(
    http_archive,
    name = "llvm_zlib",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
    strip_prefix = "zlib-ng-2.0.7",
    urls = [
        "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
    ],
)
