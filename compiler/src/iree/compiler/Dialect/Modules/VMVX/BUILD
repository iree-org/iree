# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

load("//iree:build_defs.oss.bzl", "iree_compiler_cc_library")

load("//build_tools/bazel:iree_tablegen.bzl", "iree_tablegen_doc", "iree_gentbl_cc_library", "iree_td_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

c_embed_data(
    name = "vmvx_imports",
    srcs = ["vmvx.imports.mlir"],
    c_file_output = "vmvx.imports.c",
    flatten = True,
    h_file_output = "vmvx.imports.h",
    identifier = "iree_vmvx_imports",
)
