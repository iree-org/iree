# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

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
