# Copyright 2022 The IREE Authors
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
    name = "Builtins",
    srcs = [
        "fill_i64.mlir",
        "splat_i64.mlir",
    ],
    c_file_output = "Builtins.c",
    flatten = True,
    h_file_output = "Builtins.h",
    identifier = "iree_compiler_Stream_Builtins",
)
