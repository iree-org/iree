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
    name = "shaders",
    srcs = [
        "fill_buffer.wgsl",
    ],
    c_file_output = "builtin_shaders.c",
    flatten = True,
    h_file_output = "builtin_shaders.h",
    identifier = "iree_hal_wgsl_builtin_shaders",
)
