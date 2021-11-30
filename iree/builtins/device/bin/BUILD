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
    name = "libdevice",
    srcs = [
        "libdevice_wasm32_generic.bc",
        "libdevice_wasm64_generic.bc",
    ],
    c_file_output = "libdevice.c",
    flatten = True,
    h_file_output = "libdevice.h",
    identifier = "iree_builtins_libdevice",
)
