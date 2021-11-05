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
    name = "libmmt4d",
    srcs = [
        "libmmt4d_aarch64_dotprod.bc",
        "libmmt4d_aarch64_generic.bc",
        "libmmt4d_wasm32_generic.bc",
        "libmmt4d_wasm64_generic.bc",
    ],
    c_file_output = "libmmt4d.c",
    flatten = True,
    h_file_output = "libmmt4d.h",
    identifier = "iree_builtins_libmmt4d",
)
