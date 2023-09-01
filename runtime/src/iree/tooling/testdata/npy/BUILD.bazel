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
    name = "npy",
    testonly = True,
    srcs = [
        "array_shapes.npy",
        "array_types.npy",
        "empty.npy",
        "multiple.npy",
        "single.npy",
    ],
    c_file_output = "npy_files.c",
    flatten = True,
    h_file_output = "npy_files.h",
    identifier = "iree_numpy_npy_files",
)
