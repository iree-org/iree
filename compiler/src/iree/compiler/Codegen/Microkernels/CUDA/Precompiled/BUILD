# Copyright 2023 The IREE Authors
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
    name = "libuCUDAKernels",
    srcs = [
        "ukernels-cuda-debug-nvptx64-nvidia-cuda-sm_80.ptx",
        "ukernels-cuda-nvptx64-nvidia-cuda-sm_80.bc",
        "ukernels-cuda-nvptx64-nvidia-cuda-sm_80.ptx",
    ],
    c_file_output = "uCUDAkernels.c",
    flatten = True,
    h_file_output = "uCUDAkernels.h",
)
