# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

c_embed_data(
    name = "fill_unaligned_spv",
    srcs = ["fill_unaligned.spv"],
    c_file_output = "fill_unaligned_spv.c",
    flatten = True,
    h_file_output = "fill_unaligned_spv.h",
)
