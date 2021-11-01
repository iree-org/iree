# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")
load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT "${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}" AND NOT "${IREE_TARGET_BACKEND_WASM-LLVM-AOT}")
  return()
endif()
""",
)

c_embed_data(
    name = "librt",
    srcs = ["bin/librt.bc"],
    c_file_output = "librt.c",
    flatten = True,
    h_file_output = "librt.h",
    identifier = "iree_compiler_librt",
)

c_embed_data(
    name = "librt64",
    srcs = ["bin/librt64.bc"],
    c_file_output = "librt64.c",
    flatten = True,
    h_file_output = "librt64.h",
    identifier = "iree_compiler_librt64",
)
