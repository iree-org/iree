# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_runtime_cc_library")
load("//build_tools/bazel:iree_bitcode_library.bzl", "iree_bitcode_library")
load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Common sources
#===------------------------------------------------------------------------===#

BITCODE_SRCS = [
    "device_generic.c",
]

BITCODE_HDRS = [
    "device.h",
]

#===------------------------------------------------------------------------===#
# Application-integrated static library
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "device",
    srcs = BITCODE_SRCS,
    hdrs = BITCODE_HDRS,
)

#===------------------------------------------------------------------------===#
# Compiler bitcode files
#===------------------------------------------------------------------------===#

iree_cmake_extra_content(
    content = """
if(NOT IREE_BUILD_COMPILER)
  return()
endif()
""",
    inline = True,
)

# TODO(benvanik): rule for building a matrix of bitcode files.
# TODO(benvanik): make some of these flags inside of iree_bitcode_library; maybe
# via an iree_cpu_bitcode_library so that we can have an
# iree_cuda_bitcode_library that can differ.

BITCODE_COPTS = [
    # C17 with no system deps.
    "-std=c17",
    "-nostdinc",
    "-ffreestanding",

    # Optimized and unstamped.
    "-O3",
    "-fno-ident",
    "-fdiscard-value-names",

    # Object file only in bitcode format:
    "-c",
    "-emit-llvm",

    # Force the library into standalone mode (not linking into hosting apps).
    "-DIREE_DEVICE_STANDALONE=1",
]

iree_bitcode_library(
    name = "libdevice_wasm32_generic",
    srcs = BITCODE_SRCS,
    hdrs = BITCODE_HDRS,
    copts = BITCODE_COPTS + ["--target=wasm32"],
)

iree_bitcode_library(
    name = "libdevice_wasm64_generic",
    srcs = BITCODE_SRCS,
    hdrs = BITCODE_HDRS,
    copts = BITCODE_COPTS + ["--target=wasm64"],
)

c_embed_data(
    name = "libdevice_bitcode",
    srcs = [
        ":libdevice_wasm32_generic.bc",
        ":libdevice_wasm64_generic.bc",
    ],
    c_file_output = "libdevice_bitcode.c",
    flatten = True,
    h_file_output = "libdevice_bitcode.h",
    identifier = "iree_builtins_libdevice_bitcode",
    deps = [
        "//runtime/src:runtime_defines",
    ],
)
