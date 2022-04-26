# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_HAL_DRIVER_VMVX} OR NOT ${IREE_TARGET_BACKEND_VMVX})
  return()
endif()
""",
    inline = True,
)

iree_bytecode_module(
    name = "simple_mul_module",
    src = "simple_mul.mlir",
    c_identifier = "iree_runtime_testdata_simple_mul_module",
    flags = [
        "-iree-input-type=mhlo",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=vmvx",
    ],
)
