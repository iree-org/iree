# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_bytecode_module(
    name = "add_dynamic",
    testonly = True,
    src = "add_dynamic.mlir",
    c_identifier = "iree_tflite_testdata_add_dynamic",
    flags = [
        "--iree-native-bindings-support=false",
        "--iree-tflite-bindings-support",
        "--iree-hal-target-backends=vmvx",
    ],
)

iree_bytecode_module(
    name = "add_multi",
    testonly = True,
    src = "add_multi.mlir",
    c_identifier = "iree_tflite_testdata_add_multi",
    flags = [
        "--iree-native-bindings-support=false",
        "--iree-tflite-bindings-support",
        "--iree-hal-target-backends=vmvx",
    ],
)

iree_bytecode_module(
    name = "add_static",
    testonly = True,
    src = "add_static.mlir",
    c_identifier = "iree_tflite_testdata_add_static",
    flags = [
        "--iree-native-bindings-support=false",
        "--iree-tflite-bindings-support",
        "--iree-hal-target-backends=vmvx",
    ],
)
