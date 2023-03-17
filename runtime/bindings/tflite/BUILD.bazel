# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library", "iree_runtime_cc_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# TFLite C API compatibility shim
#===------------------------------------------------------------------------===#
# See README.md for more information.
# Use this *in place of tflite*; do not link both this and tflite into your
# binary (you'll get ODR violations at the least, if not worse).

iree_runtime_cc_library(
    name = "shim",
    srcs = [
        "interpreter.c",
        "interpreter.h",
        "model.c",
        "model.h",
        "options.c",
        "options.h",
        "shim.c",
        "shim.h",
        "tensor.c",
        "tensor.h",
    ],
    hdrs = [
        "include/tensorflow/lite/c/c_api.h",
        "include/tensorflow/lite/c/c_api_experimental.h",
        "include/tensorflow/lite/c/common.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:synchronization",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/drivers",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm/bytecode:module",
    ],
)

#===------------------------------------------------------------------------===#
# Tests for our own unique functionality and tflite compatibility
#===------------------------------------------------------------------------===#

iree_runtime_cc_test(
    name = "smoke_test",
    srcs = ["smoke_test.cc"],
    # TODO(benvanik): bazel/cmake to include this generated file.
    # Checked in right now until I can figure it out.
    # data = [
    #     "//runtime/bindings/tflite/testdata:add.vmfb",
    # ],
    deps = [
        ":shim",
        "//runtime/bindings/tflite/testdata:add_dynamic_c",
        "//runtime/bindings/tflite/testdata:add_multi_c",
        "//runtime/bindings/tflite/testdata:add_static_c",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)
