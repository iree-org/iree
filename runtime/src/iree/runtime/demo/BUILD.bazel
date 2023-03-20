# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_runtime_cc_binary")
load("//build_tools/bazel:native_binary.bzl", "native_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Hello World!
#===------------------------------------------------------------------------===#

iree_runtime_cc_binary(
    name = "hello_world_file",
    srcs = ["hello_world_explained.c"],
    defines = [
        # Load data from a file passed on the command line.
        "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_COMMAND_LINE_ARG",
    ],
    deps = [
        "//runtime/src/iree/runtime",
    ],
)

# TODO(benvanik): native_test that passes the file as a flag. Right now we
# can't specify data through native_test, though, so this isn't possible to
# automate.

iree_cmake_extra_content(
    content = """
if(NOT IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE OR NOT IREE_TARGET_BACKEND_VMVX)
  return()
endif()
""",
    inline = True,
)

iree_runtime_cc_binary(
    name = "hello_world_embedded",
    srcs = ["hello_world_explained.c"],
    defines = [
        # Load data directly from memory.
        "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_EMBEDDED_DATA",
    ],
    deps = [
        "//runtime/src/iree/runtime",
        "//runtime/src/iree/runtime/testdata:simple_mul_module_c",
    ],
)

native_test(
    name = "hello_world_embedded_test",
    src = ":hello_world_embedded",
)

iree_runtime_cc_binary(
    name = "hello_world_terse",
    srcs = ["hello_world_terse.c"],
    deps = [
        "//runtime/src/iree/runtime",
        "//runtime/src/iree/runtime/testdata:simple_mul_module_c",
    ],
)

native_test(
    name = "hello_world_terse_test",
    src = ":hello_world_terse",
)
