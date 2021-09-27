# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Hello World!
#===------------------------------------------------------------------------===#

iree_cmake_extra_content(
    content = """
if (NOT ${IREE_HAL_DRIVER_VMVX} OR NOT ${IREE_TARGET_BACKEND_VMVX})
  return()
endif()
""",
    inline = True,
)

cc_binary(
    name = "hello_world_embedded",
    srcs = ["hello_world_explained.c"],
    defines = [
        # Load data directly from memory.
        "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_EMBEDDED_DATA",
    ],
    deps = [
        "//iree/runtime",
        "//iree/runtime/testdata:simple_mul_module_c",
    ],
)

run_binary_test(
    name = "hello_world_embedded_test",
    test_binary = ":hello_world_embedded",
)

cc_binary(
    name = "hello_world_file",
    srcs = ["hello_world_explained.c"],
    defines = [
        # Load data from a file passed on the command line.
        "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_COMMAND_LINE_ARG",
    ],
    deps = [
        "//iree/runtime",
    ],
)

# TODO(benvanik): run_binary_test that passes the file as a flag. Right now we
# can't specify data through run_binary_test, though, so this isn't possible to
# automate.

cc_binary(
    name = "hello_world_terse",
    srcs = ["hello_world_terse.c"],
    deps = [
        "//iree/runtime",
        "//iree/runtime/testdata:simple_mul_module_c",
    ],
)

run_binary_test(
    name = "hello_world_terse_test",
    test_binary = ":hello_world_terse",
)
