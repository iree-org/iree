# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT} OR NOT ${IREE_TARGET_BACKEND_VULKAN-SPIRV}
   OR NOT ${IREE_HAL_DRIVER_DYLIB} OR NOT ${IREE_HAL_DRIVER_VULKAN})
  return()
endif()
""",
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module",
    src = "simple_embedding_test.mlir",
    cc_namespace = "iree::samples",
    flags = [
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
        "-iree-hal-target-backends=vulkan-spirv",
    ],
)

cc_test(
    name = "simple_embedding_test",
    srcs = ["simple_embedding_test.cc"],
    deps = [
        ":simple_embedding_test_bytecode_module_cc",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/hal:api",
        "//iree/hal/dylib/registration",
        "//iree/hal/vulkan/registration",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
