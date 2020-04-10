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

# Tests for end-to-end IREE support starting from the XLA HLO dialect.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")
load("//iree:lit_test.bzl", "iree_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

VMLA_CHECK_TESTS = [
    "abs.mlir",
    "reverse.mlir",
]

VULKAN_CHECK_TESTS = [
    "abs.mlir",
    "reverse.mlir",
]

LINALG_TO_SPIRV_VULKAN_CHECK_TESTS = [
    "abs.mlir",
]

LLVMJIT_CHECK_TESTS = [
    "abs.mlir",
]

LEGACY_LIT_TESTS = glob(
    ["*.mlir"],
    exclude =
        VMLA_CHECK_TESTS +
        VULKAN_CHECK_TESTS +
        LINALG_TO_SPIRV_VULKAN_CHECK_TESTS +
        LLVMJIT_CHECK_TESTS,
)

iree_lit_test_suite(
    name = "lit",
    srcs = LEGACY_LIT_TESTS,
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-run-mlir",
    ],
)

iree_check_single_backend_test_suite(
    name = "check_vmla_vmla",
    srcs = VMLA_CHECK_TESTS,
    driver = "vmla",
    target_backend = "vmla",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = VULKAN_CHECK_TESTS,
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_linalg-to-spirv_vulkan",
    srcs = LINALG_TO_SPIRV_VULKAN_CHECK_TESTS,
    compiler_flags = ["-iree-use-linalg-to-spirv-path"],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_llvm-ir_llvm",
    srcs = LLVMJIT_CHECK_TESTS,
    driver = "llvm",
    target_backend = "llvm-ir",
)
