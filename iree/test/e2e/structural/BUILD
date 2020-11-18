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

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(#2395): Enable all the tests for both LLVM-ir and SPIR-V.
iree_check_single_backend_test_suite(
    name = "check_vmla_vmla",
    srcs = glob(["*.mlir"]),
    driver = "vmla",
    target_backend = "vmla",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = [
        "slice_add.mlir",
    ],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

# TODO(ataei): Enable dylib-llvm-aot tests.
# See: https://github.com/google/iree/issues/2645
iree_check_single_backend_test_suite(
    name = "check_llvm-ir_llvm",
    srcs = [
        "slice_add.mlir",
    ],
    driver = "llvm",
    target_backend = "llvm-ir",
)

test_suite(
    name = "check",
    tests = [
        ":check_llvm-ir_llvm",
        ":check_vmla_vmla",
        ":check_vulkan-spirv_vulkan",
    ],
)
