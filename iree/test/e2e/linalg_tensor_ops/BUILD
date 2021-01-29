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

# Tests of end-to-end IREE support for individual ops in the XLA HLO dialect.
# Each test file should have a name matching the corresponding XLA HLO op and test only the
# functionality of that op (though may make use of other ops where necessary). Tests should be
# written using the IREE Check framework and should always pass on the reference VMLA backend.
# See https://google.github.io/iree/TestingGuide#iree-core-end-to-end-tests.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_llvm-ir_llvm",
    srcs = glob(["*.mlir"]),
    compiler_flags = [
        "-iree-flow-dispatch-linalg-on-tensors",
        "-iree-codegen-llvm-experimental-linalg-on-tensors",
    ],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = ["add.mlir"],  #glob(["*.mlir"]),
    compiler_flags = [
        "-iree-flow-dispatch-linalg-on-tensors",
        "-iree-codegen-spirv-experimental-linalg-on-tensors",
    ],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

test_suite(
    name = "check",
    tests = [
        ":check_llvm-ir_llvm",
        ":check_vulkan-spirv_vulkan",
    ],
)
