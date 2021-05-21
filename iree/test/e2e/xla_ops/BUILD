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
# See https://github.com/google/iree/blob/main/docs/developers/developing_iree/testing_guide.md#iree-core-end-to-end-tests.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_cuda",
    srcs = glob(
        ["*.mlir"],
        exclude = [
            "concatenate.mlir",
            "dot.mlir",
            "dot_general.mlir",
            "reduce.mlir",
            "reduce_window.mlir",
            "remainder.mlir",  # frem has a bug in LLVM PTX backend for negative values.
            "round.mlir",
            "sort.mlir",
        ],
    ),
    driver = "cuda",
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backend = "cuda",
)

iree_check_single_backend_test_suite(
    name = "check_dylib-llvm-aot_dylib",
    srcs = glob(
        ["*.mlir"],
        exclude = [
            "round.mlir",
            "sort.mlir",
        ],
    ),
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_vmvx_vmvx",
    srcs = glob(
        ["*.mlir"],
        exclude = [
            "round.mlir",
            "sort.mlir",
        ],
    ),
    driver = "vmvx",
    target_backend = "vmvx",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = glob(
        ["*.mlir"],
        exclude = [
            "fft.mlir",  # TODO(GH-5444): Enable the test.
            "round.mlir",
            "sort.mlir",
        ],
    ),
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

test_suite(
    name = "check",
    tests = [
        ":check_dylib-llvm-aot_dylib",
        ":check_vmvx_vmvx",
        ":check_vulkan-spirv_vulkan",
    ],
)
