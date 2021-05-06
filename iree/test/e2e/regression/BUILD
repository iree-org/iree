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

# Tests for end-to-end IREE support of specific features to prevent regression.
# These should focus on support by IREE itself, not for issues with specific runner tools. Place
# those tests in iree/tools/test/

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//iree:lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

BACKEND_TESTS = [
    "linalg_ops.mlir",
]

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "dynamic_abs.mlir",
            "dynamic_add.mlir",
            "dynamic_compare_and_select.mlir",
            "dynamic_dot.mlir",
            "dynamic_dot_general.mlir",
            "dynamic_torch_index_select_high_rank.mlir",
            "dynamic_torch_index_select_negative.mlir",
            "dynamic_torch_index_select_scalar.mlir",
            "dynamic_torch_index_select_vector.mlir",
            "executable_benchmark.mlir",
            "globals.mlir",
            "scalar.mlir",
            "trace_dispatch_tensors.mlir",
            "unused_args.mlir",
        ],
        include =
            ["*.mlir"],
        # Disabled temporarily. See GH Issue #4733
        exclude = [
            "dynamic_linalg_matmul_on_tensors.mlir",
            "dynamic_linalg_matmul_on_tensors_fuse_0.mlir",
            "dynamic_linalg_matmul_on_tensors_fuse_1.mlir",
            "dynamic_linalg_matmul_on_tensors_fuse_2.mlir",
        ] + BACKEND_TESTS,
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-benchmark-module",
        "//iree/tools:iree-run-mlir",
        "//iree/tools:iree-translate",
    ],
    tags = ["hostonly"],
)

iree_check_single_backend_test_suite(
    name = "check_regression_linalg_ops_dylib-llvm-aot",
    srcs = BACKEND_TESTS,
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_regression_linalg_ops_vulkan-spirv",
    srcs = BACKEND_TESTS,
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)
