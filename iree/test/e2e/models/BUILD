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

# Tests for end-to-end IREE support of entire models or their close derivatives.

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//iree:lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

CHECK_FRAMEWORK_TESTS = [
    "bert_encoder_unrolled_fake_weights.mlir",
    "mobilenetv2_fake_weights.mlir",
]

iree_lit_test_suite(
    name = "lit",
    size = "medium",
    srcs = enforce_glob(
        [
            "collatz.mlir",
            "edge_detection.mlir",
            "fragment_000.mlir",
            "fullyconnected.mlir",
            "mnist_fake_weights.mlir",
            "resnet_fake_weights.mlir",
            "unidirectional_lstm.mlir",
        ],
        include =
            ["*.mlir"],
        exclude = CHECK_FRAMEWORK_TESTS,
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-run-mlir",
    ],
    tags = [
        "hostonly",
        "optonly",  # swiftshader is too slow in dbg
    ],
)

iree_check_single_backend_test_suite(
    name = "check_vmla_vmla",
    srcs = CHECK_FRAMEWORK_TESTS,
    driver = "vmla",
    target_backend = "vmla",
)

iree_check_single_backend_test_suite(
    name = "check_linalg_on_tensors_vulkan-spirv_vulkan",
    srcs = CHECK_FRAMEWORK_TESTS,
    compiler_flags = [
        "-iree-flow-dispatch-linalg-on-tensors",
        "-iree-flow-inline-constants-max-byte-length=2048",
        "-iree-codegen-spirv-experimental-linalg-on-tensors",
        "-iree-spirv-enable-vectorization",
    ],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_linalg_on_tensors_dylib-llvm-aot_dylib",
    srcs = CHECK_FRAMEWORK_TESTS,
    compiler_flags = [
        "-iree-flow-dispatch-linalg-on-tensors",
        "-iree-flow-inline-constants-max-byte-length=2048",
        "-iree-codegen-llvm-experimental-linalg-on-tensors",
    ],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)
