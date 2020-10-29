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

# Tests for end-to-end IREE support specific to the vulkan-spirv lowering.
# TODO(ravishankarm): Reorganize these tests.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_llvm-ir-conv_img2col",
    srcs = [
        "conv.mlir",
    ],
    compiler_flags = ["-iree-codegen-linalg-to-llvm-conv-img2col-conversion"],
    driver = "llvm",
    target_backend = "llvm-ir",
)

iree_check_single_backend_test_suite(
    name = "check_llvm-ir-dot_tile_and_distribute",
    srcs = [
        "dot.mlir",
    ],
    compiler_flags = ["-iree-codegen-linalg-to-llvm-tile-and-distrobute"],
    driver = "llvm",
    target_backend = "llvm-ir",
)
