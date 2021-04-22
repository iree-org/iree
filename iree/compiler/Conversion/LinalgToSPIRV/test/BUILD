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

# Tests for common transforms.

load("//iree:lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "batch_matmul_vectorization.mlir",
            "concretize_tile_among_workgroups.mlir",
            "concretize_tile_among_workgroups_dynamic.mlir",
            "convert_to_gpu.mlir",
            "convert_to_spirv.mlir",
            "dead_alloc.mlir",
            "elementwise_vectorization.mlir",
            "fold-gpu-procid-uses.mlir",
            "forop_canonicalization.mlir",
            "linalg_tile_and_fuse.mlir",
            "materialize_launch_configuration.mlir",
            "materialize_launch_configuration2.mlir",
            "matmul_fused_vectorization.mlir",
            "matmul_vectorization.mlir",
            "matmul_vectorization_licm.mlir",
            "memref_vecrotization.mlir",
            "pipeline_matmul_vectorization.mlir",
            "pipeline_test_cooperative_mat.mlir",
            "split_dispatch_function.mlir",
            "tile_and_vectorize_conv.mlir",
            "tile_and_vectorize_matmul.mlir",
            "vector_to_gpu.mlir",
            "workgroup_memory_promotion.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)
