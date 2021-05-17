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
            "concretize_tile_among_workgroups.mlir",
            "concretize_tile_among_workgroups_dynamic.mlir",
            "convert_to_gpu.mlir",
            "convert_to_spirv.mlir",
            "dead_alloc.mlir",
            "fold_gpu_procid_uses.mlir",
            "materialize_launch_configuration.mlir",
            "materialize_launch_configuration2.mlir",
            "pipeline_matmul_cooperative_matrix.mlir",
            "pipeline_matmul_vectorization.mlir",
            "promote_workgroup_memory.mlir",
            "tile_and_vectorize_batch_matmul.mlir",
            "tile_and_vectorize_conv.mlir",
            "tile_and_vectorize_matmul.mlir",
            "vector_to_cooperative_matrix.mlir",
            "vector_to_gpu.mlir",
            "vectorize_elementwise_ops.mlir",
            "vectorize_matmul.mlir",
            "vectorize_memref_load_store.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)
