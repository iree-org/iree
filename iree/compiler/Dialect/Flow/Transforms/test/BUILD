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
            "create_benchmark_funcs.mlir",
            "deduplicate_executables.mlir",
            "dispatch_linalg_on_tensors.mlir",
            "dispatch_linalg_on_tensors_dynamic.mlir",
            "expand_variable_dynamic_dims.mlir",
            "fold_compatible_dispatch_regions.mlir",
            "form_streams.mlir",
            "hlo_to_hlo_preprocessing.mlir",
            "hlo_to_hlo_preprocessing_canoncalize_dot_general.mlir",
            "hlo_to_hlo_preprocessing_extract_pad_from_conv.mlir",
            "hoist_unstreamable_ops.mlir",
            "identify_dispatch_regions.mlir",
            "identify_dispatch_regions2_enable_matmul_fusion.mlir",
            "identify_dispatch_regions2_hlo.mlir",
            "identify_dispatch_regions2_linalg.mlir",
            "identify_dispatch_regions2_shapes.mlir",
            "identify_dispatch_regions2_std_fusion.mlir",
            "inject_dispatch_tracing.mlir",
            "legalize_input_types.mlir",
            "materialize_and_merge_exported_reflection.mlir",
            "materialize_exported_reflection.mlir",
            "merge_exported_reflection.mlir",
            "outline_dispatch_regions2.mlir",
            "outline_dispatch_regions_ranked_dynamic.mlir",
            "outline_large_constants.mlir",
            "rematerialize_dispatch_constants.mlir",
            "strip_and_splat_constant_variables.mlir",
            "transformation.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)
