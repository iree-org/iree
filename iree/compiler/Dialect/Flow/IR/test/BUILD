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
            "dispatch_ops.mlir",
            "dispatch_region_folding.mlir",
            "dispatch_regions.mlir",
            "dispatch_tensor_folding.mlir",
            "dispatch_workgroups.mlir",
            "dispatch_workgroups_folding.mlir",
            "executable_ops.mlir",
            "stream_folding.mlir",
            "stream_ops.mlir",
            "tensor_folding.mlir",
            "tensor_ops.mlir",
            "types.mlir",
            "variable_folding.mlir",
            "variable_ops.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)
