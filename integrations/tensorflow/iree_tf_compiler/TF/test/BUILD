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

load("@iree//iree:lit_test.bzl", "iree_lit_test_suite")
load("@iree//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "convert_to_mhlo.mlir",
            "lower_global_tensors.mlir",
            "lower_global_tensors_complex.mlir",
            "lower_global_tensors_invalid.mlir",
            "propagate_resource_casts.mlir",
            "saved_model_to_iree_abi.mlir",
            "strip_asserts.mlir",
            "strip_metadata.mlir",
            "verify_fully_converted.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree_tf_compiler:iree-tf-opt",
        "@iree//iree/tools:IreeFileCheck",
    ],
    driver = "@iree//iree/tools:run_lit.sh",
)
