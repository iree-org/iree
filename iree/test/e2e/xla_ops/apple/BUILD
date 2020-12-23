# Copyright 2020 Google LLC
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

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_metal-spirv_metal",
    srcs = [
        "//iree/test/e2e/xla_ops:abs.mlir",
        "//iree/test/e2e/xla_ops:add.mlir",
        "//iree/test/e2e/xla_ops:broadcast.mlir",
        "//iree/test/e2e/xla_ops:broadcast_add.mlir",
        "//iree/test/e2e/xla_ops:broadcast_in_dim.mlir",
        "//iree/test/e2e/xla_ops:clamp.mlir",
        "//iree/test/e2e/xla_ops:compare.mlir",
        "//iree/test/e2e/xla_ops:constant.mlir",
        "//iree/test/e2e/xla_ops:convert.mlir",
        "//iree/test/e2e/xla_ops:cosine.mlir",
        "//iree/test/e2e/xla_ops:divide.mlir",
        "//iree/test/e2e/xla_ops:exponential.mlir",
        "//iree/test/e2e/xla_ops:gather.mlir",
        "//iree/test/e2e/xla_ops:log.mlir",
        "//iree/test/e2e/xla_ops:log_plus_one.mlir",
        "//iree/test/e2e/xla_ops:maximum.mlir",
        "//iree/test/e2e/xla_ops:minimum.mlir",
        "//iree/test/e2e/xla_ops:multiply.mlir",
        "//iree/test/e2e/xla_ops:negate.mlir",
        "//iree/test/e2e/xla_ops:remainder.mlir",
        "//iree/test/e2e/xla_ops:reshape.mlir",
        "//iree/test/e2e/xla_ops:rsqrt.mlir",
        "//iree/test/e2e/xla_ops:select.mlir",
        "//iree/test/e2e/xla_ops:sine.mlir",
        "//iree/test/e2e/xla_ops:slice.mlir",
        "//iree/test/e2e/xla_ops:sqrt.mlir",
        "//iree/test/e2e/xla_ops:subtract.mlir",
        "//iree/test/e2e/xla_ops:tanh.mlir",
        "//iree/test/e2e/xla_ops:torch_index_select.mlir",
        "//iree/test/e2e/xla_ops:transpose.mlir",
        "//iree/test/e2e/xla_ops:while.mlir",
    ],
    driver = "metal",
    target_backend = "metal-spirv",
)

test_suite(
    name = "check",
    tests = [
        ":check_metal-spirv_metal",
    ],
)
