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
            "benchmark_batch_dispatches.mlir",
            "cse_variable_loads.mlir",
            "identify_constant_pools.mlir",
            "inline_device_switches.mlir",
            "materialize_constant_pool_buffers.mlir",
            "materialize_interfaces.mlir",
            "materialize_interfaces2.mlir",
            "materialize_resource_caches.mlir",
            "memoize_device_queries.mlir",
            "pack_allocations.mlir",
            "pack_constant_pool_storage.mlir",
            "propagate_constant_workgroup_info.mlir",
            "public_abi_generation.mlir",
            "resolve_entry_point_ordinals.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)
