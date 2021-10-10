# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
            "assign_target_devices.mlir",
            "benchmark_batch_dispatches.mlir",
            "elide_redundant_commands.mlir",
            "identify_constant_pools.mlir",
            "inline_device_switches.mlir",
            "materialize_constant_pool_buffers.mlir",
            "materialize_interfaces.mlir",
            "materialize_resource_caches.mlir",
            "memoize_device_queries.mlir",
            "pack_allocations.mlir",
            "pack_constant_pool_storage.mlir",
            "propagate_constant_workgroup_info.mlir",
            "resolve_entry_point_ordinals.mlir",
            "verify_target_environment.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)
