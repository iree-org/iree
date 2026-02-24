// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/platform_topology.h"

//===----------------------------------------------------------------------===//
// Platform topology query dispatch
//===----------------------------------------------------------------------===//
// This file dispatches to platform-specific implementations at compile time.
// The build system selects exactly one implementation file based on the target
// platform:
// - Linux (not Android cpuinfo, not Emscripten): platform_topology_sysfs.c
// - macOS/iOS: platform_topology_darwin.c
// - Windows: platform_topology_win32.c
// - Fallback: platform_topology_fallback.c
//
// This mirrors the structure of iree/task/topology.c.

// Platform-specific implementations must define:
// - iree_hal_platform_query_numa_node_count_impl()
// - iree_hal_platform_query_numa_distance_impl()
// - iree_hal_platform_query_pcie_same_root_impl()
// - iree_hal_platform_query_pcie_bdf_from_path_impl()

// Declare platform-specific implementation functions.
extern iree_host_size_t iree_hal_platform_query_numa_node_count_impl(void);

extern iree_status_t iree_hal_platform_query_numa_distance_impl(
    uint8_t node_a, uint8_t node_b, uint8_t* out_distance);

extern bool iree_hal_platform_query_pcie_same_root_impl(
    iree_hal_platform_pcie_bdf_t bdf_a, iree_hal_platform_pcie_bdf_t bdf_b);

extern iree_status_t iree_hal_platform_query_pcie_bdf_from_path_impl(
    const char* device_path, iree_hal_platform_pcie_bdf_t* out_bdf);

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_host_size_t iree_hal_platform_query_numa_node_count(void) {
  return iree_hal_platform_query_numa_node_count_impl();
}

iree_status_t iree_hal_platform_query_numa_distance(uint8_t node_a,
                                                    uint8_t node_b,
                                                    uint8_t* out_distance) {
  IREE_ASSERT_ARGUMENT(out_distance);
  return iree_hal_platform_query_numa_distance_impl(node_a, node_b,
                                                    out_distance);
}

bool iree_hal_platform_query_pcie_same_root(
    iree_hal_platform_pcie_bdf_t bdf_a, iree_hal_platform_pcie_bdf_t bdf_b) {
  return iree_hal_platform_query_pcie_same_root_impl(bdf_a, bdf_b);
}

iree_status_t iree_hal_platform_query_pcie_bdf_from_path(
    const char* device_path, iree_hal_platform_pcie_bdf_t* out_bdf) {
  IREE_ASSERT_ARGUMENT(device_path);
  IREE_ASSERT_ARGUMENT(out_bdf);
  return iree_hal_platform_query_pcie_bdf_from_path_impl(device_path, out_bdf);
}
