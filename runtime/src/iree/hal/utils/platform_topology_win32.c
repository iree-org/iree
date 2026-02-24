// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/platform_topology.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <windows.h>

//===----------------------------------------------------------------------===//
// NUMA topology (Windows)
//===----------------------------------------------------------------------===//

iree_host_size_t iree_hal_platform_query_numa_node_count_impl(void) {
  // Query the highest NUMA node number.
  // GetNumaHighestNodeNumber returns the highest node number, not the count.
  ULONG highest_node_number = 0;
  if (GetNumaHighestNodeNumber(&highest_node_number)) {
    // Node count is highest_node + 1.
    return (iree_host_size_t)(highest_node_number + 1);
  }

  // Fallback: assume single NUMA node.
  return 1;
}

iree_status_t iree_hal_platform_query_numa_distance_impl(
    uint8_t node_a, uint8_t node_b, uint8_t* out_distance) {
  IREE_ASSERT_ARGUMENT(out_distance);
  *out_distance = 10;  // Default: same node.

  // Validate node IDs.
  iree_host_size_t node_count = iree_hal_platform_query_numa_node_count_impl();
  if (node_a >= node_count || node_b >= node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NUMA node out of range (node_a=%u, node_b=%u, "
                            "node_count=%zu)",
                            node_a, node_b, node_count);
  }

  // Same node: distance is 10.
  if (node_a == node_b) {
    *out_distance = 10;
    return iree_ok_status();
  }

  // Query NUMA distance using GetNumaProximityNodeEx.
  // This API returns proximity domain information, but Windows doesn't expose
  // SLIT-style distance tables directly. We'll use a heuristic based on
  // proximity domain equality.

  // Windows doesn't provide a direct NUMA distance query API like Linux SLIT.
  // We'll use a heuristic: nodes with the same proximity domain are closer.
  // For simplicity, return a fixed cross-node distance.
  *out_distance = 20;  // Default: one hop away.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// PCIe topology (Windows)
//===----------------------------------------------------------------------===//

bool iree_hal_platform_query_pcie_same_root_impl(
    iree_hal_platform_pcie_bdf_t bdf_a, iree_hal_platform_pcie_bdf_t bdf_b) {
  // Windows: PCIe topology queries would require Setup API or WMI.
  // For simplicity, assume all devices are under the same root complex.
  // This is conservative for single-socket systems but may be incorrect for
  // multi-socket systems.
  (void)bdf_a;
  (void)bdf_b;
  return true;
}

iree_status_t iree_hal_platform_query_pcie_bdf_from_path_impl(
    const char* device_path, iree_hal_platform_pcie_bdf_t* out_bdf) {
  IREE_ASSERT_ARGUMENT(device_path);
  IREE_ASSERT_ARGUMENT(out_bdf);

  // Windows: PCIe BDF query would require parsing device instance paths with
  // Setup API (SetupDiGetClassDevs, SetupDiEnumDeviceInfo, etc.).
  // This is complex and beyond the scope of the initial implementation.
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "PCIe BDF query not implemented on Windows (would require Setup API)");
}

#endif  // IREE_PLATFORM_WINDOWS
