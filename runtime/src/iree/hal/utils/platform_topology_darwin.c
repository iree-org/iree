// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/platform_topology.h"

#if defined(IREE_PLATFORM_APPLE)

#include <stdint.h>
#include <sys/sysctl.h>
#include <sys/types.h>

//===----------------------------------------------------------------------===//
// NUMA topology (Darwin/macOS)
//===----------------------------------------------------------------------===//

iree_host_size_t iree_hal_platform_query_numa_node_count_impl(void) {
  // macOS does not expose NUMA topology via public APIs.
  // Modern Apple Silicon uses Unified Memory Architecture (UMA) which is
  // effectively single-NUMA-node from a software perspective.
  // Intel Macs may have multiple NUMA nodes on high-end workstations, but
  // there's no standard API to query this.
  return 1;
}

iree_status_t iree_hal_platform_query_numa_distance_impl(
    uint8_t node_a, uint8_t node_b, uint8_t* out_distance) {
  IREE_ASSERT_ARGUMENT(out_distance);

  // macOS: only node 0 exists.
  if (node_a != 0 || node_b != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NUMA node out of range (only node 0 exists on "
                            "macOS/Darwin)");
  }

  // Same node distance.
  *out_distance = 10;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// PCIe topology (Darwin/macOS)
//===----------------------------------------------------------------------===//

bool iree_hal_platform_query_pcie_same_root_impl(
    iree_hal_platform_pcie_bdf_t bdf_a, iree_hal_platform_pcie_bdf_t bdf_b) {
  // macOS: PCIe topology queries would require IOKit framework.
  // For simplicity, assume all devices are under the same root complex.
  // This is reasonable for most Mac systems which have a single PCIe root.
  (void)bdf_a;
  (void)bdf_b;
  return true;
}

iree_status_t iree_hal_platform_query_pcie_bdf_from_path_impl(
    const char* device_path, iree_hal_platform_pcie_bdf_t* out_bdf) {
  IREE_ASSERT_ARGUMENT(device_path);
  IREE_ASSERT_ARGUMENT(out_bdf);

  // macOS: PCIe BDF query would require parsing IORegistry paths with IOKit.
  // This is complex and beyond the scope of the fallback implementation.
  // Most HAL drivers on macOS (Metal) don't need PCIe topology.
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "PCIe BDF query not implemented on macOS (would require IOKit)");
}

#endif  // IREE_PLATFORM_APPLE
