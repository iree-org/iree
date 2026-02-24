// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/platform_topology.h"

// Fallback platform topology implementation for platforms without specific
// support. Returns conservative defaults that assume a simple single-NUMA-node,
// single-PCIe-root system.
//
// This implementation is used when:
// - The platform doesn't have a specific implementation
// - Platform-specific APIs are unavailable at runtime
// - The build explicitly disables platform topology support

// Only compiles when no platform-specific implementation is available.
// Each platform with a dedicated implementation guards by its own
// IREE_PLATFORM_* define, so the fallback must exclude all of them.
#if !defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_EMSCRIPTEN)
#if !defined(IREE_PLATFORM_APPLE) && !defined(IREE_PLATFORM_WINDOWS)

//===----------------------------------------------------------------------===//
// NUMA topology (fallback)
//===----------------------------------------------------------------------===//

iree_host_size_t iree_hal_platform_query_numa_node_count_impl(void) {
  // Fallback: assume single NUMA node.
  return 1;
}

iree_status_t iree_hal_platform_query_numa_distance_impl(
    uint8_t node_a, uint8_t node_b, uint8_t* out_distance) {
  IREE_ASSERT_ARGUMENT(out_distance);

  // Fallback: only node 0 exists.
  if (node_a != 0 || node_b != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NUMA node out of range (only node 0 exists in "
                            "fallback implementation)");
  }

  // Same node distance.
  *out_distance = 10;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// PCIe topology (fallback)
//===----------------------------------------------------------------------===//

bool iree_hal_platform_query_pcie_same_root_impl(
    iree_hal_platform_pcie_bdf_t bdf_a, iree_hal_platform_pcie_bdf_t bdf_b) {
  // Fallback: assume all devices are under the same root complex.
  // This is conservative for single-socket systems but may be incorrect for
  // multi-socket or multi-chassis systems.
  (void)bdf_a;
  (void)bdf_b;
  return true;
}

iree_status_t iree_hal_platform_query_pcie_bdf_from_path_impl(
    const char* device_path, iree_hal_platform_pcie_bdf_t* out_bdf) {
  IREE_ASSERT_ARGUMENT(device_path);
  IREE_ASSERT_ARGUMENT(out_bdf);

  // Fallback: PCIe topology queries are not supported.
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "PCIe BDF query not supported on this platform (no platform topology "
      "implementation available)");
}

#endif  // !IREE_PLATFORM_APPLE && !IREE_PLATFORM_WINDOWS
#endif  // !IREE_PLATFORM_LINUX || IREE_PLATFORM_EMSCRIPTEN
