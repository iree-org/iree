// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_PLATFORM_TOPOLOGY_H_
#define IREE_HAL_UTILS_PLATFORM_TOPOLOGY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Platform topology queries
//===----------------------------------------------------------------------===//
// Platform-agnostic APIs for querying system topology information that is
// not device-specific. This includes NUMA node information, CPU topology,
// and PCIe topology.
//
// These APIs abstract over platform-specific mechanisms:
// - Linux: sysfs (/sys/devices/system/, /sys/bus/pci/)
// - Windows: GetNumaHighestNodeNumber, GetNumaProximityNodeEx, Setup API
// - macOS: sysctlbyname, IOKit
// - Fallback: Returns conservative defaults when platform support unavailable
//
// HAL drivers use these APIs to refine topology edges with platform
// information (NUMA distances, PCIe topology) that complements device-specific
// queries (GPU memory accessibility, link types).

//===----------------------------------------------------------------------===//
// NUMA topology
//===----------------------------------------------------------------------===//

// Returns the total number of NUMA nodes in the system.
// Returns 1 if NUMA information is unavailable (single-node system).
iree_host_size_t iree_hal_platform_query_numa_node_count(void);

// Queries the NUMA distance between two nodes.
// NUMA distance is a relative cost metric where:
// - 10 = same node (local)
// - 20 = one hop away (typical cross-socket on 2-socket systems)
// - 30+ = multiple hops (NUMA domains, cross-chassis)
//
// Returns IREE_STATUS_OUT_OF_RANGE if node_a or node_b >= node count.
// Returns a default distance of 10 if NUMA information is unavailable.
iree_status_t iree_hal_platform_query_numa_distance(uint8_t node_a,
                                                    uint8_t node_b,
                                                    uint8_t* out_distance);

//===----------------------------------------------------------------------===//
// PCIe topology
//===----------------------------------------------------------------------===//

// PCIe Domain:Bus:Device.Function (DBDF) identifier.
//
// Packs the full 32-bit PCIe address into a single uint32_t following the
// standard PCIe specification addressing scheme. This matches the Linux kernel
// convention and sysfs device path format (/sys/bus/pci/devices/DDDD:BB:DD.F).
//
// Bit layout (32 bits total):
// - Bits [31:16]: Domain/Segment number (0-65535)
// - Bits [15:8]:  Bus number (0-255)
// - Bits [7:3]:   Device number (0-31)
// - Bits [2:0]:   Function number (0-7)
//
// The domain (also called segment) is critical for uniquely identifying devices
// on systems with multiple PCIe root complexes:
// - Single-socket consumer systems: typically domain 0 only
// - Multi-socket servers: each socket often has its own domain (0, 1, 2, ...)
// - Systems with PCIe expansion chassis: additional domains per chassis
// - Large-scale systems: multiple domains per socket for I/O scalability
//
// Example addresses:
// - 0000:01:00.0 = Domain 0, Bus 1, Device 0, Function 0
// - 0001:81:00.0 = Domain 1, Bus 129, Device 0, Function 0
typedef uint32_t iree_hal_platform_pcie_bdf_t;

// Constructs a BDF identifier from domain, bus, device, and function numbers.
static inline iree_hal_platform_pcie_bdf_t iree_hal_platform_make_pcie_bdf(
    uint16_t domain, uint8_t bus, uint8_t device, uint8_t function) {
  return (iree_hal_platform_pcie_bdf_t)(((uint32_t)domain << 16) |
                                        ((uint32_t)bus << 8) |
                                        ((uint32_t)device << 3) |
                                        (uint32_t)function);
}

// Extracts domain/segment number from a BDF identifier.
static inline uint16_t iree_hal_platform_pcie_bdf_domain(
    iree_hal_platform_pcie_bdf_t bdf) {
  return (uint16_t)((bdf >> 16) & 0xFFFF);
}

// Extracts bus number from a BDF identifier.
static inline uint8_t iree_hal_platform_pcie_bdf_bus(
    iree_hal_platform_pcie_bdf_t bdf) {
  return (uint8_t)((bdf >> 8) & 0xFF);
}

// Extracts device number from a BDF identifier.
static inline uint8_t iree_hal_platform_pcie_bdf_device(
    iree_hal_platform_pcie_bdf_t bdf) {
  return (uint8_t)((bdf >> 3) & 0x1F);
}

// Extracts function number from a BDF identifier.
static inline uint8_t iree_hal_platform_pcie_bdf_function(
    iree_hal_platform_pcie_bdf_t bdf) {
  return (uint8_t)(bdf & 0x07);
}

// Queries whether two PCIe devices share the same root complex.
// Devices under the same root complex can typically communicate with lower
// latency and higher bandwidth than devices across different root complexes.
//
// Returns true if both devices are under the same root complex.
// Returns false if they are under different root complexes or if PCIe topology
// information is unavailable.
bool iree_hal_platform_query_pcie_same_root(iree_hal_platform_pcie_bdf_t bdf_a,
                                            iree_hal_platform_pcie_bdf_t bdf_b);

// Queries the PCIe BDF for a device from a platform-specific path.
// Platform-specific path formats:
// - Linux: sysfs path like "/sys/class/kfd/kfd/topology/nodes/1"
// - Windows: device instance path
// - macOS: IORegistry path
//
// Returns IREE_STATUS_NOT_FOUND if the device path doesn't exist or doesn't
// correspond to a PCIe device.
// Returns IREE_STATUS_UNIMPLEMENTED on platforms without PCIe topology support.
iree_status_t iree_hal_platform_query_pcie_bdf_from_path(
    const char* device_path, iree_hal_platform_pcie_bdf_t* out_bdf);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_UTILS_PLATFORM_TOPOLOGY_H_
