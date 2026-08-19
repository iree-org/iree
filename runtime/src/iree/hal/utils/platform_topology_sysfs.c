// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/platform_topology.h"

#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_WASM)

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/sysfs.h"

//===----------------------------------------------------------------------===//
// NUMA topology (Linux sysfs)
//===----------------------------------------------------------------------===//

// Node ids are uint8_t throughout this API (see
// iree_hal_platform_query_numa_distance), so at most 256 nodes are
// addressable. That bounds both files this reads:
//   node/nodeN/distance - one value per node, at most three digits plus a
//                         separator, so 256 * 4 bytes.
//   node/online         - a list of node ids, whose worst case is smaller.
#define IREE_HAL_PLATFORM_MAX_NUMA_NODES 256
#define IREE_HAL_PLATFORM_NODE_FILE_MAX_BYTES \
  (IREE_HAL_PLATFORM_MAX_NUMA_NODES * 4 + 1)

// Raises |user_data|, a uint32_t holding an exclusive upper bound on the online
// node ids, to cover this range. 0 means the file named no nodes.
static bool iree_hal_platform_numa_online_callback(uint32_t start_id,
                                                   uint32_t end_id,
                                                   void* user_data) {
  (void)start_id;
  uint32_t* node_id_limit = (uint32_t*)user_data;
  if (end_id > *node_id_limit) *node_id_limit = end_id;
  return true;  // continue enumeration
}

iree_host_size_t iree_hal_platform_query_numa_node_count_impl(void) {
  // Read /sys/devices/system/node/online, which uses the kernel list format
  // ("0", "0-3", "0,2-4") that iree_sysfs_try_parse_cpu_list handles. Unlike
  // the task system's node enumeration this counts every online node,
  // including memory-only ones (CXL, Optane): those have NUMA distances and
  // can host device memory even though no CPU belongs to them.
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(&builder, "%s/node/online",
                                                  iree_sysfs_get_root_path()));
  char buffer[IREE_HAL_PLATFORM_NODE_FILE_MAX_BYTES];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length);
  // Truncating a node list would silently report the wrong nodes, so a file
  // that exists but does not fit is fatal. Absent or unreadable falls back.
  if (iree_status_is_out_of_range(status)) {
    IREE_CHECK_OK(status);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 1;  // no NUMA information; assume a single node
  }

  // The accumulated value is already what callers want: a bound on node *ids*,
  // i.e. the highest online id plus one. An unparseable or empty list reports a
  // single node, matching the unreadable-file fallback above.
  uint32_t node_id_limit = 0;
  if (!iree_sysfs_try_parse_cpu_list(iree_make_string_view(buffer, length),
                                     iree_hal_platform_numa_online_callback,
                                     &node_id_limit) ||
      node_id_limit == 0) {
    return 1;
  }
  return (iree_host_size_t)node_id_limit;
}

bool iree_hal_platform_try_query_numa_distance_impl(uint8_t node_a,
                                                    uint8_t node_b,
                                                    uint8_t* out_distance) {
  IREE_ASSERT_ARGUMENT(out_distance);
  *out_distance = 10;  // Default: same node.

  // Validate node IDs.
  iree_host_size_t node_count = iree_hal_platform_query_numa_node_count_impl();
  if (node_a >= node_count || node_b >= node_count) {
    return false;
  }

  // Same node: distance is 10 (standard NUMA distance for local node).
  if (node_a == node_b) {
    *out_distance = 10;
    return true;
  }

  // Read distance from /sys/devices/system/node/node<A>/distance.
  // Format: space-separated list of distances from node A to all other nodes.
  // Example (4-node system): "10 20 20 20" (node 0 distances to 0,1,2,3).
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/node/node%u/distance", iree_sysfs_get_root_path(), node_a));

  char buffer[IREE_HAL_PLATFORM_NODE_FILE_MAX_BYTES];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length);
  // Truncating a node list would silently report the wrong nodes, so a file
  // that exists but does not fit is fatal. Absent or unreadable falls back.
  if (iree_status_is_out_of_range(status)) {
    IREE_CHECK_OK(status);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    // No SLIT data for this node: the caller has its own documented fallback
    // and can refine the edge with driver-specific information, which a
    // fabricated distance would silently pre-empt.
    return false;
  }

  // Parse space-separated list of distances.
  iree_string_view_t text = iree_make_string_view(buffer, length);
  text = iree_string_view_trim(text);

  uint32_t current_node = 0;
  iree_host_size_t offset = 0;
  while (offset < text.size && current_node <= node_b) {
    // Skip leading whitespace.
    while (offset < text.size &&
           (text.data[offset] == ' ' || text.data[offset] == '\t')) {
      offset++;
    }

    if (offset >= text.size) break;

    // Find end of current number.
    iree_host_size_t number_start = offset;
    while (offset < text.size && text.data[offset] >= '0' &&
           text.data[offset] <= '9') {
      offset++;
    }

    iree_string_view_t number_str =
        iree_string_view_substr(text, number_start, offset - number_start);

    if (current_node == node_b) {
      // This is the distance we're looking for.
      uint32_t distance_value;
      if (iree_string_view_atoi_uint32(number_str, &distance_value)) {
        // Clamp to uint8_t range.
        *out_distance = (uint8_t)iree_min(distance_value, 255u);
        return true;
      } else {
        return false;  // malformed SLIT entry; see above
      }
    }

    current_node++;
  }

  return false;  // node_b absent from the SLIT row; see above
}

iree_status_t iree_hal_platform_query_numa_distance_impl(
    uint8_t node_a, uint8_t node_b, uint8_t* out_distance) {
  IREE_ASSERT_ARGUMENT(out_distance);
  if (!iree_hal_platform_try_query_numa_distance_impl(node_a, node_b,
                                                      out_distance)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NUMA node out of range (node_a=%u, node_b=%u, "
                            "node_count=%zu)",
                            node_a, node_b,
                            iree_hal_platform_query_numa_node_count_impl());
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// PCIe topology (Linux sysfs)
//===----------------------------------------------------------------------===//

// Queries the PCIe root port for a given BDF by following symbolic links.
// Returns a hash of the root port path for same-root comparison.
static void iree_hal_platform_query_pcie_root_hash(
    iree_hal_platform_pcie_bdf_t bdf, uint64_t* out_hash) {
  IREE_ASSERT_ARGUMENT(out_hash);
  *out_hash = 0;

  // Construct path: /sys/bus/pci/devices/<domain>:<bus>:<dev>.<func>/
  char device_path[256];
  iree_snprintf(device_path, sizeof(device_path),
                "/sys/bus/pci/devices/%04x:%02x:%02x.%x",
                iree_hal_platform_pcie_bdf_domain(bdf),
                iree_hal_platform_pcie_bdf_bus(bdf),
                iree_hal_platform_pcie_bdf_device(bdf),
                iree_hal_platform_pcie_bdf_function(bdf));

  // Read the device link to find the root complex.
  // We'll use domain+bus as a simple hash.
  // A more robust implementation would follow ../ links to find the actual
  // root complex, but domain+bus is sufficient for most systems.

  // For simplicity, hash domain and bus number. Devices under the same root
  // complex will have bus numbers allocated from the same root within a domain.
  // This is a heuristic: true root detection requires traversing PCI hierarchy.
  *out_hash = ((uint64_t)iree_hal_platform_pcie_bdf_domain(bdf) << 32) |
              (uint64_t)iree_hal_platform_pcie_bdf_bus(bdf);
}

bool iree_hal_platform_query_pcie_same_root_impl(
    iree_hal_platform_pcie_bdf_t bdf_a, iree_hal_platform_pcie_bdf_t bdf_b) {
  // Simple heuristic: devices in the same domain with similar bus numbers are
  // likely under the same root complex.
  // This is a conservative approximation. A full implementation would traverse
  // the PCI hierarchy via sysfs symbolic links.

  uint64_t hash_a = 0;
  uint64_t hash_b = 0;
  iree_hal_platform_query_pcie_root_hash(bdf_a, &hash_a);
  iree_hal_platform_query_pcie_root_hash(bdf_b, &hash_b);

  // Same bus implies same root (heuristic).
  return hash_a == hash_b;
}

iree_status_t iree_hal_platform_query_pcie_bdf_from_path_impl(
    const char* device_path, iree_hal_platform_pcie_bdf_t* out_bdf) {
  IREE_ASSERT_ARGUMENT(device_path);
  IREE_ASSERT_ARGUMENT(out_bdf);
  *out_bdf = 0;

  // Expected path format for KFD devices:
  // /sys/class/kfd/kfd/topology/nodes/<N>/properties
  // We need to read the properties file and extract pci_bus, pci_device,
  // pci_function.

  // Check if this is a KFD node path.
  if (strstr(device_path, "/sys/class/kfd/kfd/topology/nodes/") ==
      device_path) {
    // Read properties file.
    char properties_path[512];
    iree_snprintf(properties_path, sizeof(properties_path), "%s/properties",
                  device_path);

    char buffer[4096];
    iree_host_size_t length = 0;
    IREE_RETURN_IF_ERROR(iree_sysfs_read_small_file(properties_path, buffer,
                                                    sizeof(buffer), &length));

    // Parse properties: look for pci_bus, pci_device, pci_function.
    // Format: "key value\n" per line.
    uint8_t bus = 0, device = 0, function = 0;
    bool found_bus = false, found_device = false, found_function = false;

    iree_string_view_t text = iree_make_string_view(buffer, length);
    iree_host_size_t offset = 0;

    while (offset < text.size) {
      iree_host_size_t line_end =
          iree_string_view_find_char(text, '\n', offset);
      if (line_end == IREE_STRING_VIEW_NPOS) line_end = text.size;

      iree_string_view_t line =
          iree_string_view_substr(text, offset, line_end - offset);
      line = iree_string_view_trim(line);

      // Split on whitespace.
      iree_host_size_t space_pos = iree_string_view_find_char(line, ' ', 0);
      if (space_pos != IREE_STRING_VIEW_NPOS) {
        iree_string_view_t key = iree_string_view_substr(line, 0, space_pos);
        iree_string_view_t value =
            iree_string_view_substr(line, space_pos + 1, IREE_HOST_SIZE_MAX);
        value = iree_string_view_trim(value);

        if (iree_string_view_equal(key, IREE_SV("pci_bus"))) {
          uint32_t bus_u32;
          if (iree_string_view_atoi_uint32(value, &bus_u32)) {
            bus = (uint8_t)bus_u32;
            found_bus = true;
          }
        } else if (iree_string_view_equal(key, IREE_SV("pci_device"))) {
          uint32_t device_u32;
          if (iree_string_view_atoi_uint32(value, &device_u32)) {
            device = (uint8_t)device_u32;
            found_device = true;
          }
        } else if (iree_string_view_equal(key, IREE_SV("pci_function"))) {
          uint32_t function_u32;
          if (iree_string_view_atoi_uint32(value, &function_u32)) {
            function = (uint8_t)function_u32;
            found_function = true;
          }
        }
      }

      offset = line_end + 1;
    }

    if (found_bus && found_device && found_function) {
      // KFD properties don't include domain; assume domain 0 for KFD devices.
      *out_bdf = iree_hal_platform_make_pcie_bdf(0, bus, device, function);
      return iree_ok_status();
    }

    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "PCIe BDF not found in KFD properties file");
  }

  // Try parsing as direct sysfs PCIe device path: /sys/bus/pci/devices/<bdf>/
  // Format: <domain>:<bus>:<device>.<function> or <bus>:<device>.<function>
  const char* bdf_start = strrchr(device_path, '/');
  if (bdf_start) {
    bdf_start++;  // Skip '/'.

    uint32_t domain, bus, device, function;
    // Try full format with domain first: DDDD:BB:DD.F
    if (sscanf(bdf_start, "%x:%x:%x.%x", &domain, &bus, &device, &function) ==
        4) {
      *out_bdf = iree_hal_platform_make_pcie_bdf(
          (uint16_t)domain, (uint8_t)bus, (uint8_t)device, (uint8_t)function);
      return iree_ok_status();
    }
    // Try short format without domain: BB:DD.F (implies domain 0)
    if (sscanf(bdf_start, "%x:%x.%x", &bus, &device, &function) == 3) {
      *out_bdf = iree_hal_platform_make_pcie_bdf(
          0, (uint8_t)bus, (uint8_t)device, (uint8_t)function);
      return iree_ok_status();
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "unrecognized device path format: %s", device_path);
}

#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_WASM
