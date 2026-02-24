// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/platform_topology.h"

#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_EMSCRIPTEN)

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/sysfs.h"

//===----------------------------------------------------------------------===//
// NUMA topology (Linux sysfs)
//===----------------------------------------------------------------------===//

iree_host_size_t iree_hal_platform_query_numa_node_count_impl(void) {
  // Read /sys/devices/system/node/online to get the list of online NUMA nodes.
  // Format: "0-3" (4 nodes) or "0" (single node) or "0,2-4" (nodes 0,2,3,4).
  char path[256];
  snprintf(path, sizeof(path), "%s/node/online", iree_sysfs_get_root_path());

  char buffer[256];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    // Fallback: assume single NUMA node if sysfs file doesn't exist.
    return 1;
  }

  // Count the maximum node ID in the online list.
  uint32_t max_node_id = 0;
  iree_string_view_t text = iree_make_string_view(buffer, length);
  text = iree_string_view_trim(text);

  // Parse CPU list format (same format as CPU online).
  // We'll count the max ID seen.
  iree_host_size_t offset = 0;
  while (offset < text.size) {
    iree_host_size_t comma_pos = iree_string_view_find_char(text, ',', offset);
    iree_host_size_t segment_end =
        (comma_pos == IREE_STRING_VIEW_NPOS) ? text.size : comma_pos;
    iree_string_view_t segment =
        iree_string_view_substr(text, offset, segment_end - offset);
    segment = iree_string_view_trim(segment);

    if (!iree_string_view_is_empty(segment)) {
      iree_host_size_t dash_pos = iree_string_view_find_char(segment, '-', 0);
      if (dash_pos == IREE_STRING_VIEW_NPOS) {
        // Single node: "N".
        uint32_t node_id;
        if (iree_string_view_atoi_uint32(segment, &node_id)) {
          if (node_id > max_node_id) max_node_id = node_id;
        }
      } else {
        // Range: "N-M".
        iree_string_view_t end_str =
            iree_string_view_substr(segment, dash_pos + 1, IREE_HOST_SIZE_MAX);
        uint32_t end_node_id;
        if (iree_string_view_atoi_uint32(end_str, &end_node_id)) {
          if (end_node_id > max_node_id) max_node_id = end_node_id;
        }
      }
    }

    offset = (comma_pos == IREE_STRING_VIEW_NPOS) ? text.size : comma_pos + 1;
  }

  // Node count is max_id + 1.
  return (iree_host_size_t)(max_node_id + 1);
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

  // Same node: distance is 10 (standard NUMA distance for local node).
  if (node_a == node_b) {
    *out_distance = 10;
    return iree_ok_status();
  }

  // Read distance from /sys/devices/system/node/node<A>/distance.
  // Format: space-separated list of distances from node A to all other nodes.
  // Example (4-node system): "10 20 20 20" (node 0 distances to 0,1,2,3).
  char path[256];
  snprintf(path, sizeof(path), "%s/node/node%u/distance",
           iree_sysfs_get_root_path(), node_a);

  char buffer[1024];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length);
  if (!iree_status_is_ok(status)) {
    // Distance file doesn't exist: assume default cross-node distance.
    iree_status_ignore(status);
    *out_distance = 20;  // Default: one hop away.
    return iree_ok_status();
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
        return iree_ok_status();
      } else {
        // Parse error: use default.
        *out_distance = 20;
        return iree_ok_status();
      }
    }

    current_node++;
  }

  // Didn't find node_b in the distance list: use default.
  *out_distance = 20;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// PCIe topology (Linux sysfs)
//===----------------------------------------------------------------------===//

// Queries the PCIe root port for a given BDF by following symbolic links.
// Returns a hash of the root port path for same-root comparison.
static iree_status_t iree_hal_platform_query_pcie_root_hash(
    iree_hal_platform_pcie_bdf_t bdf, uint64_t* out_hash) {
  IREE_ASSERT_ARGUMENT(out_hash);
  *out_hash = 0;

  // Construct path: /sys/bus/pci/devices/<domain>:<bus>:<dev>.<func>/
  char device_path[256];
  snprintf(device_path, sizeof(device_path),
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

  return iree_ok_status();
}

bool iree_hal_platform_query_pcie_same_root_impl(
    iree_hal_platform_pcie_bdf_t bdf_a, iree_hal_platform_pcie_bdf_t bdf_b) {
  // Simple heuristic: devices in the same domain with similar bus numbers are
  // likely under the same root complex.
  // This is a conservative approximation. A full implementation would traverse
  // the PCI hierarchy via sysfs symbolic links.

  uint64_t hash_a = 0;
  uint64_t hash_b = 0;

  iree_status_t status_a =
      iree_hal_platform_query_pcie_root_hash(bdf_a, &hash_a);
  iree_status_t status_b =
      iree_hal_platform_query_pcie_root_hash(bdf_b, &hash_b);

  if (!iree_status_is_ok(status_a) || !iree_status_is_ok(status_b)) {
    iree_status_ignore(status_a);
    iree_status_ignore(status_b);
    // Fallback: assume same root if we can't determine.
    return true;
  }

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
    snprintf(properties_path, sizeof(properties_path), "%s/properties",
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

#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_EMSCRIPTEN
