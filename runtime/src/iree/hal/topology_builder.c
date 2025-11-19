// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/topology_builder.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/hal/device.h"

//===----------------------------------------------------------------------===//
// iree_hal_topology_builder_t
//===----------------------------------------------------------------------===//

void iree_hal_topology_builder_initialize(iree_hal_topology_builder_t* builder,
                                          uint32_t device_count) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(device_count > 0);
  IREE_ASSERT_ARGUMENT(device_count <= IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Zero the entire builder.
  memset(builder, 0, sizeof(*builder));

  // Initialize topology.
  builder->topology.device_count = device_count;

  // Initialize self-edges.
  for (uint32_t i = 0; i < device_count; ++i) {
    uint32_t idx = i * device_count + i;
    builder->topology.edges[idx] = iree_hal_topology_edge_make_self();
    builder->edges_set[idx] = true;
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_topology_builder_set_edge(
    iree_hal_topology_builder_t* builder, uint32_t src_ordinal,
    uint32_t dst_ordinal, iree_hal_topology_edge_t edge) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate ordinals.
  if (src_ordinal >= builder->topology.device_count ||
      dst_ordinal >= builder->topology.device_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device ordinals [%u,%u] out of range [0,%u)",
                            src_ordinal, dst_ordinal,
                            builder->topology.device_count);
  }

  // Validate self-edges are optimal.
  if (src_ordinal == dst_ordinal) {
    // Allow some flexibility in self-edges but ensure basic requirements.
    iree_hal_topology_interop_mode_t wait_mode =
        iree_hal_topology_edge_wait_mode(edge);
    if (wait_mode != IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "self-edge [%u,%u] must have NATIVE wait mode",
                              src_ordinal, dst_ordinal);
    }
  }

  // Store edge.
  uint32_t idx = src_ordinal * builder->topology.device_count + dst_ordinal;
  builder->topology.edges[idx] = edge;
  builder->edges_set[idx] = true;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_topology_builder_set_numa_node(
    iree_hal_topology_builder_t* builder, uint32_t device_ordinal,
    uint8_t numa_node) {
  if (device_ordinal >= builder->topology.device_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device ordinal %u out of range [0,%u)",
                            device_ordinal, builder->topology.device_count);
  }

  builder->topology.numa_nodes[device_ordinal] = numa_node;
  return iree_ok_status();
}

// Validates topology during build.
static iree_status_t iree_hal_topology_builder_validate(
    iree_hal_topology_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  uint32_t device_count = builder->topology.device_count;

  // Check all edges are set.
  for (uint32_t i = 0; i < device_count; ++i) {
    for (uint32_t j = 0; j < device_count; ++j) {
      uint32_t idx = i * device_count + j;
      if (!builder->edges_set[idx]) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "edge [%u,%u] not set", i, j);
      }
    }
  }

  // Check link class symmetry.
  for (uint32_t i = 0; i < device_count; ++i) {
    for (uint32_t j = i + 1; j < device_count; ++j) {
      uint32_t ij_idx = i * device_count + j;
      uint32_t ji_idx = j * device_count + i;

      iree_hal_topology_link_class_t ij_link =
          iree_hal_topology_edge_link_class(builder->topology.edges[ij_idx]);
      iree_hal_topology_link_class_t ji_link =
          iree_hal_topology_edge_link_class(builder->topology.edges[ji_idx]);

      if (ij_link != ji_link) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "link class mismatch: edge[%u,%u]=%d != edge[%u,%u]=%d", i, j,
            ij_link, j, i, ji_link);
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_topology_builder_finalize(
    iree_hal_topology_builder_t* builder, iree_hal_topology_t* out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate the topology.
  iree_status_t status = iree_hal_topology_builder_validate(builder);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Copy the embedded topology.
  memcpy(out_topology, &builder->topology, sizeof(iree_hal_topology_t));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Edge construction helpers
//===----------------------------------------------------------------------===//

iree_hal_topology_edge_t iree_hal_topology_edge_make_self(void) {
  iree_hal_topology_edge_t edge = 0;

  // Optimal self-edge settings.
  edge = iree_hal_topology_edge_set_wait_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  edge = iree_hal_topology_edge_set_signal_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  edge = iree_hal_topology_edge_set_buffer_read_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  edge = iree_hal_topology_edge_set_buffer_write_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Set link class to same die.
  edge = iree_hal_topology_edge_set_link_class(
      edge, IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);

  // Set all capability flags for self.
  iree_hal_topology_capability_t caps =
      IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN |
      IREE_HAL_TOPOLOGY_CAPABILITY_UNIFIED_MEMORY |
      IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_HOST_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
      IREE_HAL_TOPOLOGY_CAPABILITY_CONCURRENT_SAFE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM |
      IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE;
  edge = iree_hal_topology_edge_set_capability_flags(edge, caps);

  // Zero cost for all operations on self.
  edge = iree_hal_topology_edge_set_wait_cost(edge, 0);
  edge = iree_hal_topology_edge_set_signal_cost(edge, 0);
  edge = iree_hal_topology_edge_set_copy_cost(edge, 0);
  edge = iree_hal_topology_edge_set_latency_class(edge, 0);
  edge = iree_hal_topology_edge_set_numa_distance(edge, 0);

  // Set handle types to native.
  edge = iree_hal_topology_edge_set_semaphore_import_types(
      edge, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  edge = iree_hal_topology_edge_set_semaphore_export_types(
      edge, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  edge = iree_hal_topology_edge_set_buffer_import_types(
      edge, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  edge = iree_hal_topology_edge_set_buffer_export_types(
      edge, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);

  return edge;
}

iree_hal_topology_edge_t iree_hal_topology_edge_make_cross_driver(void) {
  iree_hal_topology_edge_t edge = 0;

  // Default cross-driver settings require import/export.
  edge = iree_hal_topology_edge_set_wait_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  edge = iree_hal_topology_edge_set_signal_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  edge = iree_hal_topology_edge_set_buffer_read_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  edge = iree_hal_topology_edge_set_buffer_write_mode(
      edge, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  // Assume PCIe link by default.
  edge = iree_hal_topology_edge_set_link_class(
      edge, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);

  // No special capabilities for cross-driver.
  edge = iree_hal_topology_edge_set_capability_flags(
      edge, IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  // Moderate costs for cross-driver operations.
  edge = iree_hal_topology_edge_set_wait_cost(edge, 5);
  edge = iree_hal_topology_edge_set_signal_cost(edge, 5);
  edge = iree_hal_topology_edge_set_copy_cost(edge, 10);
  edge =
      iree_hal_topology_edge_set_latency_class(edge, 8);  // PCIe range (<10us).
  edge = iree_hal_topology_edge_set_numa_distance(edge, 2);

  // NOTE: we could add default OPAQUE_FD/etc per platform here but only if we
  // know for certain every HAL supports it. We don't currently mandate a common
  // required primitive so we avoid setting it at all.

  return edge;
}

IREE_API_EXPORT iree_hal_topology_edge_t
iree_hal_topology_edge_from_capabilities(
    const iree_hal_device_capabilities_t* src_caps,
    const iree_hal_device_capabilities_t* dst_caps,
    iree_string_view_t src_driver_name, iree_string_view_t dst_driver_name) {
  iree_hal_topology_edge_t edge = 0;

  // Same driver detection (enables NATIVE mode).
  bool same_driver = iree_string_view_equal(src_driver_name, dst_driver_name);
  if (same_driver) {
    edge |= IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN;
  }

  // Physical device UUID matching (cross-driver same-GPU detection).
  bool same_physical_device = false;
  if (src_caps->has_physical_device_uuid &&
      dst_caps->has_physical_device_uuid) {
    same_physical_device = (memcmp(src_caps->physical_device_uuid,
                                   dst_caps->physical_device_uuid, 16) == 0);
    if (same_physical_device) {
      // Same physical GPU! Upgrade link class even if different drivers.
      edge = iree_hal_topology_edge_set_link_class(
          edge, IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);
    }
  }

  // External handle type intersections.
  uint32_t semaphore_import_types =
      dst_caps->semaphore_export_types & src_caps->semaphore_import_types;
  uint32_t semaphore_export_types =
      src_caps->semaphore_export_types & dst_caps->semaphore_import_types;
  uint32_t buffer_import_types =
      dst_caps->buffer_export_types & src_caps->buffer_import_types;
  uint32_t buffer_export_types =
      src_caps->buffer_export_types & dst_caps->buffer_import_types;

  edge = iree_hal_topology_edge_set_semaphore_import_types(
      edge, semaphore_import_types);
  edge = iree_hal_topology_edge_set_semaphore_export_types(
      edge, semaphore_export_types);
  edge =
      iree_hal_topology_edge_set_buffer_import_types(edge, buffer_import_types);
  edge =
      iree_hal_topology_edge_set_buffer_export_types(edge, buffer_export_types);

  // Derive interop modes from handle types and flags.
  iree_hal_topology_interop_mode_t wait_mode, signal_mode;
  if (same_driver &&
      (semaphore_import_types & IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE)) {
    wait_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
  } else if (semaphore_import_types != 0) {
    wait_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT;
  } else {
    wait_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  }

  if (same_driver &&
      (semaphore_export_types & IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE)) {
    signal_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
  } else if (semaphore_export_types != 0) {
    signal_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT;
  } else {
    signal_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  }

  edge = iree_hal_topology_edge_set_wait_mode(edge, wait_mode);
  edge = iree_hal_topology_edge_set_signal_mode(edge, signal_mode);

  // Buffer modes (similar logic).
  iree_hal_topology_interop_mode_t buffer_read_mode, buffer_write_mode;
  if ((src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_P2P_COPY) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_P2P_COPY) && same_driver) {
    buffer_read_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
    buffer_write_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
  } else if (buffer_import_types != 0) {
    buffer_read_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT;
    buffer_write_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT;
  } else {
    buffer_read_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
    buffer_write_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  }

  edge = iree_hal_topology_edge_set_buffer_read_mode(edge, buffer_read_mode);
  edge = iree_hal_topology_edge_set_buffer_write_mode(edge, buffer_write_mode);

  // Capability flags (bitwise AND of device flags).
  iree_hal_topology_capability_t caps = 0;

  if ((src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_UNIFIED_MEMORY) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_UNIFIED_MEMORY)) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_UNIFIED_MEMORY;
  }

  if ((src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_PEER_COHERENT) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_PEER_COHERENT) &&
      same_driver) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT;
  }

  if ((src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_HOST_COHERENT) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_HOST_COHERENT)) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_HOST_COHERENT;
  }

  if ((src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_P2P_COPY) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_P2P_COPY)) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY;
  }

  if ((src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_CONCURRENT_SAFE) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_CONCURRENT_SAFE)) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_CONCURRENT_SAFE;
  }

  if ((src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_ATOMIC_SCOPE_DEVICE) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_ATOMIC_SCOPE_DEVICE)) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE;
  }

  if ((src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_ATOMIC_SCOPE_SYSTEM) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_ATOMIC_SCOPE_SYSTEM)) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM;
  }

  if ((src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_TIMELINE_SEMAPHORES) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_TIMELINE_SEMAPHORES)) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE;
  }

  edge = iree_hal_topology_edge_set_capability_flags(edge, caps);

  // NUMA distance (if different NUMA nodes).
  if (src_caps->numa_node != dst_caps->numa_node) {
    uint32_t numa_distance = src_caps->numa_node > dst_caps->numa_node
                                 ? src_caps->numa_node - dst_caps->numa_node
                                 : dst_caps->numa_node - src_caps->numa_node;
    numa_distance =
        numa_distance > 15 ? 15 : numa_distance;  // Clamp to 4 bits.
    edge = iree_hal_topology_edge_set_numa_distance(edge, numa_distance);
  }

  // Default link class (refinement can upgrade).
  iree_hal_topology_link_class_t link_class =
      IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE;
  if (same_physical_device) {
    link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE;
  } else if (same_driver) {
    link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT;
  } else {
    link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED;
  }
  edge = iree_hal_topology_edge_set_link_class(edge, link_class);

  // Default costs (refinement can adjust).
  // Derive from interop modes and link class for accurate scheduling hints.
  uint32_t wait_cost = (wait_mode == IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE) ? 0
                       : (wait_mode == IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT)
                           ? 3
                           : 10;
  uint32_t signal_cost =
      (signal_mode == IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE)   ? 1
      : (signal_mode == IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT) ? 3
                                                               : 10;

  // Copy cost based on link class:
  // SAME_DIE: 0-3 (very low, >500GB/s direct access)
  // PCIE_SAME_ROOT: 8-11 (moderate, ~30GB/s PCIe)
  // HOST_STAGED: 12-14 (high, <10GB/s with host bounce)
  uint32_t copy_cost = 0;
  uint32_t latency_class = 0;
  if (same_physical_device) {
    copy_cost = 0;      // Zero-copy same device.
    latency_class = 0;  // <10ns same device.
  } else if (same_driver) {
    copy_cost = 9;      // PCIe Gen4x16 (~30GB/s).
    latency_class = 8;  // <10us PCIe round-trip.
  } else {
    copy_cost = 13;      // Host staging (<10GB/s).
    latency_class = 11;  // 10-100us host bounce.
  }

  edge = iree_hal_topology_edge_set_wait_cost(edge, wait_cost);
  edge = iree_hal_topology_edge_set_signal_cost(edge, signal_cost);
  edge = iree_hal_topology_edge_set_copy_cost(edge, copy_cost);
  edge = iree_hal_topology_edge_set_latency_class(edge, latency_class);

  return edge;
}
