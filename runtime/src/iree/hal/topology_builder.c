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
#include "iree/hal/utils/platform_topology.h"

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
        iree_hal_topology_edge_wait_mode(edge.lo);
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
          iree_hal_topology_edge_link_class(builder->topology.edges[ij_idx].lo);
      iree_hal_topology_link_class_t ji_link =
          iree_hal_topology_edge_link_class(builder->topology.edges[ji_idx].lo);

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
  iree_hal_topology_edge_scheduling_word_t lo = 0;
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Optimal self-edge settings.
  lo = iree_hal_topology_edge_set_wait_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  lo = iree_hal_topology_edge_set_signal_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  lo = iree_hal_topology_edge_set_buffer_read_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  lo = iree_hal_topology_edge_set_buffer_write_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Set link class to same die.
  lo = iree_hal_topology_edge_set_link_class(
      lo, IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);

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
  lo = iree_hal_topology_edge_set_capability_flags(lo, caps);

  // Zero cost for all operations on self.
  lo = iree_hal_topology_edge_set_wait_cost(lo, 0);
  lo = iree_hal_topology_edge_set_signal_cost(lo, 0);
  lo = iree_hal_topology_edge_set_copy_cost(lo, 0);
  lo = iree_hal_topology_edge_set_latency_class(lo, 0);
  lo = iree_hal_topology_edge_set_numa_distance(lo, 0);

  // Set handle types to native.
  hi = iree_hal_topology_edge_set_semaphore_import_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  hi = iree_hal_topology_edge_set_semaphore_export_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  hi = iree_hal_topology_edge_set_buffer_import_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  hi = iree_hal_topology_edge_set_buffer_export_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);

  iree_hal_topology_edge_t edge = {lo, hi};
  return edge;
}

iree_hal_topology_edge_t iree_hal_topology_edge_make_cross_driver(void) {
  iree_hal_topology_edge_scheduling_word_t lo = 0;
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Default cross-driver settings require import/export.
  lo = iree_hal_topology_edge_set_wait_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  lo = iree_hal_topology_edge_set_signal_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  lo = iree_hal_topology_edge_set_buffer_read_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  lo = iree_hal_topology_edge_set_buffer_write_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  // Assume PCIe link by default.
  lo = iree_hal_topology_edge_set_link_class(
      lo, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);

  // No special capabilities for cross-driver.
  lo = iree_hal_topology_edge_set_capability_flags(
      lo, IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  // Moderate costs for cross-driver operations.
  lo = iree_hal_topology_edge_set_wait_cost(lo, 5);
  lo = iree_hal_topology_edge_set_signal_cost(lo, 5);
  lo = iree_hal_topology_edge_set_copy_cost(lo, 10);
  lo = iree_hal_topology_edge_set_latency_class(lo, 8);  // PCIe range (<10us).
  lo = iree_hal_topology_edge_set_numa_distance(lo, 2);

  // NOTE: we could add default OPAQUE_FD/etc per platform here but only if we
  // know for certain every HAL supports it. We don't currently mandate a common
  // required primitive so we avoid setting it at all.

  iree_hal_topology_edge_t edge = {lo, hi};
  return edge;
}

IREE_API_EXPORT iree_hal_topology_edge_t
iree_hal_topology_edge_from_capabilities(
    const iree_hal_device_capabilities_t* src_caps,
    const iree_hal_device_capabilities_t* dst_caps,
    iree_string_view_t src_driver_name, iree_string_view_t dst_driver_name) {
  iree_hal_topology_edge_scheduling_word_t lo = 0;
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Same driver detection (enables NATIVE mode).
  bool same_driver = iree_string_view_equal(src_driver_name, dst_driver_name);

  // Same-driver aliasing detection: two iree_hal_device_t instances wrapping
  // the same underlying driver object. They share all resources and should
  // behave as self-edges (zero-cost NATIVE everything).
  if (same_driver && src_caps->driver_device_handle != 0 &&
      src_caps->driver_device_handle == dst_caps->driver_device_handle) {
    return iree_hal_topology_edge_make_self();
  }

  // Physical device UUID matching (cross-driver same-GPU detection).
  bool same_physical_device = false;
  if (src_caps->has_physical_device_uuid &&
      dst_caps->has_physical_device_uuid) {
    same_physical_device = (memcmp(src_caps->physical_device_uuid,
                                   dst_caps->physical_device_uuid, 16) == 0);
    if (same_physical_device) {
      // Same physical GPU! Upgrade link class even if different drivers.
      lo = iree_hal_topology_edge_set_link_class(
          lo, IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);
    }
  }

  // External handle type intersections.
  // For a directed edge src→dst:
  //   import_types = what dst can import from src (src exports ∩ dst imports)
  //   export_types = what src can export to dst (dst exports ∩ src imports)
  uint32_t semaphore_import_types =
      src_caps->semaphore_export_types & dst_caps->semaphore_import_types;
  uint32_t semaphore_export_types =
      dst_caps->semaphore_export_types & src_caps->semaphore_import_types;
  uint32_t buffer_import_types =
      src_caps->buffer_export_types & dst_caps->buffer_import_types;
  uint32_t buffer_export_types =
      dst_caps->buffer_export_types & src_caps->buffer_import_types;

  hi = iree_hal_topology_edge_set_semaphore_import_types(
      hi, semaphore_import_types);
  hi = iree_hal_topology_edge_set_semaphore_export_types(
      hi, semaphore_export_types);
  hi = iree_hal_topology_edge_set_buffer_import_types(hi, buffer_import_types);
  hi = iree_hal_topology_edge_set_buffer_export_types(hi, buffer_export_types);

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

  lo = iree_hal_topology_edge_set_wait_mode(lo, wait_mode);
  lo = iree_hal_topology_edge_set_signal_mode(lo, signal_mode);

  // Buffer modes.
  //
  // NATIVE: memory is load/store addressable across the link (unified memory,
  //   large BAR P2P). Scheduler can reference the buffer directly in
  //   dispatches.
  // IMPORT: buffer handle can be imported. Scheduler imports then uses
  // directly. COPY: a transfer command is required (P2P DMA or host-staged).
  // Scheduler
  //   must allocate on dst and issue a copy. Cost distinguishes P2P from host.
  //
  // P2P_COPY alone means the DMA engine can move data directly between devices,
  // but shader/host load/store may fault on the remote memory. Only when
  // PEER_ADDRESSABLE is also set can we safely use NATIVE mode.
  bool peer_addressable =
      (src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_PEER_ADDRESSABLE) &&
      (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_PEER_ADDRESSABLE);
  bool p2p_copy = (src_caps->flags & IREE_HAL_DEVICE_CAPABILITY_P2P_COPY) &&
                  (dst_caps->flags & IREE_HAL_DEVICE_CAPABILITY_P2P_COPY);

  iree_hal_topology_interop_mode_t buffer_read_mode, buffer_write_mode;
  if (peer_addressable && same_driver) {
    // Load/store addressable: scheduler can reference the buffer directly.
    buffer_read_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
    buffer_write_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
  } else if (buffer_import_types != 0) {
    // Can import buffer handles for sharing.
    buffer_read_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT;
    buffer_write_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT;
  } else {
    // Must issue a transfer command (P2P DMA if p2p_copy, otherwise
    // host-staged). The copy_cost and link_class encode the actual cost.
    buffer_read_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
    buffer_write_mode = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  }

  lo = iree_hal_topology_edge_set_buffer_read_mode(lo, buffer_read_mode);
  lo = iree_hal_topology_edge_set_buffer_write_mode(lo, buffer_write_mode);

  // Capability flags (bitwise AND of device flags).
  iree_hal_topology_capability_t caps = 0;

  if (same_driver) {
    caps |= IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN;
  }

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

  lo = iree_hal_topology_edge_set_capability_flags(lo, caps);

  // NUMA distance (queried from ACPI SLIT table via platform APIs).
  if (src_caps->numa_node != dst_caps->numa_node) {
    uint8_t slit_distance = 0;
    iree_status_t numa_status = iree_hal_platform_query_numa_distance(
        src_caps->numa_node, dst_caps->numa_node, &slit_distance);
    uint32_t scaled_distance;
    if (iree_status_is_ok(numa_status)) {
      // Normalize SLIT distance (10=same, 20=1hop, 30=2hop, ...) to 0-15 scale.
      // Subtract the "same node" base of 10, divide by 2 to compress range.
      scaled_distance = slit_distance > 10 ? (slit_distance - 10) / 2 : 0;
    } else {
      iree_status_ignore(numa_status);
      // Platform doesn't support SLIT queries; use a conservative default
      // for cross-node distance. This will be refined by driver-specific logic
      // via refine_topology_edge.
      scaled_distance = 3;
    }
    scaled_distance = scaled_distance > 15 ? 15 : scaled_distance;
    lo = iree_hal_topology_edge_set_numa_distance(lo, scaled_distance);
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
  lo = iree_hal_topology_edge_set_link_class(lo, link_class);

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

  // Copy cost based on link class and P2P capability:
  // SAME_DIE: 0-3 (very low, >500GB/s direct access)
  // P2P DMA: 5-7 (direct device-to-device DMA, no host bounce)
  // PCIE_SAME_ROOT: 8-11 (moderate, ~30GB/s PCIe, may need host staging)
  // HOST_STAGED: 12-14 (high, <10GB/s with host bounce)
  uint32_t copy_cost = 0;
  uint32_t latency_class = 0;
  if (same_physical_device) {
    copy_cost = 0;      // Zero-copy same device.
    latency_class = 0;  // <10ns same device.
  } else if (p2p_copy && same_driver) {
    copy_cost = 5;      // P2P DMA (~100GB/s NVLink, ~30GB/s PCIe P2P).
    latency_class = 5;  // ~1us P2P DMA round-trip.
  } else if (same_driver) {
    copy_cost = 9;      // PCIe without P2P (~30GB/s with driver staging).
    latency_class = 8;  // <10us driver-managed transfer.
  } else {
    copy_cost = 13;      // Host staging (<10GB/s).
    latency_class = 11;  // 10-100us host bounce.
  }

  lo = iree_hal_topology_edge_set_wait_cost(lo, wait_cost);
  lo = iree_hal_topology_edge_set_signal_cost(lo, signal_cost);
  lo = iree_hal_topology_edge_set_copy_cost(lo, copy_cost);
  lo = iree_hal_topology_edge_set_latency_class(lo, latency_class);

  iree_hal_topology_edge_t edge = {lo, hi};
  return edge;
}
