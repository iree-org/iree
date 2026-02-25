// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device_group.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/hal/topology_builder.h"

//===----------------------------------------------------------------------===//
// iree_hal_device_group_t
//===----------------------------------------------------------------------===//

struct iree_hal_device_group_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Number of devices in the group.
  iree_host_size_t device_count;

  // Retained devices. Order defines topology indices (device i = devices[i]).
  iree_hal_device_t* devices[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];

  // Cached driver name for each device (backed by the device's own storage,
  // valid for the device's lifetime).
  iree_string_view_t driver_names[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];

  // Immutable topology matrix built during creation.
  // Embedded (not heap-allocated) so devices can hold a stable pointer to it
  // for the group's lifetime.
  iree_hal_topology_t topology;
};

IREE_API_EXPORT iree_status_t iree_hal_device_group_create_from_device(
    iree_hal_device_t* device, iree_allocator_t host_allocator,
    iree_hal_device_group_t** out_group) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_group);
  *out_group = NULL;
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder);
  IREE_RETURN_IF_ERROR(
      iree_hal_device_group_builder_add_device(&builder, device));
  return iree_hal_device_group_builder_finalize(&builder, host_allocator,
                                                out_group);
}

static void iree_hal_device_group_destroy(iree_hal_device_group_t* group) {
  IREE_ASSERT_ARGUMENT(group);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < group->device_count; ++i) {
    iree_hal_device_release(group->devices[i]);
  }

  iree_allocator_t host_allocator = group->host_allocator;
  iree_allocator_free(host_allocator, group);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_device_group_retain(
    iree_hal_device_group_t* group) {
  if (group) {
    iree_atomic_ref_count_inc(&group->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_device_group_release(
    iree_hal_device_group_t* group) {
  if (group && iree_atomic_ref_count_dec(&group->ref_count) == 1) {
    iree_hal_device_group_destroy(group);
  }
}

IREE_API_EXPORT iree_host_size_t
iree_hal_device_group_device_count(const iree_hal_device_group_t* group) {
  IREE_ASSERT_ARGUMENT(group);
  return group->device_count;
}

IREE_API_EXPORT iree_hal_device_t* iree_hal_device_group_device_at(
    const iree_hal_device_group_t* group, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(group);
  if (index >= group->device_count) return NULL;
  return group->devices[index];
}

IREE_API_EXPORT const iree_hal_topology_t* iree_hal_device_group_topology(
    const iree_hal_device_group_t* group) {
  IREE_ASSERT_ARGUMENT(group);
  return &group->topology;
}

//===----------------------------------------------------------------------===//
// iree_hal_device_group_builder_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_device_group_builder_initialize(
    iree_hal_device_group_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  memset(builder, 0, sizeof(*builder));
}

IREE_API_EXPORT void iree_hal_device_group_builder_deinitialize(
    iree_hal_device_group_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  memset(builder, 0, sizeof(*builder));
}

IREE_API_EXPORT iree_status_t iree_hal_device_group_builder_add_device(
    iree_hal_device_group_builder_t* builder, iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(device);
  if (builder->count >= IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device group builder at max capacity (%d)",
                            IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  }
  builder->devices[builder->count++] = device;
  return iree_ok_status();
}

// Computes pre-populated bitmaps for a device's topology info by scanning
// its edges in the topology matrix.
static void iree_hal_device_group_compute_bitmaps(
    const iree_hal_topology_t* topology, uint32_t device_index,
    iree_hal_device_topology_info_t* out_info) {
  uint32_t device_count = topology->device_count;
  out_info->can_wait_from = 0;
  out_info->can_signal_to = 0;
  out_info->can_import_from = 0;
  out_info->can_p2p_with = 0;

  for (uint32_t j = 0; j < device_count; ++j) {
    if (j == device_index) continue;

    // can_wait_from: can device_index wait on device j's semaphores?
    // Edge[j][device_index] describes how device_index interacts with j's
    // resources.
    iree_hal_topology_edge_t edge_from_j =
        iree_hal_topology_query_edge(topology, j, device_index);
    if (iree_hal_topology_edge_wait_mode(edge_from_j.lo) !=
        IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE) {
      out_info->can_wait_from |= (iree_hal_topology_device_bitmap_t)1 << j;
    }

    // can_import_from: can device_index import buffers from device j?
    if (iree_hal_topology_edge_buffer_read_mode(edge_from_j.lo) ==
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE ||
        iree_hal_topology_edge_buffer_read_mode(edge_from_j.lo) ==
            IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT) {
      out_info->can_import_from |= (iree_hal_topology_device_bitmap_t)1 << j;
    }

    // can_p2p_with: P2P access between device_index and device j?
    if (iree_hal_topology_edge_buffer_read_mode(edge_from_j.lo) ==
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE ||
        (iree_hal_topology_edge_capability_flags(edge_from_j.lo) &
         IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY)) {
      out_info->can_p2p_with |= (iree_hal_topology_device_bitmap_t)1 << j;
    }

    // can_signal_to: can device_index signal to device j?
    // Edge[device_index][j] describes how j interacts with device_index's
    // resources — the signal_mode tells us if j can observe our signals.
    iree_hal_topology_edge_t edge_to_j =
        iree_hal_topology_query_edge(topology, device_index, j);
    if (iree_hal_topology_edge_signal_mode(edge_to_j.lo) !=
        IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE) {
      out_info->can_signal_to |= (iree_hal_topology_device_bitmap_t)1 << j;
    }
  }
}

IREE_API_EXPORT iree_status_t iree_hal_device_group_builder_finalize(
    iree_hal_device_group_builder_t* builder, iree_allocator_t host_allocator,
    iree_hal_device_group_t** out_group) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_group);
  *out_group = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t device_count = builder->count;
  if (device_count == 0) {
    memset(builder, 0, sizeof(*builder));
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device group builder has no devices");
  }

  // Allocate the group.
  iree_hal_device_group_t* group = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*group), (void**)&group));
  memset(group, 0, sizeof(*group));
  iree_atomic_ref_count_init(&group->ref_count);
  group->host_allocator = host_allocator;
  group->device_count = device_count;

  // Retain all devices and cache driver names.
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    group->devices[i] = builder->devices[i];
    iree_hal_device_retain(group->devices[i]);
    group->driver_names[i] = iree_hal_device_id(group->devices[i]);
  }

  // Invalidate the builder now — we've taken everything we need.
  memset(builder, 0, sizeof(*builder));

  // Query capabilities from all devices.
  iree_hal_device_capabilities_t
      capabilities[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  memset(capabilities, 0, sizeof(capabilities));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    status =
        iree_hal_device_query_capabilities(group->devices[i], &capabilities[i]);
  }

  // Build topology.
  iree_hal_topology_builder_t topology_builder;
  iree_hal_topology_builder_initialize(&topology_builder,
                                       (uint32_t)device_count);

  // Compute and set edges for all device pairs.
  for (uint32_t i = 0; i < (uint32_t)device_count && iree_status_is_ok(status);
       ++i) {
    for (uint32_t j = 0;
         j < (uint32_t)device_count && iree_status_is_ok(status); ++j) {
      if (i == j) continue;  // Self-edges are pre-initialized.

      // Compute base edge from capabilities.
      iree_hal_topology_edge_t edge = iree_hal_topology_edge_from_capabilities(
          &capabilities[i], &capabilities[j], group->driver_names[i],
          group->driver_names[j]);

      // Allow same-driver devices to refine the edge with hardware-specific
      // knowledge (e.g., NVLink topology, Infinity Fabric link widths).
      if (iree_string_view_equal(group->driver_names[i],
                                 group->driver_names[j])) {
        status = iree_hal_device_refine_topology_edge(group->devices[i],
                                                      group->devices[j], &edge);
        if (!iree_status_is_ok(status)) break;
      }

      status =
          iree_hal_topology_builder_set_edge(&topology_builder, i, j, edge);
    }
  }

  // Set NUMA nodes from queried capabilities.
  for (uint32_t i = 0; i < (uint32_t)device_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_topology_builder_set_numa_node(&topology_builder, i,
                                                     capabilities[i].numa_node);
  }

  // Finalize topology.
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_topology_builder_finalize(&topology_builder, &group->topology);
  }

  // Assign topology info to each device.
  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_device_topology_info_t topology_info;
    memset(&topology_info, 0, sizeof(topology_info));
    topology_info.topology_index = (uint32_t)i;
    topology_info.topology = &group->topology;
    topology_info.self_edge =
        iree_hal_topology_query_edge(&group->topology, (uint32_t)i, (uint32_t)i)
            .lo;

    iree_hal_device_group_compute_bitmaps(&group->topology, (uint32_t)i,
                                          &topology_info);

    status =
        iree_hal_device_assign_topology_info(group->devices[i], &topology_info);
  }

  if (iree_status_is_ok(status)) {
    *out_group = group;
  } else {
    iree_hal_device_group_release(group);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
