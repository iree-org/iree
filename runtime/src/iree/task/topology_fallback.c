// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "iree/task/topology.h"

// Fallback topology implementation when no specialized implementation is
// available. This provides a conservative single-threaded topology.
//
// Only compiles when:
// - cpuinfo is not available
// - Not on a platform with a dedicated implementation
#if !defined(IREE_TASK_USE_CPUINFO) && !defined(IREE_PLATFORM_APPLE) &&      \
    !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_EMSCRIPTEN) && \
    !defined(IREE_PLATFORM_LINUX)

#include <string.h>

// Initializes |out_topology| with a standardized behavior when no topology
// query is available (unsupported platform, etc).
static void iree_task_topology_initialize_fallback(
    iree_host_size_t max_group_count, iree_task_topology_t* out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, max_group_count);

  // Default to a single group: if a user wants more then they can manually
  // construct the topology themselves.
  iree_host_size_t group_count = 1;
  iree_task_topology_initialize_from_group_count(group_count, out_topology);

  IREE_TRACE_ZONE_END(z0);
}

iree_host_size_t iree_task_topology_query_node_count(void) { return 1; }

iree_task_topology_node_id_t iree_task_topology_query_current_node(void) {
  return 0;
}

iree_status_t iree_task_topology_fixup_constructive_sharing_masks(
    iree_task_topology_t* topology) {
  // No-op.
  return iree_ok_status();
}

iree_status_t iree_task_topology_initialize_from_logical_cpu_set(
    iree_host_size_t cpu_count, const uint32_t* cpu_ids,
    iree_task_topology_t* out_topology) {
  // Today we have a fixed limit on the number of groups within a particular
  // topology.
  if (cpu_count >= IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many CPUs specified (%" PRIhsz
                            " provided for a max capacity of %zu)",
                            cpu_count, IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, cpu_count);

  iree_task_topology_initialize(out_topology);

  out_topology->group_count = cpu_count;
  for (iree_host_size_t i = 0; i < cpu_count; ++i) {
    iree_task_topology_group_t* group = &out_topology->groups[i];
    iree_task_topology_group_initialize(i, group);
    group->processor_index = cpu_ids[i];

    // Without a topology query we can't get cache sizes so we just guess some
    // conservative values.
    group->caches.l1_data = 32 * 1024;
    group->caches.l2_data = 128 * 1024;

    // Without a topology query we can't get SMT and node info.
    iree_thread_affinity_t* affinity = &group->ideal_thread_affinity;
    memset(affinity, 0, sizeof(*affinity));
    affinity->id_assigned = 1;
    affinity->id = cpu_ids[i];
  }

  iree_status_t status =
      iree_task_topology_fixup_constructive_sharing_masks(out_topology);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_task_topology_initialize_from_physical_cores(
    iree_task_topology_node_id_t node_id,
    iree_task_topology_performance_level_t performance_level,
    iree_task_topology_distribution_t distribution,
    iree_host_size_t max_core_count, iree_task_topology_t* out_topology) {
  iree_task_topology_initialize_fallback(max_core_count, out_topology);
  return iree_ok_status();
}

#endif  // fallback guard
