// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/topology.h"

#if defined(IREE_PLATFORM_EMSCRIPTEN)

#include <emscripten/threading.h>

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

    // NOTE: without queries we can't get cache sizes so we just guess some
    // conservative values.
    group->caches.l1_data = 32 * 1024;
    group->caches.l2_data = 128 * 1024;

    // NOTE: without cpuinfo we can't get SMT and node info but this isn't
    // really used on Linux today anyway.
    iree_thread_affinity_t* affinity = &group->ideal_thread_affinity;
    memset(affinity, 0, sizeof(*affinity));
    affinity->specified = 1;
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
    iree_host_size_t max_core_count, iree_task_topology_t* out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, max_group_count);

  // Divide by 2 if there are an even number of cores assuming that most users
  // have SMT enabled. This physical cores initialization routine is intended to
  // filter out SMT but since we don't have a way to know in the browser we're
  // just guessing. This is a conservative decision as it means on a system with
  // SMT disabled we won't select all cores but it's better than oversubscribing
  // a machine with SMT enabled, especially in the browser. Hosting applications
  // can always assign a topology with their own logic/user settings/etc.
  iree_host_size_t group_count = emscripten_num_logical_cores();
  if (group_count > 1 && (group_count % 2) == 0) {
    group_count /= 2;
  }
  iree_task_topology_initialize_from_group_count(group_count, out_topology);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_EMSCRIPTEN
