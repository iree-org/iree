// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/topology.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"

void iree_task_topology_group_initialize(
    uint8_t group_index, iree_task_topology_group_t* out_group) {
  memset(out_group, 0, sizeof(*out_group));
  out_group->group_index = group_index;
  snprintf(out_group->name, IREE_ARRAYSIZE(out_group->name), "iree-worker-%u",
           group_index);
  iree_thread_affinity_set_any(&out_group->ideal_thread_affinity);
  out_group->constructive_sharing_mask = IREE_TASK_TOPOLOGY_GROUP_MASK_ALL;
}

void iree_task_topology_initialize(iree_task_topology_t* out_topology) {
  IREE_ASSERT_ARGUMENT(out_topology);
  memset(out_topology, 0, sizeof(*out_topology));
}

void iree_task_topology_deinitialize(iree_task_topology_t* topology) {
  IREE_ASSERT_ARGUMENT(topology);
}

iree_status_t iree_task_topology_parse(iree_string_view_t value,
                                       iree_task_topology_t* out_topology) {
  // TODO(benvanik): define a format that is generally useful alongside cpuinfo.
  // Maybe colon-separated group-id values from thread affinities? Like:
  //   0.0:0.2:0.4:0.8 to indicate cores 0,2,4,8 on group 0
  //   0.0:0.1:1.0:1.1 to indicate cores 0,1 of both groups 0,1
  // etc
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

bool iree_task_topology_format(const iree_task_topology_t* topology,
                               iree_host_size_t buffer_capacity, char* buffer,
                               iree_host_size_t* out_buffer_length) {
  // TODO(benvanik): formatting to match parsing.
  return false;
}

iree_host_size_t iree_task_topology_group_capacity(
    const iree_task_topology_t* topology) {
  return IREE_ARRAYSIZE(topology->groups);
}

iree_host_size_t iree_task_topology_group_count(
    const iree_task_topology_t* topology) {
  return topology->group_count;
}

const iree_task_topology_group_t* iree_task_topology_get_group(
    const iree_task_topology_t* topology, iree_host_size_t group_index) {
  if (group_index >= topology->group_count) return NULL;
  return &topology->groups[group_index];
}

iree_status_t iree_task_topology_push_group(
    iree_task_topology_t* topology, const iree_task_topology_group_t* group) {
  if (topology->group_count + 1 > IREE_ARRAYSIZE(topology->groups)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "group capacity exceeded");
  }
  iree_task_topology_group_t* dst_group =
      &topology->groups[topology->group_count];
  memcpy(dst_group, group, sizeof(*group));
  dst_group->group_index = topology->group_count++;
  return iree_ok_status();
}

// Fixes constructive_sharing_mask values such that they represent other chosen
// topology groups instead of processor indices. We do this so that code using
// the topology groups doesn't need to know anything about which physical
// processor IDs a particular group is mapped to.
//
// This is implemented by platform-specific logic and may be a no-op if the
// platform doesn't support querying the required cache information.
iree_status_t iree_task_topology_fixup_constructive_sharing_masks(
    iree_task_topology_t* topology);

void iree_task_topology_initialize_from_group_count(
    iree_host_size_t group_count, iree_task_topology_t* out_topology) {
  // Clamp to the maximum we support.
  group_count = iree_min(group_count, IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT);

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, group_count);

  // Initialize default groups with no affinities specified.
  iree_task_topology_initialize(out_topology);
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    iree_task_topology_group_t* group = &out_topology->groups[i];
    iree_task_topology_group_initialize(i, group);
  }
  out_topology->group_count = group_count;

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_task_topology_initialize_from_thread_affinities(
    iree_host_size_t group_count,
    const iree_thread_affinity_t* group_affinities,
    iree_task_topology_t* out_topology) {
  // Today we have a fixed limit on the number of groups within a particular
  // topology.
  if (group_count >= IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many groups specified (%" PRIhsz
                            " provided for a max capacity of %zu)",
                            group_count, IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, group_count);

  // Initialize each group with the given affinities.
  iree_task_topology_initialize(out_topology);
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    iree_task_topology_group_t* group = &out_topology->groups[i];
    iree_task_topology_group_initialize(i, group);
    group->ideal_thread_affinity = group_affinities[i];
  }
  out_topology->group_count = group_count;

  // Try to use platform support to set the constructive sharing masks.
  // No-op if the platform support is not available.
  iree_status_t status =
      iree_task_topology_fixup_constructive_sharing_masks(out_topology);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_task_topology_initialize_from_logical_cpu_set_string(
    iree_string_view_t cpu_id_set, iree_task_topology_t* out_topology) {
  if (iree_string_view_is_empty(cpu_id_set)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one CPU ID must be provided");
  }
  iree_host_size_t count = 1;
  for (iree_host_size_t i = 0; i < cpu_id_set.size; ++i) {
    if (cpu_id_set.data[i] == ',') ++count;
  }
  uint32_t* cpu_ids = (uint32_t*)iree_alloca(count * sizeof(uint32_t));
  memset(cpu_ids, 0, count * sizeof(uint32_t));
  iree_host_size_t cpu_count = 0;
  while (!iree_string_view_is_empty(cpu_id_set)) {
    iree_string_view_t cpu_id_string = iree_string_view_empty();
    iree_string_view_split(cpu_id_set, ',', &cpu_id_string, &cpu_id_set);
    if (!iree_string_view_atoi_uint32(iree_string_view_trim(cpu_id_string),
                                      &cpu_ids[cpu_count++])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "'%.*s' not a valid CPU ID",
                              (int)cpu_id_string.size, cpu_id_string.data);
    }
  }
  return iree_task_topology_initialize_from_logical_cpu_set(cpu_count, cpu_ids,
                                                            out_topology);
}
