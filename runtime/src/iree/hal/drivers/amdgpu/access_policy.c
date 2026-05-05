// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/access_policy.h"

static bool iree_hal_amdgpu_access_agent_list_contains(
    const iree_hal_amdgpu_access_agent_list_t* agent_list, hsa_agent_t agent) {
  for (uint32_t i = 0; i < agent_list->count; ++i) {
    if (agent_list->values[i].handle == agent.handle) return true;
  }
  return false;
}

static iree_status_t iree_hal_amdgpu_access_agent_list_append_unique(
    iree_hal_amdgpu_access_agent_list_t* agent_list, hsa_agent_t agent) {
  if (iree_hal_amdgpu_access_agent_list_contains(agent_list, agent)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(agent_list->count >= IREE_ARRAYSIZE(agent_list->values))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU access agent list capacity exceeded");
  }
  agent_list->values[agent_list->count++] = agent;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_access_agent_list_resolve(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_amdgpu_access_agent_list_t* out_agent_list) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_agent_list);
  memset(out_agent_list, 0, sizeof(*out_agent_list));

  if (IREE_UNLIKELY(queue_affinity_domain.physical_device_count >
                    topology->gpu_agent_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU queue affinity domain device count %" PRIhsz
                            " exceeds topology GPU agent count %" PRIhsz,
                            queue_affinity_domain.physical_device_count,
                            topology->gpu_agent_count);
  }

  iree_hal_amdgpu_queue_affinity_physical_device_set_t physical_device_set;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_select_physical_devices(
      queue_affinity_domain, queue_affinity, &physical_device_set));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t physical_device_ordinal = 0;
       physical_device_ordinal < topology->gpu_agent_count &&
       iree_status_is_ok(status);
       ++physical_device_ordinal) {
    if (!iree_all_bits_set(physical_device_set.physical_device_mask,
                           ((uint64_t)1) << physical_device_ordinal)) {
      continue;
    }
    const iree_host_size_t cpu_agent_ordinal =
        topology->gpu_cpu_map[physical_device_ordinal];
    if (IREE_UNLIKELY(cpu_agent_ordinal >= topology->cpu_agent_count)) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "AMDGPU topology maps GPU agent ordinal %" PRIhsz
                           " to invalid CPU agent ordinal %" PRIhsz,
                           physical_device_ordinal, cpu_agent_ordinal);
      break;
    }

    status = iree_hal_amdgpu_access_agent_list_append_unique(
        out_agent_list, topology->gpu_agents[physical_device_ordinal]);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_access_agent_list_append_unique(
          out_agent_list, topology->cpu_agents[cpu_agent_ordinal]);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_access_allow_agent_list(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_access_agent_list_t* agent_list, const void* ptr) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(agent_list);
  IREE_ASSERT_ARGUMENT(ptr);
  return iree_hsa_amd_agents_allow_access(IREE_LIBHSA(libhsa),
                                          agent_list->count, agent_list->values,
                                          /*flags=*/NULL, ptr);
}

iree_status_t iree_hal_amdgpu_access_lock_host_allocation(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_access_agent_list_t* agent_list, void* host_ptr,
    iree_device_size_t length, void** out_agent_ptr) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(agent_list);
  IREE_ASSERT_ARGUMENT(host_ptr);
  IREE_ASSERT_ARGUMENT(out_agent_ptr);
  return iree_hsa_amd_memory_lock(IREE_LIBHSA(libhsa), host_ptr, (size_t)length,
                                  (hsa_agent_t*)agent_list->values,
                                  (int)agent_list->count, out_agent_ptr);
}
