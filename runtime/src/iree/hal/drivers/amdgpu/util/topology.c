// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_topology_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_amdgpu_topology_initialize(
    iree_hal_amdgpu_topology_t* out_topology) {
  IREE_ASSERT_ARGUMENT(out_topology);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_topology, 0, sizeof(*out_topology));

  out_topology->gpu_agent_queue_count = 1;

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_amdgpu_topology_deinitialize(
    iree_hal_amdgpu_topology_t* topology) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(topology, 0, sizeof(*topology));

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_insert_cpu_agent(
    iree_hal_amdgpu_topology_t* topology, hsa_agent_t cpu_agent,
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_host_size_t* out_index) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(libhsa);
  if (out_index) *out_index = -1;

  // Scan for the agent in the current topology.
  for (iree_host_size_t i = 0; i < topology->cpu_agent_count; ++i) {
    if (topology->cpu_agents[i].handle == cpu_agent.handle) {
      *out_index = i;
      return iree_ok_status();
    }
  }

  // Check capacity before mutating the topology.
  if (topology->cpu_agent_count + 1 >= IREE_ARRAYSIZE(topology->cpu_agents)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "max CPU agent count reached");
  }

  // Verify the agent is a supported CPU agent.
  hsa_device_type_t device_type = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      libhsa, cpu_agent, HSA_AGENT_INFO_DEVICE, &device_type));
  if (device_type != HSA_DEVICE_TYPE_CPU) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HSA agent is not a HSA_DEVICE_TYPE_CPU as required");
  }

  // Add CPU agent.
  iree_host_size_t cpu_agent_index = topology->cpu_agent_count++;
  topology->gpu_agents[cpu_agent_index] = cpu_agent;
  topology->all_agents[topology->all_agent_count++] = cpu_agent;

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_insert_gpu_agent(
    iree_hal_amdgpu_topology_t* topology, hsa_agent_t gpu_agent,
    hsa_agent_t cpu_agent, const iree_hal_amdgpu_libhsa_t* libhsa) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(libhsa);

  // Ignore if the GPU agent has already been added.
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    if (topology->gpu_agents[i].handle == gpu_agent.handle) {
      return iree_ok_status();  // already present
    }
  }

  // Check capacity before mutating the topology.
  if (topology->gpu_agent_count + 1 >= IREE_ARRAYSIZE(topology->gpu_agents)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "max GPU agent count reached");
  }

  // Verify the agent is a supported GPU agent.
  hsa_device_type_t device_type = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      libhsa, gpu_agent, HSA_AGENT_INFO_DEVICE, &device_type));
  if (device_type != HSA_DEVICE_TYPE_GPU) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HSA agent is not a HSA_DEVICE_TYPE_GPU as required");
  }

  // Add dependent CPU agent (if needed).
  iree_host_size_t cpu_agent_index = -1;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_topology_insert_cpu_agent(
      topology, cpu_agent, libhsa, &cpu_agent_index));

  // Add GPU agent.
  iree_host_size_t gpu_agent_index = topology->gpu_agent_count++;
  topology->gpu_agents[gpu_agent_index] = gpu_agent;
  topology->all_agents[topology->all_agent_count++] = gpu_agent;

  // Update the GPU->CPU agent mapping.
  topology->gpu_cpu_map[gpu_agent_index] = (uint8_t)cpu_agent_index;

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_topology_insert_gpu_agent_with_nearest_cpu_agent(
    iree_hal_amdgpu_topology_t* topology, hsa_agent_t gpu_agent,
    const iree_hal_amdgpu_libhsa_t* libhsa) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(libhsa);

  // Lookup the nearest CPU agent to the GPU agent specified.
  hsa_agent_t cpu_agent;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      libhsa, gpu_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NEAREST_CPU,
      &cpu_agent));

  // Add GPU agent with the queried CPU agent.
  return iree_hal_amdgpu_topology_insert_gpu_agent(topology, gpu_agent,
                                                   cpu_agent, libhsa);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_topology_t helpers
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_available_agents_t {
  iree_host_size_t all_agent_count;
  hsa_agent_t all_agents[128];
  iree_host_size_t cpu_agent_count;
  hsa_agent_t cpu_agents[64];
  iree_host_size_t gpu_agent_count;
  hsa_agent_t gpu_agents[64];
} iree_hal_amdgpu_available_agents_t;

static hsa_status_t iree_hal_amdgpu_iterate_available_agent(hsa_agent_t agent,
                                                            void* user_data) {
  iree_hal_amdgpu_available_agents_t* agents =
      (iree_hal_amdgpu_available_agents_t*)user_data;
  if (agents->all_agent_count == IREE_ARRAYSIZE(agents->all_agents)) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  agents->all_agents[agents->all_agent_count++] = agent;
  return HSA_STATUS_SUCCESS;
}

// Queries all available agents in the system as reported to HSA.
// `ROCR_VISIBLE_DEVICES` will filter this list.
static iree_status_t iree_hal_amdgpu_query_available_agents(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_available_agents_t* out_agents) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_agents);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_agents, 0, sizeof(*out_agents));

  // Get all agents.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_iterate_agents(
              libhsa, iree_hal_amdgpu_iterate_available_agent, out_agents));

  // Categorize CPU and GPU agents.
  for (iree_host_size_t i = 0; i < out_agents->all_agent_count; ++i) {
    hsa_device_type_t device_type = 0;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hsa_agent_get_info(libhsa, out_agents->all_agents[i],
                                    HSA_AGENT_INFO_DEVICE, &device_type));
    switch (device_type) {
      case HSA_DEVICE_TYPE_CPU:
        if (out_agents->cpu_agent_count ==
            IREE_ARRAYSIZE(out_agents->cpu_agents)) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "max CPU agent count reached");
        }
        out_agents->cpu_agents[out_agents->cpu_agent_count++] =
            out_agents->all_agents[i];
        break;
      case HSA_DEVICE_TYPE_GPU:
        if (out_agents->gpu_agent_count ==
            IREE_ARRAYSIZE(out_agents->gpu_agents)) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "max GPU agent count reached");
        }
        out_agents->gpu_agents[out_agents->gpu_agent_count++] =
            out_agents->all_agents[i];
        break;
      default:
        break;  // ignored device type
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Returns the ordinal of the GPU agent with the given `GPU-XX` "UUID".
static iree_status_t iree_hal_amdgpu_available_agents_find_gpu_with_uuid(
    const iree_hal_amdgpu_available_agents_t* agents, iree_string_view_t uuid,
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_host_size_t* out_ordinal) {
  IREE_ASSERT_ARGUMENT(agents);
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_ordinal);
  *out_ordinal = -1;
  for (iree_host_size_t i = 0; i < agents->gpu_agent_count; ++i) {
    char agent_uuid[32];
    IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
        libhsa, agents->gpu_agents[i],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID, agent_uuid));
    if (iree_string_view_equal_case(uuid, iree_make_cstring_view(agent_uuid))) {
      *out_ordinal = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no GPU agent with UUID `%.*s` found", (int)uuid.size,
                          uuid.data);
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_initialize_with_defaults(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_topology_t* out_topology) {
  IREE_ASSERT_ARGUMENT(out_topology);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Query all available agents in the system.
  // This amortizes the cost of the queries and type checks during the lookup
  // below.
  iree_hal_amdgpu_available_agents_t agents;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_query_available_agents(libhsa, &agents));

  // Initialize an empty topology.
  iree_hal_amdgpu_topology_initialize(out_topology);

  // Add all visible GPU agents.
  // The user can override this set and its order with `ROCR_VISIBLE_DEVICES`.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < agents.gpu_agent_count; ++i) {
    status = iree_hal_amdgpu_topology_insert_gpu_agent_with_nearest_cpu_agent(
        out_topology, agents.gpu_agents[i], libhsa);
    if (!iree_status_is_ok(status)) break;
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_topology_deinitialize(out_topology);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_initialize_from_path(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_string_view_t path,
    const iree_string_pair_list_t params,
    iree_hal_amdgpu_topology_t* out_topology) {
  IREE_ASSERT_ARGUMENT(out_topology);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  // If no path was provided or a wildcard was specified use the default
  // behavior.
  if (iree_string_view_is_empty(path) ||
      iree_string_view_equal(path, IREE_SV("*"))) {
    iree_status_t status =
        iree_hal_amdgpu_topology_initialize_with_defaults(libhsa, out_topology);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // TODO(benvanik): use params to optionally filter devices (by nodes, types,
  // etc). For now the params are ignored. We'd scan the list for keys matching
  // what we use here and validate/fail early.

  // Query all available agents in the system.
  // This amortizes the cost of the queries and type checks during the lookup
  // below.
  iree_hal_amdgpu_available_agents_t agents;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_query_available_agents(libhsa, &agents));

  // Initialize an empty topology.
  iree_hal_amdgpu_topology_initialize(out_topology);

  // Add each GPU device specified.
  iree_status_t status = iree_ok_status();
  do {
    // Consume the next path fragment up to the next comma or end of string.
    iree_string_view_t fragment = iree_string_view_empty();
    iree_string_view_split(path, ',', &fragment, &path);

    // If the ID is an HSA "UUID" (which annoyingly isn't a UUID) then we need
    // to scan the devices to find the one that matches. Otherwise interpret the
    // string as an ordinal.
    if (iree_string_view_starts_with(fragment, IREE_SV("GPU-"))) {
      iree_host_size_t ordinal = 0;
      status = iree_hal_amdgpu_available_agents_find_gpu_with_uuid(
          &agents, fragment, libhsa, &ordinal);
      if (!iree_status_is_ok(status)) break;
      status = iree_hal_amdgpu_topology_insert_gpu_agent_with_nearest_cpu_agent(
          out_topology, agents.gpu_agents[ordinal], libhsa);
    } else {
      uint32_t ordinal = 0;
      if (!iree_string_view_atoi_uint32(fragment, &ordinal)) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "invalid device specifier `%.*s`; expected an "
                             "ordinal (`0`) or a device UUID (`GPU-XX...`)",
                             (int)fragment.size, fragment.data);
        break;
      }
      if (ordinal >= agents.gpu_agent_count) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "device ordinal `%u` out of range; %" PRIhsz
                                  " GPU devices available",
                                  ordinal, agents.gpu_agent_count);
        break;
      }
      status = iree_hal_amdgpu_topology_insert_gpu_agent_with_nearest_cpu_agent(
          out_topology, agents.gpu_agents[ordinal], libhsa);
    }
    if (!iree_status_is_ok(status)) break;
  } while (!iree_string_view_is_empty(path));

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_topology_deinitialize(out_topology);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
