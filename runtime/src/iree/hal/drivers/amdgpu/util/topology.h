// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_TOPOLOGY_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_TOPOLOGY_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_topology_t
//===----------------------------------------------------------------------===//

// Defines a system topology specifying which agents are to be used by the HAL.
typedef struct iree_hal_amdgpu_topology_t {
  // Total number of agents of all types.
  iree_host_size_t all_agent_count;
  // All agents of all types.
  // This is primarily used to pass to APIs like hsa_amd_agents_allow_access.
  hsa_agent_t all_agents[128];
  // Total number of CPU agents specified.
  iree_host_size_t cpu_agent_count;
  // CPU agents servicing GPU requests.
  hsa_agent_t cpu_agents[64];
  // Total number of GPU agents specified.
  iree_host_size_t gpu_agent_count;
  // GPU agents that will be created as HAL queues.
  hsa_agent_t gpu_agents[64];
  // Logical scheduler queues per GPU agent. Uniform across all agents.
  iree_host_size_t gpu_agent_queue_count;
  // `gpu_agents` ordinal to `cpu_agents` ordinal map used to assign GPUs to
  // CPUs in cases where there may be multiple in a system.
  uint8_t gpu_cpu_map[64];
} iree_hal_amdgpu_topology_t;

// Initializes an empty topology in |out_topology|.
// iree_hal_amdgpu_topology_deinitialize must be called to clean up.
IREE_API_EXPORT void iree_hal_amdgpu_topology_initialize(
    iree_hal_amdgpu_topology_t* out_topology);

// Deinitializes |topology|.
IREE_API_EXPORT void iree_hal_amdgpu_topology_deinitialize(
    iree_hal_amdgpu_topology_t* topology);

// Adds a CPU agent to the topology.
// Ignored if the agent has already been added.
// Returns the index of the newly inserted or existing agent in |out_index|.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_insert_cpu_agent(
    iree_hal_amdgpu_topology_t* topology, hsa_agent_t cpu_agent,
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_host_size_t* out_index);

// Adds a GPU agent to the topology using the given CPU agent as its host.
// Ignored if the GPU agent has already been added. Adds the CPU agent if it has
// not already been added.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_insert_gpu_agent(
    iree_hal_amdgpu_topology_t* topology, hsa_agent_t gpu_agent,
    hsa_agent_t cpu_agent, const iree_hal_amdgpu_libhsa_t* libhsa);

// Adds a GPU agent to the topology and assigns it to the nearest CPU agent as
// indicated by the HSA runtime.
// Ignored if the GPU agent has already been added. Adds the CPU agent if it has
// not already been added.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_topology_insert_gpu_agent_with_nearest_cpu_agent(
    iree_hal_amdgpu_topology_t* topology, hsa_agent_t gpu_agent,
    const iree_hal_amdgpu_libhsa_t* libhsa);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_topology_t helpers
//===----------------------------------------------------------------------===//

// Initializes |out_topology| with the defaults as specified by either the HSA
// API or environment variables (`ROCR_VISIBLE_DEVICES`).
//
// `ROCR_VISIBLE_DEVICES` is documented here:
// https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html
//
// iree_hal_amdgpu_topology_deinitialize must be called to clean up.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_initialize_with_defaults(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_topology_t* out_topology);

// Initializes |out_topology| with a set of devices as specified in the provided
// |path| and |params|.
//
// The path may contain one or more comma separated device identifiers in
// various forms. An empty or wildcard (`*`) path will initialize with defaults.
// Supported identifiers are `GPU-0e12865a3bf5b7ab`-style IDs (matching
// `HSA_AMD_AGENT_INFO_UUID`) or GPU ordinals (`1` being the second GPU found in
// the agent list) as with `ROCR_VISIBLE_DEVICES`.
//
// Note that system-level variables like `ROCR_VISIBLE_DEVICES` will still
// have an effect and the list of available devices and their ordinals is
// derived from that.
//
// iree_hal_amdgpu_topology_deinitialize must be called to clean up.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_initialize_from_path(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_string_view_t path,
    const iree_string_pair_list_t params,
    iree_hal_amdgpu_topology_t* out_topology);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_TOPOLOGY_H_
