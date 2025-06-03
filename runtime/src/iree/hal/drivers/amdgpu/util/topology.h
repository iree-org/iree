// Copyright 2025 The IREE Authors
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

#define IREE_HAL_AMDGPU_MAX_CPU_AGENT 64
#define IREE_HAL_AMDGPU_MAX_GPU_AGENT 64

// Defines a system topology specifying which agents are to be used by the HAL.
//
// Today many internal structures assume at most 64 CPU agents and 64 GPU
// agents. These are possible to change but keeping them as something we can
// represent with a 64-bit bitfield avoids the need for a substantial number of
// dynamic allocations or dynamically sized structures. If we ever exceed this
// maximum we'll need to build a composed type that is compile configuration
// controlled. Since each agent is meant to represent an entire physical node
// instead of cores/etc within those nodes we'll need a 64-socket NUMA machine
// before we reach the current limit. For reference the Linux
// CPU_SET/CPU_SETSIZE is 1024 and that's a core count. Large servers may have
// many large devices that are subdivided and we're likely to want to extend for
// those cases. To support this we would need to consistently use macros for
// masking agents.
typedef struct iree_hal_amdgpu_topology_t {
  // Total number of agents of all types.
  iree_host_size_t all_agent_count;
  // All agents of all types.
  // This is primarily used to pass to APIs like hsa_amd_agents_allow_access.
  hsa_agent_t
      all_agents[IREE_HAL_AMDGPU_MAX_CPU_AGENT + IREE_HAL_AMDGPU_MAX_GPU_AGENT];
  // Total number of CPU agents specified.
  iree_host_size_t cpu_agent_count;
  // CPU agents servicing GPU requests.
  hsa_agent_t cpu_agents[IREE_HAL_AMDGPU_MAX_CPU_AGENT];
  // Total number of GPU agents specified.
  iree_host_size_t gpu_agent_count;
  // GPU agents that will be created as HAL queues.
  hsa_agent_t gpu_agents[IREE_HAL_AMDGPU_MAX_GPU_AGENT];
  // Logical scheduler queues per GPU agent. Uniform across all agents.
  iree_host_size_t gpu_agent_queue_count;
  // `gpu_agents` ordinal to `cpu_agents` ordinal map used to assign GPUs to
  // CPUs in cases where there may be multiple in a system.
  uint8_t gpu_cpu_map[IREE_HAL_AMDGPU_MAX_GPU_AGENT];
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
    iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    iree_host_size_t* out_index);

// Adds a GPU agent to the topology using the given CPU agent as its host.
// Ignored if the GPU agent has already been added. Adds the given CPU agent
// only if the GPU agent was added.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_insert_gpu_agent(
    iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t gpu_agent,
    hsa_agent_t cpu_agent);

// Adds a GPU agent to the topology and assigns it to the nearest CPU agent as
// indicated by the HSA runtime.
// Ignored if the GPU agent has already been added. Adds the nearest CPU agent
// if it has not already been added.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_topology_insert_gpu_agent_with_nearest_cpu_agent(
    iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t gpu_agent);

// Returns |out_is_compatible| indicating whether the provided agent is
// compatible with existing agents in the topology. Compatibility is defined as
// "roughly equivalent for all the things we care about" and may mean "exactly
// the same" for some agent types.
// Returns OK with compatibility in the provided flag.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_topology_query_agent_compatibility(
    iree_hal_amdgpu_topology_t* topology,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    bool* out_is_compatible);

// Verifies that the devices added to the topology are compatible.
// Currently we require that all GPUs are the same within a single logical HAL
// device.
//
// We could support a mix and pick the lowest-common-denominator so long as all
// ISAs are compatible - several place in the code querying agent properties
// would need to be changed to have a "lead" GPU agent that was used for
// capabilities/limits instead of using the physical GPU they are today.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_topology_verify(const iree_hal_amdgpu_topology_t* topology,
                                const iree_hal_amdgpu_libhsa_t* libhsa);

//===----------------------------------------------------------------------===//
// Topology Creation
//===----------------------------------------------------------------------===//

// Initializes |out_topology| with the defaults as specified by either the HSA
// API or environment variables (`ROCR_VISIBLE_DEVICES`).
//
// `ROCR_VISIBLE_DEVICES` is documented here:
// https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html
//
// Example:
//  $ export ROCR_VISIBLE_DEVICES="0,GPU-DEADBEEFDEADBEEF"
//
// iree_hal_amdgpu_topology_deinitialize must be called to clean up.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_topology_initialize_with_defaults(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_topology_t* out_topology);

// Initializes |out_topology| with a set of devices as specified in the provided
// |path|.
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
    iree_hal_amdgpu_topology_t* out_topology);

typedef uint64_t iree_hal_amdgpu_gpu_agent_mask_t;
static_assert(sizeof(uint64_t) * 8 >= IREE_HAL_AMDGPU_MAX_GPU_AGENT,
              "agent mask must have capacity for at least "
              "IREE_HAL_AMDGPU_MAX_GPU_AGENT entries");

// Initializes |out_topology| with a set of GPU agents indicated by the
// |gpu_agent_mask| bitfield.
//
// Each agent bit matches the HSA agent ordinal within the GPU devices
// enumerated from the runtime. `ROCR_VISIBLE_DEVICES` will be used to filter
// the visible agents and change the ordinal collection;
// `ROCR_VISIBLE_DEVICES=6,7,8` + device_mask=0b110 = devices 7 and 8.
//
// iree_hal_amdgpu_topology_deinitialize must be called to clean up.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_topology_initialize_from_gpu_agent_mask(
    const iree_hal_amdgpu_libhsa_t* libhsa, uint64_t gpu_agent_mask,
    iree_hal_amdgpu_topology_t* out_topology);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_TOPOLOGY_H_
