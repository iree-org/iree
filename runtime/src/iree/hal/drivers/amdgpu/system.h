// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SYSTEM_H_
#define IREE_HAL_DRIVERS_AMDGPU_SYSTEM_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/device_library.h"
#include "iree/hal/drivers/amdgpu/util/info.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_system_t
//===----------------------------------------------------------------------===//

// Options defining total system behavior.
typedef struct iree_hal_amdgpu_system_options_t {
  // Enable dispatch-level tracing (if device instrumentation is compiled in).
  uint64_t trace_execution : 1;
  // Force queues to run one entry at a time instead of overlapping or
  // aggressively scheduling queue entries out-of-order.
  uint64_t exclusive_execution : 1;
} iree_hal_amdgpu_system_options_t;

// A set of queried HSA regions and memory pools for a particular CPU agent.
// In a NUMA topology each CPU agent has memory that is faster to access and
// GPU agents attached to a particular CPU agent will require fewer hops to
// access the CPU memory. We try to allocate resources that may be used
// frequently on a particular cluster of CPU/GPU agents closest to the agents.
typedef struct iree_hal_amdgpu_host_memory_pools_t {
  // Memory pool used for various system-level resources.
  // Allocations from this pool must be accessible to all agents. Must have the
  // HSA_REGION_GLOBAL_FLAG_FINE_GRAINED region flag as memory within this
  // pool will be used cross-agent with atomics. This is likely to be located
  // in host memory and we take the hit for device accesses for the ability to
  // share access with the host. For resources we expect to only be accessed on
  // a particular agent we instead allocate from there.
  hsa_amd_memory_pool_t fine_pool;
  // Memory region used for various system-level resources.
  // Allocations from this pool must be accessible to all agents.
  // The more modern memory pool API and `shared_fine_pool` should be used
  // unless the particular HSA API requires a region.
  hsa_region_t fine_region;
  // TODO(benvanik): coarse/readonly host regions for bulk buffers/constants.
} iree_hal_amdgpu_host_memory_pools_t;

// An initialized AMDGPU system topology and agent resources.
// This is the authoritative list of available devices and their configuration.
//
// Thread-safe; systems are immutable once initialized.
typedef struct iree_hal_amdgpu_system_t {
  // Allocator used for host operations.
  iree_allocator_t host_allocator;

  // HSA API handle.
  iree_hal_amdgpu_libhsa_t libhsa;

  // /dev/kfd handle, if needed on the platform.
  // TODO(benvanik): drop this when HSA supports all of the ioctls we need.
  int kfd_fd;

  // System topology as visible to the HAL device. This may be a subset of
  // the devices available in the system.
  iree_hal_amdgpu_topology_t topology;

  // Cached system information queried from HSA or the platform.
  iree_hal_amdgpu_system_info_t info;

  // Options configuring system behavior.
  iree_hal_amdgpu_system_options_t options;

  // Loaded device library. Instantiated on all GPU devices in the topology.
  iree_hal_amdgpu_device_library_t device_library;

  // Memory pools for shared CPU/GPU agent access.
  iree_hal_amdgpu_host_memory_pools_t host_memory_pools[/*cpu_agent_count*/];
} iree_hal_amdgpu_system_t;

// Allocates a system in |out_system| with the given |topology|.
// The provided |libhsa| and |topology| will be copied and a reference to the
// HSA library will be retained for the lifetime of the system. Assumes that
// |topology| has been verified as valid.
iree_status_t iree_hal_amdgpu_system_allocate(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_system_options_t options, iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_t** out_system);

// Frees |system| and releases any held resources.
void iree_hal_amdgpu_system_free(iree_hal_amdgpu_system_t* system);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SYSTEM_H_
