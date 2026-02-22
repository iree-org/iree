// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Locality domain for NUMA-aware resource placement.
//
// An affinity domain groups resources that share physical proximity: CPU cores
// (same complex, shared L3), memory controller (NUMA node), PCIe root complex,
// and attached devices (GPUs, NICs). The async layer uses affinities to:
//
//   - Place buffer pools on the correct NUMA node for zero-copy I/O
//   - Match proactor threads to the NICs/GPUs they serve
//   - Avoid cross-NUMA memory traffic (2-3x latency penalty on modern servers)
//
// Affinities are value types: they capture topology information at discovery
// time and are passed by pointer to creation functions. The async layer does
// not interpret the user_context field — it exists for the caller (e.g., HAL
// driver) to attach device-specific handles.
//
// ## Topology discovery
//
// iree_async_query_affinities() discovers the system's locality domains at
// runtime. On a typical multi-GPU server with 4 CPU complexes and 8 GPUs, this
// returns 4 affinities (one per complex, each serving 2 GPUs). The caller then
// creates per-domain buffer pools and proactor threads.
//
// On simple systems (single socket, no discrete GPU), one affinity is returned
// representing the entire machine.

#ifndef IREE_ASYNC_AFFINITY_H_
#define IREE_ASYNC_AFFINITY_H_

#include "iree/base/api.h"
#include "iree/base/threading/thread.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Affinity domain
//===----------------------------------------------------------------------===//

// Sentinel value for iree_async_affinity_t::numa_node indicating that NUMA
// placement is not applicable (single-node system or unknown topology).
// When this value is set, buffer pool allocations use the system allocator
// without NUMA-specific placement.
#define IREE_ASYNC_AFFINITY_NUMA_NODE_ANY UINT32_MAX

// A locality domain grouping CPU cores, memory, and attached devices.
// Used at pool and proactor creation time to ensure NUMA-local allocation.
// Value type — can be stack-allocated, embedded in structs, or copied freely.
typedef struct iree_async_affinity_t {
  // Unique ID within the system (assigned by iree_async_query_affinities).
  // Stable for the lifetime of the process (hardware topology doesn't change).
  uint32_t id;

  // CPU affinity for threads operating in this domain.
  // The group field corresponds to the NUMA node on Linux. Proactor threads
  // and pool-owning threads should be pinned to this affinity for optimal
  // cache and memory controller locality.
  iree_thread_affinity_t cpu_affinity;

  // NUMA node ID for memory allocation (mbind/set_mempolicy on Linux).
  // Set to IREE_ASYNC_AFFINITY_NUMA_NODE_ANY if NUMA is not applicable.
  // When set to a valid node ID, buffer pool allocations target that node
  // for NUMA-local placement.
  uint32_t numa_node;

  // Opaque user context for application-specific topology data (e.g., HAL
  // device handle, GPU context, NIC handle). Not interpreted by the async
  // layer — the caller sets this after query_affinities returns.
  void* user_context;
} iree_async_affinity_t;

// Returns an affinity with no locality constraint (any NUMA node, any CPU).
// Use when NUMA placement is not important or the system topology is unknown.
static inline iree_async_affinity_t iree_async_affinity_any(void) {
  iree_async_affinity_t affinity;
  memset(&affinity, 0, sizeof(affinity));
  affinity.numa_node = IREE_ASYNC_AFFINITY_NUMA_NODE_ANY;
  iree_thread_affinity_set_any(&affinity.cpu_affinity);
  return affinity;
}

// Queries system topology and returns one affinity per locality domain.
//
// On a typical multi-GPU server: one domain per CPU complex (shared L3 +
// local DRAM + attached PCIe devices). On single-socket systems: one domain.
//
// The caller owns the returned array and must free it with |allocator|.
// |out_count| is set to the number of domains found.
//
// Discovery uses platform-specific mechanisms:
//   Linux: /sys/devices/system/node/ (NUMA nodes) + PCIe topology
//   Other: falls back to a single domain covering the entire system
iree_status_t iree_async_query_affinities(
    iree_allocator_t allocator, iree_async_affinity_t** out_affinities,
    iree_host_size_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_AFFINITY_H_
