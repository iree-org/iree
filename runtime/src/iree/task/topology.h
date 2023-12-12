// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_TOPOLOGY_H_
#define IREE_TASK_TOPOLOGY_H_

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/threading.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// NUMA queries
//===----------------------------------------------------------------------===//

// A NUMA node or processor group ordinal.
typedef uint32_t iree_task_topology_node_id_t;

// Use any NUMA node (usually the first).
#define IREE_TASK_TOPOLOGY_NODE_ID_ANY ((iree_task_topology_node_id_t)-1)

// Returns the total number of NUMA nodes in the system or 1 if the query is
// not available on the platform.
iree_host_size_t iree_task_topology_query_node_count(void);

// Returns the NUMA node ID of the currently executing thread or 0 if the query
// is not available on the platform.
iree_task_topology_node_id_t iree_task_topology_query_current_node(void);

//===----------------------------------------------------------------------===//
// Topology group (worker thread(s) assigned to a processor)
//===----------------------------------------------------------------------===//

// A bitmask indicating which other groups from 0 to N may constructively share
// caches. For example, a value of 0b1100 indicates that group 2 and 3 share.
typedef uint64_t iree_task_topology_group_mask_t;

#define IREE_TASK_TOPOLOGY_GROUP_MASK_ALL UINT64_MAX
#define IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT \
  (sizeof(iree_task_topology_group_mask_t) * 8)

// Total cache sizes (that we care about).
// More information may be available but we shouldn't be specializing on it
// unless absolutely required. Values should ideally be a power-of-two if
// that's what the hardware has. Values of 0 indicate the particular cache is
// not present (or not queried).
typedef struct iree_task_topology_caches_t {
  uint32_t l1_data;
  uint32_t l2_data;
  uint32_t l3_data;
} iree_task_topology_caches_t;

// Information about a particular group within the topology.
// Groups may be of varying levels of granularity even within the same topology
// based on how the topology is defined.
typedef struct iree_task_topology_group_t {
  // Group index within the topology matching a particular bit in
  // iree_task_topology_group_mask_t.
  uint8_t group_index;

  // A name assigned to executor workers used for logging/tracing.
  char name[32 - /*group_index*/ 1];

  // Logical processor index.
  uint32_t processor_index;

  // Total cache sizes (that we care about).
  iree_task_topology_caches_t caches;

  // Ideal thread affinity for threads within this group.
  // All threads within the group share the same affinity and this is what
  // allows us to model Simultaneous Multi-Threading (SMT) (aka hyperthreading).
  iree_thread_affinity_t ideal_thread_affinity;

  // A bitmask of other group indices that share some level of the cache
  // hierarchy. Workers of this group are more likely to constructively share
  // some cache levels higher up with these other groups. For example, if the
  // workers in a group all share an L2 cache then the groups indicated here may
  // all share the same L3 cache.
  iree_task_topology_group_mask_t constructive_sharing_mask;
} iree_task_topology_group_t;

// Initializes |out_group| with a |group_index| derived name.
void iree_task_topology_group_initialize(uint8_t group_index,
                                         iree_task_topology_group_t* out_group);

//===----------------------------------------------------------------------===//
// Topology
//===----------------------------------------------------------------------===//

// Task system topology information used to define the workers within an
// executor.
//
// Topologies are used to statically configure task executors by defining the
// total number of workers in the worker pool and how those workers map to
// hardware compute resources.
//
// Users can allocate topologies, populate them with zero or more groups, and
// then pass them to the executor to construct the desired configuration. To
// ease testing and debugging topologies can be formatted as string values and
// round tripped through flags, though obviously the value of such encodings are
// machine-dependent.
//
// Several helper constructors are available that query the machine topology
// and attempt to derive some (hopefully) useful task system topology from it.
// We can add the more common heuristics over time to the core and leave the
// edge cases for applications to construct.
typedef struct iree_task_topology_t {
  iree_host_size_t group_count;
  iree_task_topology_group_t groups[IREE_TASK_EXECUTOR_MAX_WORKER_COUNT];
} iree_task_topology_t;

// Initializes an empty task topology.
void iree_task_topology_initialize(iree_task_topology_t* out_topology);

// Deinitializes a topology structure.
void iree_task_topology_deinitialize(iree_task_topology_t* topology);

// Parses a serialized topology in string form.
iree_status_t iree_task_topology_parse(iree_string_view_t value,
                                       iree_task_topology_t* out_topology);

// Formats the topology as a string value that can be parsed with
// iree_task_topology_parse.
bool iree_task_topology_format(const iree_task_topology_t* topology,
                               iree_host_size_t buffer_capacity, char* buffer,
                               iree_host_size_t* out_buffer_length);

// Returns the group capacity in the topology structure.
iree_host_size_t iree_task_topology_group_capacity(
    const iree_task_topology_t* topology);

// Returns the total group count defined by the topology.
iree_host_size_t iree_task_topology_group_count(
    const iree_task_topology_t* topology);

// Returns the group information for the given group index.
const iree_task_topology_group_t* iree_task_topology_get_group(
    const iree_task_topology_t* topology, iree_host_size_t group_index);

// Pushes a new group onto the topology set.
// The provided group data will be copied into the topology structure.
iree_status_t iree_task_topology_push_group(
    iree_task_topology_t* topology, const iree_task_topology_group_t* group);

//===----------------------------------------------------------------------===//
// Topology initialization helpers
//===----------------------------------------------------------------------===//

// Initializes a topology with the specified number of groups.
// 0 is a valid value, indicating that only donated threads will be used to
// perform work. Groups will have no specific affinity and rely on the OS
// scheduler to ensure they are distributed in a meaningful way; this generally
// works out as threads created within a process are usually rotated across
// preferred processors by default.
void iree_task_topology_initialize_from_group_count(
    iree_host_size_t group_count, iree_task_topology_t* out_topology);

// Initializes a topology with the given groups each assigned a platform thread
// affinity. See `iree_thread_affinity_t` for more information about how to
// properly initialize the thread affinities for each platform.
iree_status_t iree_task_topology_initialize_from_thread_affinities(
    iree_host_size_t group_count,
    const iree_thread_affinity_t* group_affinities,
    iree_task_topology_t* out_topology);

// Initializes a topology with one group for each logical CPU specified.
//
// The logical CPU IDs are in the platform-defined flattened domain of 0 to
// the total number of logical processors in the system such as those returned
// by `lscpu --extended`/lstopo/the bit index in cpu_set_t. The same ID is used
// on the file-based access in e.g. `/sys/devices/system/cpu/cpu<cpu_id>/`.
iree_status_t iree_task_topology_initialize_from_logical_cpu_set(
    iree_host_size_t cpu_count, const uint32_t* cpu_ids,
    iree_task_topology_t* out_topology);

// Initializes a topology with one group for each logical CPU specified in a
// comma-delimited list.
// See iree_task_topology_initialize_from_logical_cpu_set for more information.
iree_status_t iree_task_topology_initialize_from_logical_cpu_set_string(
    iree_string_view_t cpu_id_set, iree_task_topology_t* out_topology);

// Selects what core types in a heterogeneous core cluster are used.
// This maps to x86 efficiency/performance cores and ARM big.LITTLE cores.
//
// Hosting applications can decide whether they want low power consumption/less
// contention on high performance cores by forcing only low performance cores
// or predictable(ish) low latency by forcing only high performance cores. On
// homogeneous core clusters, where wall-time is the primary metric, or where
// contention is unlikely selecting all cores can usually result in the lowest
// latency. Each application with each set of programs will need to evaluate for
// themselves what to use based on their duty cycle, concurrently issued work,
// and user experience.
typedef enum iree_task_topology_performance_level_e {
  // Selects all cores.
  IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY = 0,
  // Selects "E(fficiency)" cores that favor lower power/thermal load.
  IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_LOW,
  // Selects "P(erformance)" cores that favor higher power/thermal load.
  IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_HIGH,
} iree_task_topology_performance_level_t;

// Initializes a topology with one group for each physical core with the given
// NUMA |node_id| (usually package or cluster). Up to |max_core_count| physical
// cores will be selected from the node.
iree_status_t iree_task_topology_initialize_from_physical_cores(
    iree_task_topology_node_id_t node_id,
    iree_task_topology_performance_level_t performance_level,
    iree_host_size_t max_core_count, iree_task_topology_t* out_topology);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_TOPOLOGY_H_
