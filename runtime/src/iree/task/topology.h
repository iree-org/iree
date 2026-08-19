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
#include "iree/base/threading/thread.h"
#include "iree/task/affinity_set.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Node/partitioning terminology
// NUMA node
//   A memory-locality domain. This is what a "node" means by default and what
//   the affinity group hint feeds to set_mempolicy. Sourced from
//   /sys/devices/system/node/ (sysfs), falling back to the physical package
//   when that hierarchy is unavailable.
//
// package
//   A physical socket (topology/physical_package_id). Coarser than a NUMA node
//   on SNC/NPS systems.
//
// cluster
//   A finer, on-die core grouping.
//   backend-specific.
//
// The task system partitions executors by NUMA node.

// Formats the backend's raw hardware ids for |processor| into |buffer| as a
// human-readable fragment (e.g. "cluster_id=8, physical_package_id=0") for
// diagnostics. Returns the number of characters written, or 0 when the backend
// exposes no additional ids.
iree_host_size_t iree_task_topology_format_processor_debug_ids(
    uint32_t processor, iree_host_size_t buffer_capacity, char* buffer);

// Redirects topology discovery to a captured or synthetic snapshot of a
// machine's topology data instead of the live host, for testing and debugging.
//
// Returns UNIMPLEMENTED on backends that read no redirectable data source, and
// NOT_FOUND if |path| does not hold a snapshot the backend can use.
iree_status_t iree_task_topology_set_snapshot_path(const char* path);

// Maximum number of nodes the topology system enumerates or a
// --task_topology_nodes selection can name. Machines with more nodes are
// rejected.
//
// Linux's default MAX_NUMNODES on x86_64 (CONFIG_NODES_SHIFT=6, so
// 1<<6 nodes). MAXSMP kernels raise NODES_SHIFT to 10 (1024 nodes); such a
// machine is reported as RESOURCE_EXHAUSTED.
#define IREE_TASK_TOPOLOGY_MAX_NODES 64

// A NUMA node or processor group ordinal.
typedef uint32_t iree_task_topology_node_id_t;

// Use any NUMA node (usually the first).
#define IREE_TASK_TOPOLOGY_NODE_ID_ANY ((iree_task_topology_node_id_t) - 1)

// Enumerates the ids of all NUMA nodes, writing up to |capacity| ids into
// |out_ids| and returning the total node count.
// Ids may be sparse/non-contiguous.
// Always reports at least one node; when the query is unavailable that node is
// id 0.
iree_host_size_t iree_task_topology_query_node_ids(
    iree_host_size_t capacity, iree_task_topology_node_id_t* out_ids);

// Returns the total number of NUMA nodes in the system, or 1 if the query is
// not available on the platform.
iree_host_size_t iree_task_topology_query_node_count(void);

// Writes the dense id range [0, |count|) into |out_ids| (up to |capacity|) and
// returns |count|. Helper for backends whose node ids are always dense.
iree_host_size_t iree_task_topology_dense_node_ids(
    iree_host_size_t count, iree_host_size_t capacity,
    iree_task_topology_node_id_t* out_ids);

// Returns the NUMA node id of the currently executing thread or 0 if the query
// is not available on the platform.
iree_task_topology_node_id_t iree_task_topology_query_current_node(void);

//===----------------------------------------------------------------------===//
// Topology group (worker thread(s) assigned to a processor)
//===----------------------------------------------------------------------===//

// A bitmask indicating which other groups from 0 to N may constructively share
// caches. For example, bits 2 and 3 set indicates that groups 2 and 3 share.
// Uses the same multi-word type as worker affinity sets (one bit per group,
// max groups = max workers).
typedef iree_task_affinity_set_t iree_task_topology_group_mask_t;

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
  // NUMA node the workers' memory should be bound to, as passed to
  // mbind/set_mempolicy, or IREE_TASK_TOPOLOGY_NODE_ID_ANY if unspecified.
  iree_task_topology_node_id_t numa_node_id;
  iree_host_size_t group_count;
  iree_task_topology_group_t groups[IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT];
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
  // Selects "P(performance)" cores that favor higher power/thermal load.
  IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_HIGH,
} iree_task_topology_performance_level_t;

// Strategy for distributing cores across cache domains (CCXs) within the
// selected NUMA node(s). NUMA locality is controlled by the node_id parameter -
// use IREE_TASK_TOPOLOGY_NODE_ID_ANY to select cores from any node, or specify
// a specific node_id to limit cores to that NUMA node.
typedef enum iree_task_topology_distribution_e {
  // Fill cache domains sequentially before moving to the next.
  // Maximizes L3 cache locality - best for compute-intensive workloads where
  // cache hit rate is critical (small working sets, frequent data reuse).
  // Example: 10 cores on 2 CCXs of 8 → 8 cores on CCX0, 2 cores on CCX1.
  IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT = 0,
  // Scatter cores across cache domains using round-robin distribution.
  // Maximizes memory bandwidth by utilizing multiple memory controllers.
  // Best for memory-bound workloads (large matmuls, streaming operations).
  // Example: 10 cores on 2 CCXs → 0,8,1,9,2,10,3,11,4,12 (alternating).
  IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER = 1,
} iree_task_topology_distribution_t;

// Initializes a topology with one group for each physical core belonging to
// NUMA |node_id|, or from every node when IREE_TASK_TOPOLOGY_NODE_ID_ANY is
// passed. Up to |max_core_count| physical cores are selected and distributed
// according to the |distribution| strategy across cache domains.
iree_status_t iree_task_topology_initialize_from_physical_cores(
    iree_task_topology_node_id_t node_id,
    iree_task_topology_performance_level_t performance_level,
    iree_task_topology_distribution_t distribution,
    iree_host_size_t max_core_count, iree_task_topology_t* out_topology);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_TOPOLOGY_H_
