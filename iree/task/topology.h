// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_TASK_TOPOLOGY_H_
#define IREE_TASK_TOPOLOGY_H_

#include <limits.h>

#include "iree/base/api.h"
#include "iree/base/internal/threading.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct cpuinfo_core;

// A bitmask indicating which other groups from 0 to N may constructively share
// caches. For example, a value of 0b1100 indicates that group 2 and 3 share.
typedef uint64_t iree_task_topology_group_mask_t;

#define IREE_TASK_TOPOLOGY_GROUP_MASK_ALL UINT64_MAX
#define IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT \
  (sizeof(iree_task_topology_group_mask_t) * 8)

// Information about a particular group within the topology.
// Groups may be of varying levels of granularity even within the same topology
// based on how the topology is defined.
typedef struct {
  // Group index within the topology matching a particular bit in
  // iree_task_topology_group_mask_t.
  uint8_t group_index;

  // A name assigned to executor workers used for logging/tracing.
  char name[16];

  // Processor index in the cpuinfo set.
  uint32_t processor_index;

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
typedef struct {
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

// Initializes a topology with the specified number of groups.
// 0 is a valid value, indicating that only donated threads will be used to
// perform work. Groups will have no specific affinity and rely on the OS
// scheduler to ensure they are distributed in a meaningful way; this generally
// works out as threads created within a process are usually rotated across
// preferred processors by default.
void iree_task_topology_initialize_from_group_count(
    iree_host_size_t group_count, iree_task_topology_t* out_topology);

// Initializes a topology with one group for each physical core in the machine.
//
// If detailed cache information is not available this is a decent
// approximation that can be used as a fallback.
void iree_task_topology_initialize_from_physical_cores(
    iree_host_size_t max_core_count, iree_task_topology_t* out_topology);

// Initializes a topology with one group for each physical core in the machine
// with the given microarchitecture specified as a cpuinfo_uarch value.
//
// If detailed uarch information is not available this falls back to the same
// behavior as iree_task_topology_initialize_from_physical_cores.
void iree_task_topology_initialize_from_physical_cores_with_uarch(
    uint32_t cpuinfo_uarch, iree_host_size_t max_core_count,
    iree_task_topology_t* out_topology);

// Returns true if the given |core| passes the filter and should be included.
// |user_data| is the value passed alongside the filter function.
typedef bool (*iree_task_topology_core_filter_t)(
    const struct cpuinfo_core* core, uintptr_t user_data);

// Initializes a topology with one group for each core that matches |filter_fn|.
//
// If cpuinfo is not available this falls back to the same behavior as
// iree_task_topology_initialize_from_physical_cores.
void iree_task_topology_initialize_from_physical_cores_with_filter(
    iree_task_topology_core_filter_t filter_fn, uintptr_t filter_fn_data,
    iree_host_size_t max_core_count, iree_task_topology_t* out_topology);

// Initializes a topology with one group for each unique L2 cache group across
// all available cores. This optimizes for temporal and spatial cache locality
// but may suffer from oversubscription if there are other processes trying to
// use the same cores.
//
// If detailed cache information is not available this falls back to the same
// behavior as iree_task_topology_initialize_from_physical_cores.
void iree_task_topology_initialize_from_unique_l2_cache_groups(
    iree_host_size_t max_group_count, iree_task_topology_t* out_topology);

// TODO(#4654): more helpers and better defaults for the platforms we support.
// Users can always make their own but just using these is the common path.
// Ideas:
// - _from_unique_l2_cache_groups but with a min/max count (N% utilization)
// - cluster filtering (big/little cores on ARM)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_TOPOLOGY_H_
