// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_TOPOLOGY_CPUINFO_H_
#define IREE_TASK_TOPOLOGY_CPUINFO_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/task/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct cpuinfo_core;

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

#endif  // IREE_TASK_TOPOLOGY_CPUINFO_H_
