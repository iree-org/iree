// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_API_H_
#define IREE_TASK_API_H_

#include "iree/base/api.h"
#include "iree/task/executor.h"  // IWYU pragma: export
#include "iree/task/topology.h"  // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Node selection
//===----------------------------------------------------------------------===//

// A resolved --task_topology_nodes selection: an explicit list of NUMA node ids
// to build one topology (executor) for each.
//
// Node ids may be sparse, this holds the actual ids rather than a dense
// [0, count) range.
typedef struct iree_task_node_selection_t {
  iree_host_size_t count;
  iree_task_topology_node_id_t ids[IREE_TASK_TOPOLOGY_MAX_NODES];
} iree_task_node_selection_t;

// Resolves a --task_topology_nodes |spec|.
iree_status_t iree_task_topology_select_nodes(
    iree_string_view_t spec, iree_task_node_selection_t* out_selection);

//===----------------------------------------------------------------------===//
// Flag parsing
//===----------------------------------------------------------------------===//

// Initializes |out_options| from the command line flags.
// Used in place of iree_task_executor_options_initialize.
iree_status_t iree_task_executor_options_initialize_from_flags(
    iree_task_executor_options_t* out_options);

// Initializes |out_topology| from the command line flags.
// Depending on the mode flags the topology is pinned to NUMA |node_id|; pass
// IREE_TASK_TOPOLOGY_NODE_ID_ANY to span all cores.
iree_status_t iree_task_topology_initialize_from_flags(
    iree_task_topology_node_id_t node_id, iree_task_topology_t* out_topology);

//===----------------------------------------------------------------------===//
// Task system factory functions
//===----------------------------------------------------------------------===//

// Creates zero or more task executors from the current command line flags.
// This creates one executor per topology selected using the same executor
// parameters as specified by flags. |executors| is populated with retained
// executor instances and callers must reserve the |executors| memory before
// calling and release all executors using iree_task_executor_release when done
// with them.
//
// This utility method is useful when only a single type of executor exists
// within a process as the flags are globals. When multiple executors may exist
// or programmatic configuration is needed use the iree_task_executor_create
// method directly.
//
// Returns the total number of executors in |out_executor_count|.
// Returns IREE_STATUS_OUT_OF_RANGE if |executor_capacity| is insufficient and
// the caller needs to provide more storage in |executors|.
iree_status_t iree_task_executors_create_from_flags(
    iree_allocator_t host_allocator, iree_host_size_t executor_capacity,
    iree_task_executor_t** executors, iree_host_size_t* out_executor_count);

//===----------------------------------------------------------------------===//
// Task system simple invocation utilities
//===----------------------------------------------------------------------===//

// TODO(benvanik): simple IO completion event callback.
// TODO(benvanik): simple async function call dispatch.
// TODO(benvanik): simple parallel-for grid-style function call dispatch.

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_API_H_
