// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/api.h"

#include <stdbool.h>
#include <string.h>

#include "iree/base/internal/flags.h"
#include "iree/task/topology.h"

//===----------------------------------------------------------------------===//
// Executor configuration
//===----------------------------------------------------------------------===//

IREE_FLAG(
    int32_t, task_worker_spin_us, 0,
    "Maximum duration in microseconds each worker should spin waiting for\n"
    "additional work. In almost all cases this should be 0 as spinning is\n"
    "often extremely harmful to system health. Only set to non-zero values\n"
    "when latency is the #1 priority (vs. thermals, system-wide scheduling,\n"
    "etc).");

IREE_FLAG(
    int32_t, task_worker_stack_size, 128 * 1024,
    "Minimum size in bytes of each worker thread stack.\n"
    "The underlying platform may allocate more stack space but _should_\n"
    "guarantee that the available stack space is near this amount. Note that\n"
    "the task system will take some stack space and not all bytes should be\n"
    "assumed usable. Note that as much as possible users should not rely on\n"
    "the stack for storage over ~16-32KB and instead use local workgroup\n"
    "memory.");

IREE_FLAG(
    int32_t, task_worker_local_memory, 0,
    "Overrides the bytes of per-worker local memory allocated for use by\n"
    "dispatched tiles. Tiles may use less than this but will fail to dispatch\n"
    "if they require more. Conceptually it is like a stack reservation and\n"
    "should be treated the same way: the source programs must be built to\n"
    "only use a specific maximum amount of local memory and the runtime must\n"
    "be configured to make at least that amount of local memory available.\n"
    "By default the CPU L2 cache size is used if such queries are supported.");

iree_status_t iree_task_executor_options_initialize_from_flags(
    iree_task_executor_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  iree_task_executor_options_initialize(out_options);
  out_options->worker_spin_ns =
      (iree_duration_t)FLAG_task_worker_spin_us * 1000;
  out_options->worker_stack_size =
      (iree_host_size_t)FLAG_task_worker_stack_size;
  out_options->worker_local_memory_size =
      (iree_host_size_t)FLAG_task_worker_local_memory;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Topology configuration
//===----------------------------------------------------------------------===//

IREE_FLAG(
    string, task_topology_mode, "physical_cores",
    "Available modes:\n"
    " --task_topology_group_count=non-zero:\n"
    "   Uses whatever the specified group count is and ignores the set mode.\n"
    "   All threads will be unpinned and run on system-determined processors.\n"
    " --task_topology_cpu_ids=0,1,2 [+ --task_topology_cpu_ids=3,4,5]:\n"
    "   Creates one executor per set of logical CPU IDs.\n"
    " 'physical_cores':\n"
    "   Creates one executor per NUMA node in --task_topology_nodes= and one\n"
    "   group per physical core in each NUMA node up to the value specified\n"
    "   by --task_topology_max_group_count=.");

IREE_FLAG(
    int32_t, task_topology_group_count, 0,
    "Defines the total number of task system workers that will be created.\n"
    "Workers will be distributed across cores. Specifying 0 will use a\n"
    "heuristic defined by --task_topology_mode= to automatically select the\n"
    "worker count and distribution.\n"
    "WARNING: setting this flag directly is not recommended; use\n"
    "--task_topology_max_group_count= instead.");

IREE_FLAG_LIST(
    string, task_topology_cpu_ids,
    "A list of absolute logical CPU IDs to use for a single topology. One\n"
    "topology will be created for each repetition of the flag. CPU IDs match\n"
    "the Linux logical CPU ID scheme (as used by lscpu/lstopo) or a flattened\n"
    "[0, total_processor_count) range on Windows.");

IREE_FLAG(
    string, task_topology_nodes, "current",
    "Comma-separated list of NUMA nodes that topologies will be defined for.\n"
    "Each node specified will be configured based on the other topology\n"
    "flags. 'all' can be used to indicate all available NUMA nodes and\n"
    "'current' will inherit the node of the calling thread.");

IREE_FLAG(
    int32_t, task_topology_max_group_count, 8,
    "Sets a maximum value on the worker count that can be automatically\n"
    "detected and used when --task_topology_group_count=0 and is ignored\n"
    "otherwise.");

// Builds a bitmask of NUMA nodes that topologies should be created for.
//
// NOTE: because of the mask being 64-bits we have a 64-node limit.
// We could change this mask to be variable-sized (ala cpu_set) if we wanted to
// go higher than that. Since this entire set of functionality is part of the
// private implementation and not something core to the task system it's easy to
// change in the future if we get >4096-core machines.
static iree_status_t iree_task_topologies_select_nodes_from_flags(
    uint64_t* out_node_mask) {
  IREE_ASSERT_ARGUMENT(out_node_mask);
  *out_node_mask = 0ull;

  // Query the total number of NUMA nodes in the system. On implementations
  // where this information isn't available this will return 1.
  const iree_host_size_t available_node_count =
      iree_max(1u, iree_min(64u, iree_task_topology_query_node_count()));

  // Build a bitmask based on the flags.
  iree_string_view_t nodes_flag =
      iree_make_cstring_view(FLAG_task_topology_nodes);
  uint64_t node_mask = 0ull;
  if (iree_string_view_is_empty(nodes_flag) ||
      iree_string_view_equal(nodes_flag, IREE_SV("current"))) {
    // Use a single default node.
    node_mask = 1ull << iree_task_topology_query_current_node();
  } else if (iree_string_view_equal(nodes_flag, IREE_SV("all"))) {
    // Use all nodes in the system (set bits starting at 0 for each node).
    node_mask = UINT64_MAX >> (64 - available_node_count);
  } else {
    // Use some subset of nodes.
    iree_string_view_t remaining = nodes_flag;
    while (!iree_string_view_is_empty(remaining)) {
      iree_string_view_t node_value;
      iree_string_view_split(remaining, ',', &node_value, &remaining);
      uint32_t node_id = 0;
      if (!iree_string_view_atoi_uint32(node_value, &node_id)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid NUMA node ID specified: '%.*s'",
                                (int)node_value.size, node_value.data);
      } else if (node_id >= available_node_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "NUMA node ID out of valid range [0,%" PRIhsz
                                "): %u",
                                available_node_count, node_id);
      }
      node_mask |= 1ull << node_id;
    }
  }

  *out_node_mask = node_mask;
  return iree_ok_status();
}

iree_status_t iree_task_topology_initialize_from_flags(
    iree_task_topology_node_id_t node_id, iree_task_topology_t* out_topology) {
  IREE_ASSERT_ARGUMENT(out_topology);
  iree_task_topology_initialize(out_topology);

  if (FLAG_task_topology_group_count != 0) {
    // Unpinned topology. Let the system try to figure it out.
    iree_task_topology_initialize_from_group_count(
        FLAG_task_topology_group_count, out_topology);
    return iree_ok_status();
  } else if (strcmp(FLAG_task_topology_mode, "physical_cores") == 0) {
    // Physical cores sourced from a specific NUMA node.
    return iree_task_topology_initialize_from_physical_cores(
        node_id, FLAG_task_topology_max_group_count, out_topology);
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "one of --task_topology_group_count or --task_topology_mode must be "
        "specified and be a valid value; have --task_topology_mode=%s.",
        FLAG_task_topology_mode);
  }
}

//===----------------------------------------------------------------------===//
// Topology diagnostics
//===----------------------------------------------------------------------===//

static void iree_task_flags_print_action_flag(iree_string_view_t flag_name,
                                              void* storage, FILE* file) {
  fprintf(file, "# --%.*s\n", (int)flag_name.size, flag_name.data);
}

static void iree_task_flags_dump_task_topology(
    iree_host_size_t topology_id, const iree_task_topology_t* topology) {
  fprintf(stdout,
          "# "
          "===-------------------------------------------------------------"
          "-----------===\n");
  fprintf(stdout, "# topology[%" PRIhsz "]: %" PRIhsz " worker groups\n",
          topology_id, topology->group_count);
  fprintf(stdout,
          "# "
          "===-------------------------------------------------------------"
          "-----------===\n");
  fprintf(stdout, "#\n");
  for (iree_host_size_t j = 0; j < topology->group_count; ++j) {
    const iree_task_topology_group_t* group = &topology->groups[j];
    fprintf(stdout, "# group[%d]: '%s'\n", group->group_index, group->name);
    fprintf(stdout, "#      processor: %u\n", group->processor_index);
    fprintf(stdout, "#       affinity: ");
    if (group->ideal_thread_affinity.specified) {
      fprintf(
          stdout, "group=%u, id=%u, smt=%u", group->ideal_thread_affinity.group,
          group->ideal_thread_affinity.id, group->ideal_thread_affinity.smt);
    } else {
      fprintf(stdout, "(unspecified)");
    }
    fprintf(stdout, "\n");

    fprintf(stdout, "#  caches: l1d=%u, l2d=%u\n", group->caches.l1_data,
            group->caches.l2_data);

    fprintf(stdout, "#  last level cache sharing: ");
    if (group->constructive_sharing_mask == 0) {
      fprintf(stdout, "(none)\n");
    } else if (group->constructive_sharing_mask ==
               IREE_TASK_TOPOLOGY_GROUP_MASK_ALL) {
      fprintf(stdout, "(all/undefined)\n");
    } else {
      fprintf(stdout, "%d group(s): ",
              iree_math_count_ones_u64(group->constructive_sharing_mask));
      for (iree_host_size_t ic = 0, jc = 0;
           ic < IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT; ++ic) {
        if ((group->constructive_sharing_mask >> ic) & 1) {
          if (jc > 0) fprintf(stdout, ", ");
          fprintf(stdout, "%" PRIhsz, ic);
          ++jc;
        }
      }
      fprintf(stdout, "\n");
    }

    fprintf(stdout, "#\n");
  }
}

static iree_status_t iree_task_flags_dump_task_topologies(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  const iree_flag_string_list_t cpu_ids_list =
      FLAG_task_topology_cpu_ids_list();
  if (cpu_ids_list.count == 0) {
    // Select which nodes in the machine we will be creating topologies for.
    uint64_t node_mask = 0ull;
    IREE_RETURN_IF_ERROR(
        iree_task_topologies_select_nodes_from_flags(&node_mask));

    // TODO(benvanik): macros to make this iteration easier (ala cpu_set
    // iterators).
    iree_host_size_t topology_count = iree_math_count_ones_u64(node_mask);
    uint64_t node_mask_bits = node_mask;
    iree_task_topology_node_id_t node_base_id = 0;
    for (iree_host_size_t i = 0; i < topology_count; ++i) {
      int node_offset =
          iree_task_affinity_set_count_trailing_zeros(node_mask_bits);
      iree_task_topology_node_id_t node_id = node_base_id + node_offset;
      node_base_id += node_offset + 1;
      node_mask_bits = iree_shr(node_mask_bits, node_offset + 1);
      iree_task_topology_t topology;
      IREE_RETURN_IF_ERROR(
          iree_task_topology_initialize_from_flags(node_id, &topology));
      iree_task_flags_dump_task_topology(i, &topology);
      iree_task_topology_deinitialize(&topology);
    }
  } else {
    for (iree_host_size_t i = 0; i < cpu_ids_list.count; ++i) {
      iree_task_topology_t topology;
      IREE_RETURN_IF_ERROR(
          iree_task_topology_initialize_from_logical_cpu_set_string(
              cpu_ids_list.values[i], &topology));
      iree_task_flags_dump_task_topology(i, &topology);
      iree_task_topology_deinitialize(&topology);
    }
  }

  exit(0);
  return iree_ok_status();
}

IREE_FLAG_CALLBACK(
    iree_task_flags_dump_task_topologies, iree_task_flags_print_action_flag,
    NULL, dump_task_topologies,
    "Dumps the flag-specified topology used for creating task executors.");

//===----------------------------------------------------------------------===//
// Task system factory functions
//===----------------------------------------------------------------------===//

iree_status_t iree_task_executors_create_from_flags(
    iree_allocator_t host_allocator, iree_host_size_t executor_capacity,
    iree_task_executor_t** executors, iree_host_size_t* out_executor_count) {
  IREE_ASSERT_ARGUMENT(out_executor_count);
  *out_executor_count = 0;
  if (executors) {
    memset(executors, 0, executor_capacity * sizeof(*executors));
  }

  // Each executor will have the same options based on the global flags.
  // A user constructing their own executors can differ the options.
  iree_task_executor_options_t options;
  IREE_RETURN_IF_ERROR(
      iree_task_executor_options_initialize_from_flags(&options));

  // Select which nodes in the machine we will be creating topologies for based
  // on the topology mode.
  iree_host_size_t topology_count = 0;
  uint64_t node_mask = 0ull;
  const iree_flag_string_list_t cpu_ids_list =
      FLAG_task_topology_cpu_ids_list();
  if (cpu_ids_list.count == 0) {
    IREE_RETURN_IF_ERROR(
        iree_task_topologies_select_nodes_from_flags(&node_mask));
    topology_count = iree_math_count_ones_u64(node_mask);
  } else {
    topology_count = cpu_ids_list.count;
  }

  // Since this utility function creates one executor per topology returned by
  // the query we can check the executor capacity immediately.
  if (topology_count > executor_capacity || !executors) {
    // Need more capacity.
    *out_executor_count = topology_count;
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  } else if (topology_count == 0) {
    // No executors required, early-exit.
    *out_executor_count = 0;
    return iree_ok_status();
  }

  // NOTE: the flags could use some ergonomics improvement or renaming to
  // indicate how they differ. Trying to specify a generic group count _and_
  // multiple NUMA nodes won't produce expected results (IMO) so we error out
  // on that here instead of letting users think they are running with
  // NUMA-aware scheduling. We could lighten this restriction in the future if
  // there are use cases for arbitrarily-scheduled worker groups that have just
  // their allocations pinned to NUMA nodes.
  if (FLAG_task_topology_group_count != 0 && topology_count > 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "multiple nodes specified with --task_topology_group_count=; you "
        "probably meant --task_topology_max_group_count= in order to get "
        "proper NUMA-aware scheduling");
  }

  // Create one executor per topology.
  iree_status_t status = iree_ok_status();
  if (cpu_ids_list.count == 0) {
    // TODO(benvanik): macros to make this iteration easier (ala cpu_set
    // iterators).
    uint64_t node_mask_bits = node_mask;
    iree_task_topology_node_id_t node_base_id = 0;
    for (iree_host_size_t i = 0; i < topology_count; ++i) {
      int node_offset =
          iree_task_affinity_set_count_trailing_zeros(node_mask_bits);
      iree_task_topology_node_id_t node_id = node_base_id + node_offset;
      node_base_id += node_offset + 1;
      node_mask_bits = iree_shr(node_mask_bits, node_offset + 1);

      // Query topology for the node this executor is pinned to.
      iree_task_topology_t topology;
      status = iree_task_topology_initialize_from_flags(node_id, &topology);
      if (!iree_status_is_ok(status)) break;

      // TODO(benvanik): if group count is 0 then don't create the executor.
      // Today the executor creation will fail with 0 groups so the program
      // won't get in a weird state but it's probably not what a user would
      // expect.

      // Create executor with the given topology.
      status = iree_task_executor_create(options, &topology, host_allocator,
                                         &executors[i]);

      // Executor has consumed the topology and it can be dropped now.
      iree_task_topology_deinitialize(&topology);
      if (!iree_status_is_ok(status)) break;
    }
  } else {
    for (iree_host_size_t i = 0; i < topology_count; ++i) {
      // Query topology for the node this executor is pinned to.
      iree_task_topology_t topology;
      status = iree_task_topology_initialize_from_logical_cpu_set_string(
          cpu_ids_list.values[i], &topology);
      if (!iree_status_is_ok(status)) break;

      // TODO(benvanik): if group count is 0 then don't create the executor.
      // Today the executor creation will fail with 0 groups so the program
      // won't get in a weird state but it's probably not what a user would
      // expect.

      // Create executor with the given topology.
      status = iree_task_executor_create(options, &topology, host_allocator,
                                         &executors[i]);

      // Executor has consumed the topology and it can be dropped now.
      iree_task_topology_deinitialize(&topology);
      if (!iree_status_is_ok(status)) break;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executor_count = topology_count;
  } else {
    // Release executors for the caller in case we partially initialized them.
    for (iree_host_size_t i = 0; i < topology_count; ++i) {
      iree_task_executor_release(executors[i]);
    }
  }
  return status;
}
