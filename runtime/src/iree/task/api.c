// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/api.h"

#include <stdbool.h>
#include <string.h>

#include "iree/base/tooling/flags.h"
#include "iree/task/affinity_set.h"
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
    "Specifies which cores become task executors, and how they are grouped. "
    "One\n"
    "executor is created per NUMA node named below, each with its own workers\n"
    "and queue:\n"
    "  'current'  - the calling thread's NUMA node only (default).\n"
    "  'numa'     - one executor per NUMA node.\n"
    "  'all'      - every core in a single executor, not split per node.\n"
    "  '0,2,...'  - explicit NUMA node ids; one executor per listed id.\n"
    "Node ids come from /sys/devices/system/node on Linux and may be sparse.\n"
    "Each worker takes the affinity of its own core; an executor whose cores\n"
    "span several nodes has no single node to allocate from.");

IREE_FLAG(
    int32_t, task_topology_max_group_count, 64,
    "Sets a maximum value on the worker count that can be automatically\n"
    "detected and used when --task_topology_group_count=0 and is ignored\n"
    "otherwise.");

IREE_FLAG(string, task_topology_snapshot, "",
          "Reads CPU topology from a captured snapshot instead of the host.\n"
          "For testing only.");

static iree_status_t iree_task_topology_apply_snapshot_flag(void) {
  const char* path = FLAG_task_topology_snapshot;
  if (!path || !*path) return iree_ok_status();
  return iree_task_topology_set_snapshot_path(path);
}

IREE_FLAG(string, task_topology_performance_level, "any",
          "Selects only cores that match the specified performance level from\n"
          "[`any`, `low` (or `efficiency`), `high` (or `performance`)].");

IREE_FLAG(
    string, task_topology_distribution, "scatter",
    "Strategy for distributing cores across cache domains (CCXs) within the\n"
    "selected NUMA node(s) (use --task_topology_nodes to control NUMA "
    "locality):\n"
    "  `compact` - Fill cache domains sequentially (better for compute).\n"
    "  `scatter` - Round-robin across domains (better for memory bandwidth).");

IREE_FLAG(
    string, task_topology_favor, "",
    "High-level preset for common deployment scenarios (overrides "
    "distribution and performance_level if specified):\n"
    "  `latency`    - Minimize single-request latency (compact + high-perf).\n"
    "  `throughput` - Maximize batch throughput (scatter + any-perf).\n"
    "  `efficiency` - Minimize power consumption (compact + low-perf).");

iree_status_t iree_task_topology_select_nodes(
    iree_string_view_t spec, iree_task_node_selection_t* out_selection) {
  IREE_ASSERT_ARGUMENT(out_selection);
  memset(out_selection, 0, sizeof(*out_selection));

  iree_string_view_t nodes_flag = spec;
  if (iree_string_view_is_empty(nodes_flag)) {
    nodes_flag = IREE_SV("current");
  }

  // 'current' (default): a single executor scoped to the calling thread's NUMA
  // node; does not spread across NUMA nodes.
  if (iree_string_view_equal(nodes_flag, IREE_SV("current"))) {
    out_selection->count = 1;
    out_selection->ids[0] = iree_task_topology_query_current_node();
    return iree_ok_status();
  }

  // 'all': a SINGLE executor spanning every core (node id ANY, no per-node
  // partitioning) so all workers get work regardless of node layout.
  if (iree_string_view_equal(nodes_flag, IREE_SV("all"))) {
    out_selection->count = 1;
    out_selection->ids[0] = IREE_TASK_TOPOLOGY_NODE_ID_ANY;
    return iree_ok_status();
  }

  // Explicit ids are validated against the enumerated node ids rather than the
  // node count, which would reject valid sparse ids.
  iree_task_topology_node_id_t available_ids[IREE_TASK_TOPOLOGY_MAX_NODES];
  const iree_host_size_t available_count = iree_task_topology_query_node_ids(
      IREE_TASK_TOPOLOGY_MAX_NODES, available_ids);
  if (available_count > IREE_TASK_TOPOLOGY_MAX_NODES) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "machine has %" PRIhsz
        " NUMA nodes but a selection can name at most %d; use explicit node "
        "ids to choose a subset",
        available_count, IREE_TASK_TOPOLOGY_MAX_NODES);
  }

  // 'numa': one executor per NUMA node across the whole machine.
  if (iree_string_view_equal(nodes_flag, IREE_SV("numa"))) {
    memcpy(out_selection->ids, available_ids,
           available_count * sizeof(available_ids[0]));
    out_selection->count = available_count;
    return iree_ok_status();
  }

  // Explicit list of node ids -> one executor per id.
  iree_string_view_t remaining = nodes_flag;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t node_value;
    iree_string_view_split(remaining, ',', &node_value, &remaining);
    uint32_t node_id = 0;
    if (!iree_string_view_atoi_uint32(node_value, &node_id)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid --task_topology_nodes value: '%.*s'",
                              (int)node_value.size, node_value.data);
    }
    bool node_available = false;
    for (iree_host_size_t i = 0; i < available_count; ++i) {
      if (available_ids[i] == node_id) {
        node_available = true;
        break;
      }
    }
    if (!node_available) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "NUMA node id %u is not present on this machine; "
                              "%" PRIhsz " node(s) available",
                              node_id, available_count);
    }
    if (out_selection->count >= IREE_TASK_TOPOLOGY_MAX_NODES) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "too many nodes specified (max %d)",
                              IREE_TASK_TOPOLOGY_MAX_NODES);
    }
    out_selection->ids[out_selection->count++] = node_id;
  }
  return iree_ok_status();
}

// Resolves --task_topology_nodes into a node selection. See the flag help for
// the grammar.
static iree_status_t iree_task_topologies_select_nodes_from_flags(
    iree_task_node_selection_t* out_selection) {
  IREE_RETURN_IF_ERROR(iree_task_topology_apply_snapshot_flag());
  return iree_task_topology_select_nodes(
      iree_make_cstring_view(FLAG_task_topology_nodes), out_selection);
}

static iree_status_t iree_task_topology_parse_performance_level(
    const char* value, iree_task_topology_performance_level_t* out_level) {
  *out_level = IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY;
  if (strcmp(value, "any") == 0) {
    *out_level = IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY;
    return iree_ok_status();
  } else if (strcmp(value, "low") == 0 || strcmp(value, "efficiency") == 0) {
    *out_level = IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_LOW;
    return iree_ok_status();
  } else if (strcmp(value, "high") == 0 || strcmp(value, "performance") == 0) {
    *out_level = IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_HIGH;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown value `%s` for performance level; expected "
                          "one of [any, low/efficiency, high/performance]",
                          value);
}

static iree_status_t iree_task_topology_parse_distribution(
    const char* value, iree_task_topology_distribution_t* out_distribution) {
  *out_distribution = IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER;
  if (strcmp(value, "compact") == 0) {
    *out_distribution = IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT;
    return iree_ok_status();
  } else if (strcmp(value, "scatter") == 0) {
    *out_distribution = IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown value `%s` for distribution strategy; "
                          "expected one of [compact, scatter]",
                          value);
}

// Parses --task_topology_favor presets into distribution + performance_level.
// Returns true if a preset was specified, false if flag was empty.
static bool iree_task_topology_parse_favor_preset(
    const char* value, iree_task_topology_performance_level_t* out_perf_level,
    iree_task_topology_distribution_t* out_distribution) {
  if (!value || strcmp(value, "") == 0) {
    return false;  // No preset specified.
  }

  if (strcmp(value, "latency") == 0) {
    // Minimize single-request latency: compact + high-performance cores.
    *out_perf_level = IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_HIGH;
    *out_distribution = IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT;
    return true;
  } else if (strcmp(value, "throughput") == 0) {
    // Maximize batch throughput: scatter + all available cores.
    *out_perf_level = IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY;
    *out_distribution = IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER;
    return true;
  } else if (strcmp(value, "efficiency") == 0) {
    // Minimize power: compact + low-power cores.
    *out_perf_level = IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_LOW;
    *out_distribution = IREE_TASK_TOPOLOGY_DISTRIBUTION_COMPACT;
    return true;
  }

  // Unknown preset - caller will handle error.
  return false;
}

iree_status_t iree_task_topology_initialize_from_flags(
    iree_task_topology_node_id_t node_id, iree_task_topology_t* out_topology) {
  IREE_ASSERT_ARGUMENT(out_topology);
  iree_task_topology_initialize(out_topology);
  IREE_RETURN_IF_ERROR(iree_task_topology_apply_snapshot_flag());

  if (FLAG_task_topology_group_count != 0) {
    // Unpinned topology. Let the system try to figure it out.
    iree_task_topology_initialize_from_group_count(
        FLAG_task_topology_group_count, out_topology);
    return iree_ok_status();
  } else if (strcmp(FLAG_task_topology_mode, "physical_cores") == 0) {
    // Physical cores sourced from a specific NUMA node.
    iree_task_topology_performance_level_t performance_level =
        IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY;
    iree_task_topology_distribution_t distribution =
        IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER;

    // Check if --task_topology_favor preset overrides individual flags.
    if (iree_task_topology_parse_favor_preset(
            FLAG_task_topology_favor, &performance_level, &distribution)) {
      // Preset specified - use its values.
    } else {
      // No preset - parse individual flags.
      IREE_RETURN_IF_ERROR(iree_task_topology_parse_performance_level(
          FLAG_task_topology_performance_level, &performance_level));
      IREE_RETURN_IF_ERROR(iree_task_topology_parse_distribution(
          FLAG_task_topology_distribution, &distribution));
    }

    return iree_task_topology_initialize_from_physical_cores(
        node_id, performance_level, distribution,
        FLAG_task_topology_max_group_count, out_topology);
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

// Prints the active backend's raw hardware ids for |processor|, or nothing when
// the backend exposes none.
static void iree_task_flags_dump_processor_debug_ids(uint32_t processor) {
  char ids[128];
  if (iree_task_topology_format_processor_debug_ids(processor, sizeof(ids),
                                                    ids) == 0) {
    return;
  }
  fprintf(stdout, "#     hardware ids: %s\n", ids);
}

static void iree_task_flags_dump_task_topology(
    iree_host_size_t topology_id, const iree_task_topology_t* topology) {
  fprintf(stdout,
          "# "
          "===-------------------------------------------------------------"
          "-----------===\n");
  char numa_node[16];
  if (topology->numa_node_id == IREE_TASK_TOPOLOGY_NODE_ID_ANY) {
    iree_snprintf(numa_node, sizeof(numa_node), "any");
  } else {
    iree_snprintf(numa_node, sizeof(numa_node), "%u", topology->numa_node_id);
  }
  fprintf(stdout,
          "# topology[%" PRIhsz "]: %" PRIhsz " worker groups, numa node: %s\n",
          topology_id, topology->group_count, numa_node);
  fprintf(stdout,
          "# "
          "===-------------------------------------------------------------"
          "-----------===\n");
  fprintf(stdout, "#\n");
  for (iree_host_size_t j = 0; j < topology->group_count; ++j) {
    const iree_task_topology_group_t* group = &topology->groups[j];
    fprintf(stdout, "# group[%d]: '%s'\n", group->group_index, group->name);
    fprintf(stdout, "#      processor: %u\n", group->processor_index);
    iree_task_flags_dump_processor_debug_ids(group->processor_index);
    fprintf(stdout, "#       affinity: ");
    if (group->ideal_thread_affinity.group_any) {
      fprintf(stdout, "group=%u (any)", group->ideal_thread_affinity.group);
    } else if (group->ideal_thread_affinity.id_assigned) {
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
    if (iree_task_affinity_set_is_empty(group->constructive_sharing_mask)) {
      fprintf(stdout, "(none)\n");
    } else if (iree_task_affinity_set_equal(
                   group->constructive_sharing_mask,
                   iree_task_affinity_for_any_worker())) {
      fprintf(stdout, "(all/undefined)\n");
    } else {
      fprintf(
          stdout, "%" PRIhsz " group(s): ",
          iree_task_affinity_set_count_ones(group->constructive_sharing_mask));
      for (iree_host_size_t ic = 0, jc = 0;
           ic < IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT; ++ic) {
        iree_task_affinity_bit_t bit = iree_task_affinity_bit_for_worker(ic);
        if (iree_task_affinity_set_test(group->constructive_sharing_mask,
                                        bit)) {
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
    iree_task_node_selection_t selection;
    IREE_RETURN_IF_ERROR(
        iree_task_topologies_select_nodes_from_flags(&selection));

    iree_host_size_t topology_index = 0;
    for (iree_host_size_t i = 0; i < selection.count; ++i) {
      iree_task_topology_t topology;
      IREE_RETURN_IF_ERROR(iree_task_topology_initialize_from_flags(
          selection.ids[i], &topology));
      // Skip topologies with no worker groups.
      if (topology.group_count > 0) {
        iree_task_flags_dump_task_topology(topology_index++, &topology);
      }
      iree_task_topology_deinitialize(&topology);
    }
  } else {
    iree_host_size_t topology_index = 0;
    for (iree_host_size_t i = 0; i < cpu_ids_list.count; ++i) {
      iree_task_topology_t topology;
      IREE_RETURN_IF_ERROR(
          iree_task_topology_initialize_from_logical_cpu_set_string(
              cpu_ids_list.values[i], &topology));
      // Skip topologies with no worker groups.
      if (topology.group_count > 0) {
        iree_task_flags_dump_task_topology(topology_index++, &topology);
      }
      iree_task_topology_deinitialize(&topology);
    }
  }

  exit(0);
  return iree_ok_status();
}

IREE_FLAG_CALLBACK(iree_task_flags_dump_task_topologies,
                   iree_task_flags_print_action_flag, NULL,
                   dump_task_topologies, "Prints the task executor topology.");

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
  iree_task_node_selection_t selection;
  memset(&selection, 0, sizeof(selection));
  const iree_flag_string_list_t cpu_ids_list =
      FLAG_task_topology_cpu_ids_list();
  if (cpu_ids_list.count == 0) {
    IREE_RETURN_IF_ERROR(
        iree_task_topologies_select_nodes_from_flags(&selection));
    topology_count = selection.count;
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
  iree_host_size_t executor_index = 0;
  if (cpu_ids_list.count == 0) {
    for (iree_host_size_t i = 0; i < topology_count; ++i) {
      // Query topology for the node this executor is pinned to.
      iree_task_topology_t topology;
      status =
          iree_task_topology_initialize_from_flags(selection.ids[i], &topology);
      if (!iree_status_is_ok(status)) break;

      // Skip topologies with no worker groups.
      if (topology.group_count > 0) {
        // Create executor with the given topology.
        status = iree_task_executor_create(options, &topology, host_allocator,
                                           &executors[executor_index++]);
      }

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

      // Skip topologies with no worker groups.
      if (topology.group_count > 0) {
        // Create executor with the given topology.
        status = iree_task_executor_create(options, &topology, host_allocator,
                                           &executors[executor_index++]);
      }

      // Executor has consumed the topology and it can be dropped now.
      iree_task_topology_deinitialize(&topology);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // A selection that resolves to no usable cores would otherwise hand callers
  // an executor-less driver (queue_count=0) and fail far from the cause.
  if (iree_status_is_ok(status) && executor_index == 0) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "--task_topology_nodes=%s selected %" PRIhsz
        " node(s) but none contain any usable cores; no executors created",
        FLAG_task_topology_nodes, topology_count);
  }

  if (iree_status_is_ok(status)) {
    *out_executor_count = executor_index;
  } else {
    // Release executors for the caller in case we partially initialized them.
    for (iree_host_size_t i = 0; i < executor_index; ++i) {
      iree_task_executor_release(executors[i]);
    }
  }
  return status;
}
