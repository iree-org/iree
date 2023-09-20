// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/task/topology.h"

#if !defined(IREE_PLATFORM_WINDOWS)

// Initializes |out_topology| with a standardized behavior when cpuinfo is not
// available (unsupported arch, failed to query, etc).
static void iree_task_topology_initialize_fallback(
    iree_host_size_t max_group_count, iree_task_topology_t* out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, max_group_count);

  // TODO(benvanik): implement our own query... but that seems not so great.
  // For now we default to a single group: if a user wants more then they can
  // either get cpuinfo working for their platform or manually construct the
  // topology themselves.
  iree_host_size_t group_count = 1;
  iree_task_topology_initialize_from_group_count(group_count, out_topology);

  IREE_TRACE_ZONE_END(z0);
}

#if defined(IREE_TASK_CPUINFO_DISABLED)

iree_host_size_t iree_task_topology_query_node_count(void) { return 1; }

iree_task_topology_node_id_t iree_task_topology_query_current_node(void) {
  return 0;
}

iree_status_t iree_task_topology_fixup_constructive_sharing_masks(
    iree_task_topology_t* topology) {
  // No-op.
  return iree_ok_status();
}

iree_status_t iree_task_topology_initialize_from_logical_cpu_set(
    iree_host_size_t cpu_count, const uint32_t* cpu_ids,
    iree_task_topology_t* out_topology) {
  // Today we have a fixed limit on the number of groups within a particular
  // topology.
  if (cpu_count >= IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many CPUs specified (%" PRIhsz
                            " provided for a max capacity of %zu)",
                            cpu_count, IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, cpu_count);

  iree_task_topology_initialize(out_topology);

  out_topology->group_count = cpu_count;
  for (iree_host_size_t i = 0; i < cpu_count; ++i) {
    iree_task_topology_group_t* group = &out_topology->groups[i];
    iree_task_topology_group_initialize(i, group);
    group->processor_index = cpu_ids[i];

    // NOTE: without cpuinfo we can't get SMT and node info but this isn't
    // really used on Linux today anyway.
    iree_thread_affinity_t* affinity = &group->ideal_thread_affinity;
    memset(affinity, 0, sizeof(*affinity));
    affinity->specified = 1;
    affinity->id = cpu_ids[i];
  }

  iree_status_t status =
      iree_task_topology_fixup_constructive_sharing_masks(out_topology);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_task_topology_initialize_from_physical_cores(
    iree_task_topology_node_id_t node_id, iree_host_size_t max_core_count,
    iree_task_topology_t* out_topology) {
  iree_task_topology_initialize_fallback(max_core_count, out_topology);
  return iree_ok_status();
}

#else

#include <cpuinfo.h>

static bool iree_task_topology_is_cpuinfo_available() {
  return cpuinfo_initialize() && cpuinfo_get_cores_count() > 0;
}

// TODO(benvanik): change to a system API and move to iree/base/allocator.h so
// it can be used there for binding memory to nodes.
iree_host_size_t iree_task_topology_query_node_count(void) {
  if (!iree_task_topology_is_cpuinfo_available()) return 1;
  // NOTE: this may span across packages!
  return cpuinfo_get_clusters_count();
}

// Returns the core of the calling thread or NULL if not supported.
// We wrap this here because cpuinfo only returns non-NULL on linux.
static const struct cpuinfo_core* iree_task_topology_get_current_core() {
  const struct cpuinfo_core* current_core = cpuinfo_get_current_core();
#if defined(IREE_PLATFORM_WINDOWS)
  // TODO(benvanik): drop cpuinfo.
  if (current_core == NULL) {
    PROCESSOR_NUMBER processor_number;
    GetCurrentProcessorNumberEx(&processor_number);
    uint32_t processor_id =
        cpuinfo_get_package(processor_number.Group)->processor_start +
        processor_number.Number;
    current_core = cpuinfo_get_processor(processor_id)->core;
  }
#endif  // IREE_PLATFORM_WINDOWS
  return current_core;
}

iree_task_topology_node_id_t iree_task_topology_query_current_node(void) {
  if (!iree_task_topology_is_cpuinfo_available()) return 0;
  const struct cpuinfo_core* current_core =
      iree_task_topology_get_current_core();
  return current_core ? current_core->cluster->cluster_id : 0;
}

// Returns |core_id| rotated by the calling base core ID.
// On many systems the kernel will have already assigned a randomized starting
// core for thread distribution and we can just reuse that.
static uint32_t iree_task_topology_rotate_from_base_core(uint32_t core_id) {
  const struct cpuinfo_core* current_core =
      iree_task_topology_get_current_core();
  if (!current_core) {
    return core_id;  // don't modify if we don't know
  }
  uint32_t next_core_id =
      (current_core->core_id + 1) % cpuinfo_get_cores_count();
  return (next_core_id + core_id) % cpuinfo_get_cores_count();
}

// Sets a platform-specific iree_thread_affinity_t based on the cpuinfo
// processor.
static void iree_task_topology_set_affinity_from_processor(
    const struct cpuinfo_processor* processor,
    iree_thread_affinity_t* out_affinity) {
  memset(out_affinity, 0, sizeof(*out_affinity));
  out_affinity->specified = 1;

  // Special bit to indicate that (if required) we want the entire core.
  if (processor->core->processor_count > 1) {
    out_affinity->smt = 1;
  }

  // cpuinfo #ifdefs the fields we need to extract the right platform IDs.
  // We purposefully use the same exact macros they do there so that we don't
  // have to worry about skew.

#if defined(__MACH__) && defined(__APPLE__)
  // TODO(benvanik): run on darwin to see how the l2 caches map. We ideally want
  // a unique affinity ID per L2 cache.
  // For now, we just use some random pointer bytes. It's just a tag used by
  // the kernel to distribute the threads so the exact bits don't matter as long
  // as they are unique per group we want isolated.
  out_affinity->group = processor->cluster->cluster_id;
  out_affinity->id = (uint32_t)(uintptr_t)processor;
#elif defined(__linux__)
  out_affinity->group = processor->cluster->cluster_id;
  out_affinity->id = processor->linux_id;
#elif defined(_WIN32) || defined(__CYGWIN__)
  out_affinity->group = processor->windows_group_id;
  out_affinity->id = processor->windows_processor_id;
#else
  // WASM? Unusued today.
  out_affinity->specified = 0;
#endif  // cpuinfo-like platform field
}

// Populates |out_group| with the information from |processor|.
static void iree_task_topology_group_initialize_from_processor(
    uint32_t group_index, const struct cpuinfo_processor* processor,
    iree_task_topology_group_t* out_group) {
  iree_task_topology_group_initialize(group_index, out_group);
#if defined(__linux__)
  out_group->processor_index = processor->linux_id;
#else
  out_group->processor_index =
      processor->core->processor_start + processor->smt_id;
#endif  // __linux__
  iree_task_topology_set_affinity_from_processor(
      processor, &out_group->ideal_thread_affinity);
}

// Populates |out_group| with the information from |core|.
static void iree_task_topology_group_initialize_from_core(
    uint32_t group_index, const struct cpuinfo_core* core,
    iree_task_topology_group_t* out_group) {
  // Guess: always pick the first processor in a core.
  // When pinning to threads we'll take into account whether the core is SMT
  // and use all threads anyway so this alignment is just helpful for debugging.
  uint32_t processor_i = core->processor_start;
  const struct cpuinfo_processor* processor =
      cpuinfo_get_processor(processor_i);
  iree_task_topology_group_initialize_from_processor(group_index, processor,
                                                     out_group);
}

// Returns a bitset with all *processors* that share the same |cache|.
static uint64_t iree_task_topology_calculate_cache_bits(
    const struct cpuinfo_cache* cache) {
  if (!cache) return 0;
  uint64_t mask = 0;
  for (uint32_t processor_i = 0; processor_i < cache->processor_count;
       ++processor_i) {
    uint32_t i = cache->processor_start + processor_i;
    if (i < IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT) {
      mask |= 1ull << i;
    }
  }
  return mask;
}

// Constructs a constructive sharing mask for all *processors* that share the
// same cache as the specified |processor|.
static uint64_t iree_task_topology_calculate_constructive_sharing_mask(
    const struct cpuinfo_processor* processor) {
  uint64_t mask = 0;
  mask |= iree_task_topology_calculate_cache_bits(processor->cache.l1i);
  mask |= iree_task_topology_calculate_cache_bits(processor->cache.l1d);
  mask |= iree_task_topology_calculate_cache_bits(processor->cache.l2);
  // TODO(benvanik): include L3 here too (for systems that have it)? Or use L3
  // info purely for distribution and focus the group mask on lower-latency
  // caches?
  return mask;
}

iree_status_t iree_task_topology_fixup_constructive_sharing_masks(
    iree_task_topology_t* topology) {
  if (!iree_task_topology_is_cpuinfo_available()) {
    // No-op when cpuinfo is unavailable.
    return iree_ok_status();
  }

  // O(n^2), but n is always <= 64 (and often <= 8).
  for (iree_host_size_t i = 0; i < topology->group_count; ++i) {
    iree_task_topology_group_t* group = &topology->groups[i];

    // Compute the processors that we can constructively share with.
    uint64_t constructive_sharing_mask =
        iree_task_topology_calculate_constructive_sharing_mask(
            cpuinfo_get_processor(group->processor_index));

    iree_task_topology_group_mask_t group_mask = 0;
    for (iree_host_size_t j = 0; j < topology->group_count; ++j) {
      const iree_task_topology_group_t* other_group = &topology->groups[j];
      uint64_t group_processor_bits =
          iree_math_rotl_u64(1ull, other_group->processor_index);
      if (constructive_sharing_mask & group_processor_bits) {
        group_mask |= iree_math_rotl_u64(1ull, other_group->group_index);
      }
    }

    group->constructive_sharing_mask = group_mask;
  }

  return iree_ok_status();
}

iree_status_t iree_task_topology_initialize_from_logical_cpu_set(
    iree_host_size_t cpu_count, const uint32_t* cpu_ids,
    iree_task_topology_t* out_topology) {
  // Ensure cpuinfo is available; if not we fall back to random.
  if (!iree_task_topology_is_cpuinfo_available()) {
    iree_task_topology_initialize_fallback(cpu_count, out_topology);
    return iree_ok_status();
  }

  // Today we have a fixed limit on the number of groups within a particular
  // topology.
  if (cpu_count >= IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many CPUs specified (%" PRIhsz
                            " provided for a max capacity of %zu)",
                            cpu_count, IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT);
  }

  // Validate the CPU IDs provided.
  const uint32_t processor_count = cpuinfo_get_processors_count();
  for (iree_host_size_t i = 0; i < cpu_count; ++i) {
    if (cpu_ids[i] >= processor_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "cpu_ids[%" PRIhsz
          "] %u out of bounds, only %u logical processors available",
          i, cpu_ids[i], processor_count);
    }
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, cpu_count);

  iree_task_topology_initialize(out_topology);

  out_topology->group_count = cpu_count;
  for (iree_host_size_t i = 0; i < cpu_count; ++i) {
    const struct cpuinfo_processor* processor =
        cpuinfo_get_processor(cpu_ids[i]);
    iree_task_topology_group_initialize_from_processor(
        i, processor, &out_topology->groups[i]);
  }

  iree_task_topology_fixup_constructive_sharing_masks(out_topology);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Returns true if the given |core| passes the filter and should be included.
// |user_data| is the value passed alongside the filter function.
typedef bool (*iree_task_topology_core_filter_t)(
    const struct cpuinfo_core* core, uintptr_t user_data);

// Matches all cores.
static bool iree_task_topology_core_filter_all(const struct cpuinfo_core* core,
                                               uintptr_t user_data) {
  return true;
}

// Matches all cores that have the provided cluster ID.
static bool iree_task_topology_core_filter_by_cluster_id(
    const struct cpuinfo_core* core, uintptr_t user_data) {
  uint32_t cluster_id = (uint32_t)user_data;
  if (cluster_id == IREE_TASK_TOPOLOGY_NODE_ID_ANY) return true;
  return core->cluster->cluster_id == cluster_id;
}

// Initializes a topology with one group for each core that matches |filter_fn|.
//
// If cpuinfo is not available this falls back to the same behavior as
// iree_task_topology_initialize_from_physical_cores.
static void iree_task_topology_initialize_from_physical_cores_with_filter(
    iree_task_topology_core_filter_t filter_fn, uintptr_t filter_fn_data,
    iree_host_size_t max_core_count, iree_task_topology_t* out_topology) {
  if (!iree_task_topology_is_cpuinfo_available()) {
    iree_task_topology_initialize_fallback(max_core_count, out_topology);
    return;
  }

  max_core_count = iree_min(max_core_count, IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, max_core_count);

  // Count cores that match the filter.
  iree_host_size_t core_count = 0;
  for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
    const struct cpuinfo_core* core = cpuinfo_get_core(i);
    if (filter_fn(core, filter_fn_data)) ++core_count;
  }
  core_count = iree_min(core_count, max_core_count);

  iree_task_topology_initialize(out_topology);

  // Build each core up to the max allowed.
  // TODO(benvanik): if our group_count <= core_count/2 then distribute better;
  // for now we just do a straight-line through (cores 0-N) when instead we may
  // want to take advantage of L3 cache info (half of groups on one L3 cache,
  // half of groups on another, etc).
  out_topology->group_count = core_count;
  for (uint32_t core_i = 0, group_i = 0; group_i < out_topology->group_count;
       ++core_i) {
    // Rotate the core ID so that we avoid setting the affinity to the calling
    // thread which we assume is something the user has plans for and doesn't
    // want to have our workers stealing their time.
    const struct cpuinfo_core* core =
        cpuinfo_get_core(iree_task_topology_rotate_from_base_core(core_i));
    if (filter_fn(core, filter_fn_data)) {
      iree_task_topology_group_initialize_from_core(
          group_i, core, &out_topology->groups[group_i]);
      ++group_i;
    }
  }

  iree_task_topology_fixup_constructive_sharing_masks(out_topology);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_task_topology_initialize_from_physical_cores(
    iree_task_topology_node_id_t node_id, iree_host_size_t max_core_count,
    iree_task_topology_t* out_topology) {
  iree_task_topology_initialize_from_physical_cores_with_filter(
      iree_task_topology_core_filter_by_cluster_id, (uintptr_t)node_id,
      max_core_count, out_topology);
  return iree_ok_status();
}

#endif  // IREE_TASK_CPUINFO_DISABLED

#endif  // !IREE_PLATFORM_WINDOWS
