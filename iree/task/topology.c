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

#include "iree/task/topology.h"

#include <cpuinfo.h>
#include <stdio.h>

#include "iree/base/math.h"
#include "iree/base/tracing.h"

struct iree_task_topology_s {
  iree_allocator_t allocator;
  iree_host_size_t group_capacity;
  iree_host_size_t group_count;
  iree_task_topology_group_t groups[0];
};

iree_status_t iree_task_topology_allocate(iree_host_size_t group_capacity,
                                          iree_allocator_t allocator,
                                          iree_task_topology_t** out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t topology_size =
      sizeof(iree_task_topology_t) +
      group_capacity * sizeof(iree_task_topology_group_t);

  iree_task_topology_t* topology = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, topology_size, (void**)&topology));
  topology->allocator = allocator;
  topology->group_capacity = group_capacity;
  topology->group_count = 0;

  *out_topology = topology;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_task_topology_free(iree_task_topology_t* topology) {
  if (!topology) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(topology->allocator, topology);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_task_topology_parse(iree_string_view_t value,
                                       iree_allocator_t allocator,
                                       iree_task_topology_t** out_topology) {
  // TODO(benvanik): define a format that is generally useful alongside cpuinfo.
  // Maybe colon-separated group-id values from thread affinities? Like:
  //   0.0:0.2:0.4:0.8 to indicate cores 0,2,4,8 on group 0
  //   0.0:0.1:1.0:1.1 to indicate cores 0,1 of both groups 0,1
  // etc
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

bool iree_task_topology_format(const iree_task_topology_t* topology,
                               iree_host_size_t buffer_capacity, char* buffer,
                               iree_host_size_t* out_buffer_length) {
  // TODO(benvanik): formatting to match parsing.
  return false;
}

iree_host_size_t iree_task_topology_group_count(
    const iree_task_topology_t* topology) {
  return topology->group_count;
}

const iree_task_topology_group_t* iree_task_topology_get_group(
    const iree_task_topology_t* topology, iree_host_size_t group_index) {
  if (group_index >= topology->group_count) return NULL;
  return &topology->groups[group_index];
}

iree_status_t iree_task_topology_push_group(
    iree_task_topology_t* topology, const iree_task_topology_group_t* group) {
  if (topology->group_count + 1 > topology->group_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "group capacity exceeded");
  }
  iree_task_topology_group_t* dst_group =
      &topology->groups[topology->group_count];
  memcpy(dst_group, group, sizeof(*group));
  dst_group->group_index = topology->group_count++;
  return iree_ok_status();
}

iree_status_t iree_task_topology_from_group_count(
    iree_host_size_t group_count, iree_allocator_t allocator,
    iree_task_topology_t** out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_task_topology_t* topology = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_task_topology_allocate(group_count, allocator, &topology));

  for (iree_host_size_t i = 0; i < group_count; ++i) {
    iree_task_topology_group_t* group = &topology->groups[i];
    group->group_index = i;
    snprintf(group->name, IREE_ARRAYSIZE(group->name), "worker[%d]", (int)i);
    iree_thread_affinity_set_any(&group->ideal_thread_affinity);
    group->constructive_sharing_mask = IREE_TASK_TOPOLOGY_GROUP_MASK_ALL;
  }
  topology->group_count = group_count;

  *out_topology = topology;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Runs the cpuinfo initializer which caches its result on the first call.
// Returns a failure if cpuinfo does not support the CPU/platform.
static iree_status_t iree_task_topology_ensure_cpuinfo_available() {
  if (!cpuinfo_initialize()) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "cpuinfo failed to initialize");
  }
  return iree_ok_status();
}

// Returns the core of the calling thread or NULL if not supported.
// We wrap this here because cpuinfo only returns non-NULL on linux.
static const struct cpuinfo_core* iree_task_topology_get_current_core() {
  const struct cpuinfo_core* current_core = cpuinfo_get_current_core();
#if defined(IREE_PLATFORM_WINDOWS)
  // TODO(benvanik): upstream into cpuinfo.
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
  out_affinity->id = (uint32_t)(uintptr_t)processor;
#elif defined(__linux__)
  out_affinity->id = processor->linux_id;
#elif defined(_WIN32) || defined(__CYGWIN__)
  out_affinity->group = processor->windows_group_id;
  out_affinity->id = processor->windows_processor_id;
#else
  // WASM? Unusued today.
  out_affinity->specified = 0;
#endif  // cpuinfo-like platform field
}

// Returns a bitset with all *processors* that share the same |cache|.
static uint64_t iree_task_topology_calculate_cache_bits(
    const struct cpuinfo_cache* cache) {
  if (!cache) return 0;
  uint64_t mask = 0;
  for (uint32_t processor_i = 0; processor_i < cache->processor_count;
       ++processor_i) {
    mask |= 1ull << (cache->processor_start + processor_i);
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

// Populates |our_group| with the information from |core|.
static void iree_task_topology_group_initialize_from_core(
    uint32_t group_index, const struct cpuinfo_core* core,
    iree_task_topology_group_t* out_group) {
  memset(out_group, 0, sizeof(*out_group));
  out_group->group_index = group_index;
  snprintf(out_group->name, IREE_ARRAYSIZE(out_group->name), "worker[%u]",
           group_index);

  // Guess: always pick the first processor in a core.
  // When pinning to threads we'll take into account whether the core is SMT
  // and use all threads anyway so this alignment is just helpful for debugging.
  uint32_t processor_i = core->processor_start;
  out_group->processor_index = processor_i;

  const struct cpuinfo_processor* processor =
      cpuinfo_get_processor(processor_i);
  iree_task_topology_set_affinity_from_processor(
      processor, &out_group->ideal_thread_affinity);
}

// Fixes constructive_sharing_mask values such that they represent other chosen
// topology groups instead of processor indices. We do this so that code using
// the topology groups doesn't need to know anything about which physical
// processor IDs a particular group is mapped to.
static void iree_task_topology_fixup_constructive_sharing_masks(
    iree_task_topology_t* topology) {
  // O(n^2), but n is always <= 64 (and often <= 8).
  for (iree_host_size_t i = 0; i < topology->group_count; ++i) {
    iree_task_topology_group_t* group = &topology->groups[i];

    // Compute the processors that we can constructively share with.
    uint64_t constructive_sharing_mask =
        iree_task_topology_calculate_constructive_sharing_mask(
            cpuinfo_get_processor(group->processor_index));

    iree_task_topology_group_mask_t group_mask = 0;
    for (iree_host_size_t j = 0; j < topology->group_count; ++j) {
      if (i == j) continue;
      const iree_task_topology_group_t* other_group = &topology->groups[j];
      uint64_t group_processor_bits = 1ull << other_group->processor_index;
      if (constructive_sharing_mask & group_processor_bits) {
        group_mask |= 1ull << other_group->group_index;
      }
    }

    group->constructive_sharing_mask = group_mask;
  }
}

// Matches all cores.
static bool iree_task_topology_core_filter_all(const struct cpuinfo_core* core,
                                               uintptr_t user_data) {
  return true;
}

iree_status_t iree_task_topology_from_physical_cores(
    iree_host_size_t max_core_count, iree_allocator_t allocator,
    iree_task_topology_t** out_topology) {
  return iree_task_topology_from_physical_cores_with_filter(
      iree_task_topology_core_filter_all, 0, max_core_count, allocator,
      out_topology);
}

// Matches only cores with the uarch as specified in |user_data|.
static bool iree_task_topology_core_filter_uarch(
    const struct cpuinfo_core* core, uintptr_t user_data) {
  return core->uarch == user_data;
}

iree_status_t iree_task_topology_from_physical_cores_with_uarch(
    uint32_t cpuinfo_uarch, iree_host_size_t max_core_count,
    iree_allocator_t allocator, iree_task_topology_t** out_topology) {
  return iree_task_topology_from_physical_cores_with_filter(
      iree_task_topology_core_filter_uarch, cpuinfo_uarch, max_core_count,
      allocator, out_topology);
}

iree_status_t iree_task_topology_from_physical_cores_with_filter(
    iree_task_topology_core_filter_t filter_fn, uintptr_t filter_fn_data,
    iree_host_size_t max_core_count, iree_allocator_t allocator,
    iree_task_topology_t** out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_task_topology_ensure_cpuinfo_available());

  // Count cores that match the filter.
  iree_host_size_t core_count = 0;
  for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
    const struct cpuinfo_core* core = cpuinfo_get_core(i);
    if (filter_fn(core, filter_fn_data)) ++core_count;
  }
  core_count = iree_min(core_count, max_core_count);

  iree_task_topology_t* topology = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_task_topology_allocate(core_count, allocator, &topology));

  // Build each core up to the max allowed.
  // TODO(benvanik): if our group_count <= core_count/2 then distribute better;
  // for now we just do a straight-line through (cores 0-N) when instead we may
  // want to take advantage of L3 cache info (half of groups on one L3 cache,
  // half of groups on another, etc).
  topology->group_count = core_count;
  for (uint32_t core_i = 0, group_i = 0; group_i < topology->group_count;
       ++core_i) {
    // Rotate the core ID so that we avoid setting the affinity to the calling
    // thread which we assume is something the user has plans for and doesn't
    // want to have our workers stealing their time.
    const struct cpuinfo_core* core =
        cpuinfo_get_core(iree_task_topology_rotate_from_base_core(core_i));
    if (filter_fn(core, filter_fn_data)) {
      iree_task_topology_group_initialize_from_core(group_i, core,
                                                    &topology->groups[group_i]);
      ++group_i;
    }
  }

  iree_task_topology_fixup_constructive_sharing_masks(topology);
  *out_topology = topology;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_task_topology_from_unique_l2_cache_groups(
    iree_host_size_t max_group_count, iree_allocator_t allocator,
    iree_task_topology_t** out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_task_topology_ensure_cpuinfo_available());

  iree_host_size_t cache_count = cpuinfo_get_l2_caches_count();
  cache_count = iree_min(cache_count, max_group_count);

  iree_task_topology_t* topology = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_task_topology_allocate(cache_count, allocator, &topology));

  // TODO(benvanik): iree_task_topology_rotate_from_base_core to offset all of
  // the selection here (while still preserving the cache groups). May need to
  // rework this to instead walk the core list and skip until a new cache is
  // found instead of starting with the cache list.

  // TODO(benvanik): if our group_count <= cache_count/2 then distribute better;
  // we could use l3 cache in addition to ensure we are selecting cores that do
  // (or do not) share.
  topology->group_count = cache_count;
  for (uint32_t cache_i = 0, group_i = 0; group_i < topology->group_count;
       ++cache_i) {
    const struct cpuinfo_cache* cache = cpuinfo_get_l2_cache(cache_i);
    const struct cpuinfo_core* core =
        cpuinfo_get_processor(cache->processor_start)->core;
    iree_task_topology_group_initialize_from_core(group_i, core,
                                                  &topology->groups[group_i]);
    ++group_i;
  }

  iree_task_topology_fixup_constructive_sharing_masks(topology);
  *out_topology = topology;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
