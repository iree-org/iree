// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linux sysfs-based topology detection.
// Defers to cpuinfo if available, otherwise provides sysfs implementation.
//
// Documentation:
// https://docs.kernel.org/admin-guide/abi-stable-files.html#abi-file-stable-sysfs-devices-system-cpu

// Must define _GNU_SOURCE before includes to get CPU_* macros from sched.h.
#define _GNU_SOURCE

#include "iree/base/internal/math.h"
#include "iree/base/internal/sysfs.h"
#include "iree/task/topology.h"

#if !defined(IREE_TASK_USE_CPUINFO) && defined(IREE_PLATFORM_LINUX) && \
    !defined(IREE_PLATFORM_EMSCRIPTEN)

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

// Maximum cache indices to scan when enumerating cache hierarchy.
// Modern CPUs have 3-4 indices (L1i, L1d, L2, L3), with some server CPUs
// having L4 (5 indices). This conservative limit handles exotic architectures.
#define IREE_SYSFS_MAX_CACHE_INDICES 8

//===----------------------------------------------------------------------===//
// Topology query functions
//===----------------------------------------------------------------------===//

// Callback context for counting maximum CPU ID in a CPU list.
typedef struct {
  uint32_t max_cpu_id;
} iree_sysfs_cpu_count_context_t;

// Callback for CPU list enumeration that tracks the maximum CPU ID.
static bool iree_sysfs_count_cpus_callback(uint32_t start_cpu, uint32_t end_cpu,
                                           void* user_data) {
  iree_sysfs_cpu_count_context_t* ctx =
      (iree_sysfs_cpu_count_context_t*)user_data;
  if (end_cpu - 1 > ctx->max_cpu_id) {
    ctx->max_cpu_id = end_cpu - 1;
  }
  return true;  // Continue enumeration.
}

// Queries the number of logical processors from sysfs.
// Returns 0 on error (caller should use fallback).
static uint32_t iree_sysfs_query_processor_count(void) {
  // Try /sys/devices/system/cpu/present first (most reliable).
  char path[256];
  snprintf(path, sizeof(path), "%s/cpu/present", iree_sysfs_get_root_path());
  char buffer[256];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length);
  if (iree_status_is_ok(status)) {
    iree_sysfs_cpu_count_context_t ctx = {.max_cpu_id = 0};
    status = iree_sysfs_parse_cpu_list(iree_make_string_view(buffer, length),
                                       iree_sysfs_count_cpus_callback, &ctx);
    if (iree_status_is_ok(status)) {
      uint32_t count = ctx.max_cpu_id + 1;
      return count;  // Convert max ID to count.
    }
  }
  iree_status_ignore(status);

  // Fallback to /sys/devices/system/cpu/kernel_max.
  snprintf(path, sizeof(path), "%s/cpu/kernel_max", iree_sysfs_get_root_path());
  uint32_t kernel_max = 0;
  status = iree_sysfs_read_uint32(path, &kernel_max);
  if (iree_status_is_ok(status)) {
    return kernel_max + 1;  // kernel_max is 0-based.
  }
  iree_status_ignore(status);

  return 0;  // Unknown.
}

// Reads the core ID for a specific logical processor.
// Returns IREE_STATUS_NOT_FOUND if the file doesn't exist.
static iree_status_t iree_sysfs_query_core_id(uint32_t processor,
                                              uint32_t* out_core_id) {
  char path[256];
  snprintf(path, sizeof(path), "%s/cpu/cpu%u/topology/core_id",
           iree_sysfs_get_root_path(), processor);
  return iree_sysfs_read_uint32(path, out_core_id);
}

// Gets the current CPU ID using the getcpu syscall.
// Returns 0 on error (safe default).
static uint32_t iree_sysfs_query_current_cpu(void) {
#if defined(__linux__) && defined(__NR_getcpu)
  unsigned cpu = 0;
  if (syscall(__NR_getcpu, &cpu, NULL, NULL) == 0) {
    return (uint32_t)cpu;
  }
#endif  // __linux__ && __NR_getcpu

  // Fallback to CPU 0.
  return 0;
}

// Linux uses -1 as sentinel for "not available." When read as unsigned
// this can be UINT16_MAX or UINT32_MAX (some sysfs inconsistency?).
static inline bool iree_sysfs_is_valid_cluster(uint32_t cluster_id) {
  return cluster_id != UINT16_MAX && cluster_id != UINT32_MAX;
}

// Reads the cluster ID for a specific logical processor.
// Tries multiple fallback sources if cluster_id is not available.
// Returns IREE_STATUS_NOT_FOUND if no cluster info is available.
static iree_status_t iree_sysfs_query_cluster_id(uint32_t processor,
                                                 uint32_t* out_cluster_id) {
  *out_cluster_id = UINT32_MAX;
  char path[256];

  // Try cluster_id first (kernel 5.16+).
  snprintf(path, sizeof(path), "%s/cpu/cpu%u/topology/cluster_id",
           iree_sysfs_get_root_path(), processor);
  iree_status_t status = iree_sysfs_read_uint32(path, out_cluster_id);
  if (iree_status_is_ok(status)) {
    return status;
  }
  iree_status_ignore(status);

  // Fallback to physical_package_id (socket/package).
  snprintf(path, sizeof(path), "%s/cpu/cpu%u/topology/physical_package_id",
           iree_sysfs_get_root_path(), processor);
  status = iree_sysfs_read_uint32(path, out_cluster_id);
  if (iree_status_is_ok(status)) {
    return status;
  }
  iree_status_ignore(status);

  // No cluster info available.
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no cluster information available for CPU %u",
                          processor);
}

// Reads the CPU capacity for a specific logical processor.
// Used for ARM big.LITTLE detection and performance level classification.
// Returns 0 if not available (x86 systems or older kernels).
static uint32_t iree_sysfs_query_cpu_capacity(uint32_t processor) {
  char path[256];
  snprintf(path, sizeof(path), "%s/cpu/cpu%u/cpu_capacity",
           iree_sysfs_get_root_path(), processor);
  uint32_t capacity = 0;
  iree_status_ignore(iree_sysfs_read_uint32(path, &capacity));
  return capacity;
}

//===----------------------------------------------------------------------===//
// Cache hierarchy queries
//===----------------------------------------------------------------------===//

// Cache information for a single cache level.
typedef struct {
  // Cache size in bytes (0 if not available).
  uint64_t size;
  // Cache level (1, 2, 3, etc.).
  uint32_t level;
  // True if this is a data or unified cache.
  bool is_data_cache;
} iree_sysfs_cache_info_t;

// Queries cache information for a specific cache index.
// Returns IREE_STATUS_NOT_FOUND if the cache index doesn't exist.
static iree_status_t iree_sysfs_query_cache_level(
    uint32_t processor, uint32_t cache_index,
    iree_sysfs_cache_info_t* out_cache) {
  // Read cache type (Data, Instruction, or Unified).
  // If this fails the cache index doesn't exist.
  char path[256];
  snprintf(path, sizeof(path), "%s/cpu/cpu%u/cache/index%u/type",
           iree_sysfs_get_root_path(), processor, cache_index);
  char buffer[64];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length));

  iree_string_view_t type_str =
      iree_string_view_trim(iree_make_string_view(buffer, length));
  out_cache->is_data_cache =
      iree_string_view_starts_with(type_str, IREE_SV("Data")) ||
      iree_string_view_starts_with(type_str, IREE_SV("Unified"));

  // Read cache level (optional - ignore failures).
  snprintf(path, sizeof(path), "%s/cpu/cpu%u/cache/index%u/level",
           iree_sysfs_get_root_path(), processor, cache_index);
  uint32_t level = 0;
  iree_status_ignore(iree_sysfs_read_uint32(path, &level));
  out_cache->level = level;

  // Read cache size (optional - ignore failures).
  snprintf(path, sizeof(path), "%s/cpu/cpu%u/cache/index%u/size",
           iree_sysfs_get_root_path(), processor, cache_index);
  uint64_t size = 0;
  iree_status_ignore(iree_sysfs_read_size(path, &size));
  out_cache->size = size;

  return iree_ok_status();
}

// Queries all cache levels for a processor and populates the group's cache
// info. This enumerates cache/index0, index1, index2, etc. and extracts
// L1/L2/L3 sizes.
static void iree_sysfs_populate_cache_info(
    uint32_t processor, iree_task_topology_group_t* out_group) {
  // Initialize to zero (fallback if cache info unavailable).
  out_group->caches.l1_data = 0;
  out_group->caches.l2_data = 0;
  out_group->caches.l3_data = 0;

  // Enumerate cache indices (typically 0-3).
  for (uint32_t cache_index = 0; cache_index < IREE_SYSFS_MAX_CACHE_INDICES;
       ++cache_index) {
    iree_sysfs_cache_info_t cache = {0};
    iree_status_t status =
        iree_sysfs_query_cache_level(processor, cache_index, &cache);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;  // No more cache levels.
    }

    // Skip instruction-only caches.
    if (!cache.is_data_cache) {
      continue;
    }

    // Store size based on level.
    switch (cache.level) {
      case 1:
        out_group->caches.l1_data = cache.size;
        break;
      case 2:
        out_group->caches.l2_data = cache.size;
        break;
      case 3:
        out_group->caches.l3_data = cache.size;
        break;
      default:
        break;  // L4+ caches ignored.
    }
  }
}

//===----------------------------------------------------------------------===//
// Public API implementation
//===----------------------------------------------------------------------===//

iree_host_size_t iree_task_topology_query_node_count(void) {
  // Count unique cluster IDs across all processors.
  const uint32_t processor_count = iree_sysfs_query_processor_count();
  if (processor_count == 0) {
    // Fallback to single node.
    return 1;
  }

  // Track unique cluster IDs and count them as we discover new ones.
  cpu_set_t cluster_set;
  CPU_ZERO(&cluster_set);
  iree_host_size_t unique_clusters = 0;
  for (uint32_t cpu = 0; cpu < processor_count; ++cpu) {
    uint32_t cluster_id = 0;
    iree_status_t status = iree_sysfs_query_cluster_id(cpu, &cluster_id);
    if (iree_status_is_ok(status) && iree_sysfs_is_valid_cluster(cluster_id)) {
      if (!CPU_ISSET(cluster_id, &cluster_set)) {
        CPU_SET(cluster_id, &cluster_set);
        ++unique_clusters;
      }
    }
    iree_status_ignore(status);
  }

  return unique_clusters > 0 ? unique_clusters : 1;
}

iree_task_topology_node_id_t iree_task_topology_query_current_node(void) {
  const uint32_t current_cpu = iree_sysfs_query_current_cpu();
  uint32_t cluster_id = 0;
  iree_status_t status = iree_sysfs_query_cluster_id(current_cpu, &cluster_id);
  if (iree_status_is_ok(status)) {
    if (!iree_sysfs_is_valid_cluster(cluster_id)) {
      return 0;  // Invalid clusters are node 0.
    }
    return cluster_id;
  }
  iree_status_ignore(status);
  return 0;  // Fallback to node 0.
}

//===----------------------------------------------------------------------===//
// Constructive sharing mask utilities
//===----------------------------------------------------------------------===//

// Context for building processor bitmask from CPU list.
typedef struct {
  cpu_set_t processor_mask;
} iree_sysfs_processor_mask_context_t;

// Callback to accumulate processor IDs into a bitmask.
static bool iree_sysfs_accumulate_processor_mask(uint32_t start_cpu,
                                                 uint32_t end_cpu,
                                                 void* user_data) {
  iree_sysfs_processor_mask_context_t* ctx =
      (iree_sysfs_processor_mask_context_t*)user_data;
  for (uint32_t cpu = start_cpu; cpu < end_cpu; ++cpu) {
    CPU_SET(cpu, &ctx->processor_mask);
  }
  return true;  // Continue enumeration.
}

// Reads shared_cpu_list for a given cache index into processor bitmask.
// Returns true if successful, false if the file doesn't exist or can't be
// parsed.
static bool iree_sysfs_read_cache_shared_cpu_list(uint32_t processor,
                                                  uint32_t cache_index,
                                                  cpu_set_t* out_mask) {
  char path[256];
  snprintf(path, sizeof(path), "%s/cpu/cpu%u/cache/index%u/shared_cpu_list",
           iree_sysfs_get_root_path(), processor, cache_index);

  char buffer[256];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_sysfs_read_small_file(path, buffer, sizeof(buffer), &length);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return false;
  }

  // Parse CPU list into bitmask.
  iree_sysfs_processor_mask_context_t ctx;
  CPU_ZERO(&ctx.processor_mask);
  status =
      iree_sysfs_parse_cpu_list(iree_make_string_view(buffer, length),
                                iree_sysfs_accumulate_processor_mask, &ctx);
  const bool valid_bitmask = iree_status_is_ok(status);
  iree_status_ignore(status);
  *out_mask = ctx.processor_mask;
  return valid_bitmask;
}

// Finds the best cache level for constructive sharing.
// Prefers L3 Data/Unified, falls back to L2 Data/Unified.
// Populates |out_mask| with processors sharing that cache level.
// Returns true if a mask was found, false otherwise.
static bool iree_sysfs_find_sharing_cache_mask(uint32_t processor,
                                               cpu_set_t* out_mask) {
  cpu_set_t l3_mask, l2_mask;
  CPU_ZERO(&l3_mask);
  CPU_ZERO(&l2_mask);
  bool found_l3 = false;
  bool found_l2 = false;

  // Scan cache indices looking for L3 (preferred) and L2 (fallback).
  for (uint32_t cache_index = 0; cache_index < IREE_SYSFS_MAX_CACHE_INDICES;
       ++cache_index) {
    iree_sysfs_cache_info_t cache = {0};
    iree_status_t status =
        iree_sysfs_query_cache_level(processor, cache_index, &cache);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;  // No more cache levels.
    }

    // Only consider Data or Unified caches.
    if (!cache.is_data_cache) {
      continue;
    }

    cpu_set_t shared_mask;
    if (iree_sysfs_read_cache_shared_cpu_list(processor, cache_index,
                                              &shared_mask)) {
      if (cache.level == 3) {
        l3_mask = shared_mask;
        found_l3 = true;
        break;  // L3 is best, use it immediately.
      } else if (cache.level == 2) {
        l2_mask = shared_mask;
        found_l2 = true;
      }
    }
  }

  // Prefer L3, fall back to L2.
  if (found_l3) {
    *out_mask = l3_mask;
    return true;
  } else if (found_l2) {
    *out_mask = l2_mask;
    return true;
  }
  return false;
}

// Builds constructive sharing masks based on cache sharing.
// We parse shared_cpu_list from cache/index*/shared_cpu_list to determine
// which processors share cache levels. We prefer L3 cache sharing, falling
// back to L2 if L3 is not available.
iree_status_t iree_task_topology_fixup_constructive_sharing_masks(
    iree_task_topology_t* topology) {
  // O(n^2), but n is always <= 64 (and often <= 8).
  for (iree_host_size_t i = 0; i < topology->group_count; ++i) {
    iree_task_topology_group_t* group = &topology->groups[i];
    uint32_t processor = group->processor_index;

    // Find processors that share L3 (or L2 as fallback) cache.
    cpu_set_t processor_sharing_mask;
    const bool has_sharing_mask =
        iree_sysfs_find_sharing_cache_mask(processor, &processor_sharing_mask);

    // Convert processor bitmask to group bitmask.
    // Only processors in the topology can contribute to the group mask.
    iree_task_topology_group_mask_t group_mask = 0;
    if (has_sharing_mask) {
      for (iree_host_size_t j = 0; j < topology->group_count; ++j) {
        const iree_task_topology_group_t* other_group = &topology->groups[j];
        uint32_t other_processor = other_group->processor_index;
        if (CPU_ISSET(other_processor, &processor_sharing_mask)) {
          group_mask |= 1ull << other_group->group_index;
        }
      }
    }

    group->constructive_sharing_mask = group_mask;
  }

  return iree_ok_status();
}

iree_status_t iree_task_topology_initialize_from_logical_cpu_set(
    iree_host_size_t cpu_count, const uint32_t* cpu_ids,
    iree_task_topology_t* out_topology) {
  // Validate input.
  if (cpu_count >= IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many CPUs specified (%" PRIhsz
                            " provided for a max capacity of %zu)",
                            cpu_count, IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT);
  }
  uint32_t processor_count = iree_sysfs_query_processor_count();
  if (processor_count == 0) {
    // Cannot query system topology - fall back to single-group topology.
    iree_task_topology_initialize_from_group_count(1, out_topology);
    return iree_ok_status();
  }
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
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)cpu_count);

  iree_task_topology_initialize(out_topology);
  out_topology->group_count = cpu_count;

  // Populate each group from sysfs.
  for (iree_host_size_t i = 0; i < cpu_count; ++i) {
    iree_task_topology_group_t* group = &out_topology->groups[i];
    iree_task_topology_group_initialize(i, group);
    group->processor_index = cpu_ids[i];

    // Query cache info.
    iree_sysfs_populate_cache_info(cpu_ids[i], group);

    // Set thread affinity (platform-specific).
    group->ideal_thread_affinity.id_assigned = 1;
    group->ideal_thread_affinity.id = cpu_ids[i];

    // Query cluster ID for affinity grouping.
    uint32_t cluster_id = 0;
    iree_status_t cluster_status =
        iree_sysfs_query_cluster_id(cpu_ids[i], &cluster_id);
    if (iree_status_is_ok(cluster_status)) {
      group->ideal_thread_affinity.group = cluster_id;
    }
    iree_status_ignore(cluster_status);
  }

  iree_status_t status =
      iree_task_topology_fixup_constructive_sharing_masks(out_topology);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Cache domain enumeration
//===----------------------------------------------------------------------===//

// Cache domain descriptor grouping cores that share L3 cache.
typedef struct {
  // All cores in this domain.
  cpu_set_t cores;
  // Processor bitmask defining this domain.
  cpu_set_t sharing_mask;
} iree_sysfs_cache_domain_t;

// Groups cores into cache domains based on L3 sharing masks.
// Returns the number of domains found. If cache info is unavailable, returns 1
// domain containing all cores (graceful degradation).
static iree_host_size_t iree_sysfs_enumerate_cache_domains(
    iree_host_size_t core_count, const uint32_t* core_map,
    iree_sysfs_cache_domain_t* out_domains, iree_host_size_t max_domains) {
  if (core_count == 0 || max_domains == 0) return 0;

  // Build domains by grouping cores with identical sharing masks.
  iree_host_size_t domain_count = 0;
  for (iree_host_size_t i = 0; i < core_count; ++i) {
    uint32_t processor = core_map[i];
    cpu_set_t sharing_mask;
    CPU_ZERO(&sharing_mask);
    const bool has_mask =
        iree_sysfs_find_sharing_cache_mask(processor, &sharing_mask);

    // If no cache info available, put all cores in one domain.
    if (!has_mask) {
      CPU_ZERO(&out_domains[0].cores);
      for (iree_host_size_t j = 0; j < core_count; ++j) {
        CPU_SET(core_map[j], &out_domains[0].cores);
      }
      CPU_ZERO(&out_domains[0].sharing_mask);
      return 1;  // Single domain fallback.
    }

    // Check if this sharing mask matches an existing domain.
    bool found_domain = false;
    for (iree_host_size_t d = 0; d < domain_count; ++d) {
      if (CPU_EQUAL(&out_domains[d].sharing_mask, &sharing_mask)) {
        // Add to existing domain.
        CPU_SET(processor, &out_domains[d].cores);
        found_domain = true;
        break;
      }
    }

    // Create new domain if this is a unique sharing mask.
    if (!found_domain && domain_count < max_domains) {
      CPU_ZERO(&out_domains[domain_count].cores);
      CPU_SET(processor, &out_domains[domain_count].cores);
      out_domains[domain_count].sharing_mask = sharing_mask;
      ++domain_count;
    }
  }

  // Sort domains by lowest core ID for deterministic ordering.
  // Find the lowest set bit in each domain's cores cpu_set_t.
  for (iree_host_size_t i = 0; i < domain_count - 1; ++i) {
    for (iree_host_size_t j = i + 1; j < domain_count; ++j) {
      // Find first set CPU in each domain.
      int first_i = -1, first_j = -1;
      for (int cpu = 0; cpu < CPU_SETSIZE && (first_i < 0 || first_j < 0);
           ++cpu) {
        if (first_i < 0 && CPU_ISSET(cpu, &out_domains[i].cores)) first_i = cpu;
        if (first_j < 0 && CPU_ISSET(cpu, &out_domains[j].cores)) first_j = cpu;
      }
      if (first_j >= 0 && first_i >= 0 && first_j < first_i) {
        iree_sysfs_cache_domain_t temp = out_domains[i];
        out_domains[i] = out_domains[j];
        out_domains[j] = temp;
      }
    }
  }

  return domain_count;
}

iree_status_t iree_task_topology_initialize_from_physical_cores(
    iree_task_topology_node_id_t node_id,
    iree_task_topology_performance_level_t performance_level,
    iree_task_topology_distribution_t distribution,
    iree_host_size_t max_core_count, iree_task_topology_t* out_topology) {
  uint32_t processor_count = iree_sysfs_query_processor_count();
  if (processor_count == 0) {
    // Fallback to single-group topology.
    iree_task_topology_initialize_from_group_count(1, out_topology);
    return iree_ok_status();
  }

  max_core_count = iree_min(max_core_count, IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT);

  // Detect heterogeneous systems (ARM big.LITTLE) by scanning CPU capacities.
  // Capacity values are normalized to 1024 for the highest-performance cores.
  // If all cores have the same capacity (or capacity unavailable), system is
  // treated as homogeneous and performance_level filtering is skipped.
  uint32_t max_capacity = 0;
  uint32_t min_capacity = UINT32_MAX;
  for (uint32_t cpu = 0; cpu < processor_count; ++cpu) {
    const uint32_t capacity = iree_sysfs_query_cpu_capacity(cpu);
    if (capacity > 0) {
      max_capacity = iree_max(max_capacity, capacity);
      min_capacity = iree_min(min_capacity, capacity);
    }
  }
  const bool is_heterogeneous =
      max_capacity > 0 && max_capacity != min_capacity;
  const uint32_t capacity_threshold = (max_capacity * 3) / 4;  // 75% of max.

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)max_core_count);

  iree_task_topology_initialize(out_topology);

  // Find unique cores by enumerating processors and grouping by core_id.
  // We build a simple map of core_id -> first processor in that core.
  uint32_t core_map[IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT];
  iree_host_size_t core_count = 0;
  for (uint32_t cpu = 0; cpu < processor_count && core_count < max_core_count;
       ++cpu) {
    uint32_t core_id = 0;
    iree_status_t status = iree_sysfs_query_core_id(cpu, &core_id);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;  // Skip CPUs we can't query.
    }

    // Filter by cluster/node if specified.
    if (node_id != IREE_TASK_TOPOLOGY_NODE_ID_ANY) {
      // Only filter if cluster info is valid and doesn't match.
      uint32_t cluster_id = 0;
      iree_status_t cluster_status =
          iree_sysfs_query_cluster_id(cpu, &cluster_id);
      // When invalid we skip filtering on invalid values to avoid removing all
      // cores on homogeneous systems.
      if (iree_status_is_ok(cluster_status) &&
          iree_sysfs_is_valid_cluster(cluster_id) &&
          cluster_id != (uint32_t)node_id) {
        continue;  // Wrong cluster.
      }
      iree_status_ignore(cluster_status);
    }

    // Filter by performance level on heterogeneous systems (ARM big.LITTLE).
    // On homogeneous systems or when ANY is requested, use all cores.
    if (is_heterogeneous &&
        performance_level != IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY) {
      const uint32_t capacity = iree_sysfs_query_cpu_capacity(cpu);
      const bool is_high_performance = capacity >= capacity_threshold;
      if (performance_level == IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_HIGH &&
          !is_high_performance) {
        continue;  // Skip LITTLE cores when HIGH performance requested.
      }
      if (performance_level == IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_LOW &&
          is_high_performance) {
        continue;  // Skip big cores when LOW performance requested.
      }
    }

    // Check if we've already seen this core.
    bool core_seen = false;
    for (iree_host_size_t i = 0; i < core_count; ++i) {
      const uint32_t existing_cpu = core_map[i];
      uint32_t existing_core_id = 0;
      iree_status_t existing_status =
          iree_sysfs_query_core_id(existing_cpu, &existing_core_id);
      if (iree_status_is_ok(existing_status) && existing_core_id == core_id) {
        core_seen = true;
        break;
      }
      iree_status_ignore(existing_status);
    }
    if (!core_seen) {
      // First processor in this core.
      core_map[core_count++] = cpu;
    }
  }

  // Reorder cores according to distribution strategy across cache domains.
  // COMPACT fills cache domains sequentially, SCATTER distributes round-robin.
  // For COMPACT or single-domain systems, use cores in original order.
  if (core_count > 1 &&
      distribution == IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER) {
    // Enumerate cache domains from the cores we found.
    iree_sysfs_cache_domain_t domains[IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT];
    const iree_host_size_t domain_count = iree_sysfs_enumerate_cache_domains(
        core_count, core_map, domains, IREE_ARRAYSIZE(domains));
    if (domain_count > 1) {
      // SCATTER: Distribute cores evenly across domains using round-robin.
      // This maximizes memory bandwidth by utilizing multiple controllers.
      uint32_t new_core_map[IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT];
      iree_host_size_t new_core_count = 0;

      // Track next CPU to check for each domain.
      int domain_next_cpu[IREE_TASK_TOPOLOGY_GROUP_BIT_COUNT];
      for (iree_host_size_t d = 0; d < domain_count; ++d) {
        domain_next_cpu[d] = 0;
      }

      while (new_core_count < core_count) {
        bool assigned_any = false;
        for (iree_host_size_t d = 0; d < domain_count; ++d) {
          // Find next set CPU in this domain.
          for (int cpu = domain_next_cpu[d]; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &domains[d].cores)) {
              new_core_map[new_core_count++] = (uint32_t)cpu;
              domain_next_cpu[d] = cpu + 1;  // Start after this next time.
              assigned_any = true;
              break;
            }
          }
          if (new_core_count >= core_count) break;
        }

        // All domains exhausted.
        if (!assigned_any) break;
      }

      // Use reordered map.
      for (iree_host_size_t i = 0; i < new_core_count; ++i) {
        core_map[i] = new_core_map[i];
      }
      core_count = new_core_count;
    }
  }

  // Populate topology groups from the unique cores we found.
  out_topology->group_count = core_count;
  for (iree_host_size_t i = 0; i < core_count; ++i) {
    uint32_t processor = core_map[i];
    iree_task_topology_group_t* group = &out_topology->groups[i];
    iree_task_topology_group_initialize(i, group);
    group->processor_index = processor;

    // Query cache info.
    iree_sysfs_populate_cache_info(processor, group);

    // Set thread affinity.
    group->ideal_thread_affinity.id_assigned = 1;
    group->ideal_thread_affinity.id = processor;

    uint32_t cluster_id = 0;
    iree_status_t cluster_status =
        iree_sysfs_query_cluster_id(processor, &cluster_id);
    if (iree_status_is_ok(cluster_status)) {
      group->ideal_thread_affinity.group = cluster_id;
    } else {
      iree_status_ignore(cluster_status);
    }
  }

  iree_status_t status =
      iree_task_topology_fixup_constructive_sharing_masks(out_topology);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // !IREE_TASK_USE_CPUINFO && IREE_PLATFORM_LINUX &&
        // !IREE_PLATFORM_EMSCRIPTEN
