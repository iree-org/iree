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
    !defined(IREE_PLATFORM_WASM)

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
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(&builder, "%s/cpu/present",
                                                  iree_sysfs_get_root_path()));
  char buffer[256];
  iree_host_size_t length = 0;
  if (iree_sysfs_try_read_small_file(path, buffer, sizeof(buffer), &length)) {
    iree_sysfs_cpu_count_context_t ctx = {.max_cpu_id = 0};
    if (iree_sysfs_try_parse_cpu_list(iree_make_string_view(buffer, length),
                                      iree_sysfs_count_cpus_callback, &ctx)) {
      uint32_t count = ctx.max_cpu_id + 1;
      return count;  // Convert max ID to count.
    }
  }

  // Fallback to /sys/devices/system/cpu/kernel_max.
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(&builder, "%s/cpu/kernel_max",
                                                  iree_sysfs_get_root_path()));
  uint32_t kernel_max = 0;
  if (iree_sysfs_try_read_uint32(path, &kernel_max)) {
    return kernel_max + 1;  // kernel_max is 0-based.
  }

  return 0;  // Unknown.
}

// Queries the CPU set currently available to the calling thread.
// Returns false if the platform query fails, in which case callers should not
// constrain sysfs topology discovery to an affinity mask.
static bool iree_sysfs_query_current_affinity(cpu_set_t* out_cpu_set) {
  CPU_ZERO(out_cpu_set);
  // Against a redirected root the tree describes a machine unrelated to this
  // host, so the host's affinity would spuriously drop (or admit) processors.
  // Report "unknown" so discovery covers every processor the tree describes.
  if (!iree_sysfs_host_matches_root()) return false;
  return sched_getaffinity(/*pid=*/0, sizeof(*out_cpu_set), out_cpu_set) == 0;
}

// Returns true if |processor| is currently available to the calling thread.
static bool iree_sysfs_is_processor_available(
    uint32_t processor, const cpu_set_t* current_affinity) {
  if (!current_affinity) return true;
  return processor < CPU_SETSIZE && CPU_ISSET(processor, current_affinity);
}

// Reads the core ID for a specific logical processor.
// Returns false if the file doesn't exist or can't be parsed.
static bool iree_sysfs_try_query_core_id(uint32_t processor,
                                         uint32_t* out_core_id) {
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/cpu/cpu%u/topology/core_id", iree_sysfs_get_root_path(),
      processor));
  return iree_sysfs_try_read_uint32(path, out_core_id);
}

// Gets the current CPU and NUMA node using the getcpu syscall.
// Returns false on error or when the sysfs root does not describe
// this host.
static bool iree_sysfs_try_query_current_cpu_and_node(uint32_t* out_cpu,
                                                      uint32_t* out_node) {
  *out_cpu = 0;
  *out_node = 0;
#if defined(__linux__) && defined(__NR_getcpu)
  if (!iree_sysfs_host_matches_root()) return false;
  unsigned cpu = 0;
  unsigned node = 0;
  if (syscall(__NR_getcpu, &cpu, &node, NULL) == 0) {
    *out_cpu = (uint32_t)cpu;
    *out_node = (uint32_t)node;
    return true;
  }
#endif  // __linux__ && __NR_getcpu
  return false;
}

// Gets the current CPU ID using the getcpu syscall.
// Returns 0 on error.
static uint32_t iree_sysfs_query_current_cpu(void) {
  uint32_t cpu = 0;
  uint32_t node = 0;
  iree_sysfs_try_query_current_cpu_and_node(&cpu, &node);
  return cpu;
}

// Linux uses -1 as sentinel for "not available." When read as unsigned
// this can be UINT16_MAX or UINT32_MAX (some sysfs inconsistency?).
static inline bool iree_sysfs_is_valid_node(uint32_t node_id) {
  return node_id != UINT16_MAX && node_id != UINT32_MAX;
}

// Reads the physical package (socket) id for a logical processor. This is the
// coarse fallback for NUMA node identity when the kernel does not expose the
// /sys/devices/system/node/ hierarchy. Returns false if unavailable.
static bool iree_sysfs_try_query_package_id(uint32_t processor,
                                            uint32_t* out_package_id) {
  *out_package_id = UINT32_MAX;
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/cpu/cpu%u/topology/physical_package_id",
      iree_sysfs_get_root_path(), processor));
  return iree_sysfs_try_read_uint32(path, out_package_id);
}

//===----------------------------------------------------------------------===//
// NUMA node table
//===----------------------------------------------------------------------===//
#define IREE_TASK_TOPOLOGY_SYSFS_NODE_LIST_BUFFER_SIZE 1024

#define IREE_TASK_TOPOLOGY_SYSFS_MAX_CPU_ID 4096

#define IREE_TASK_TOPOLOGY_SYSFS_NODE_UNKNOWN UINT8_MAX
static_assert(IREE_TASK_TOPOLOGY_MAX_NODES <=
                  IREE_TASK_TOPOLOGY_SYSFS_NODE_UNKNOWN,
              "node table indices are uint8_t with one value reserved as the "
              "unknown sentinel");

typedef struct iree_sysfs_numa_table_t {
  // True if the kernel exposed a usable node hierarchy.
  bool valid;
  // Number of CPU-nodes stored in node_ids (max IREE_TASK_TOPOLOGY_MAX_NODES).
  iree_host_size_t node_count;
  // Number of CPU-nodes the kernel exposes. Exceeds node_count on
  // machines with more nodes than the table holds.
  iree_host_size_t total_node_count;
  // Node ids in enumeration order. Sparse/non-contiguous values are preserved.
  uint32_t node_ids[IREE_TASK_TOPOLOGY_MAX_NODES];
  // Maps logical CPU id -> index into node_ids, or
  // IREE_TASK_TOPOLOGY_SYSFS_NODE_UNKNOWN.
  uint8_t cpu_node_index[IREE_TASK_TOPOLOGY_SYSFS_MAX_CPU_ID];
} iree_sysfs_numa_table_t;

// Marks every CPU in a parsed cpulist range as belonging to the node currently
// being ingested.
typedef struct {
  iree_sysfs_numa_table_t* table;
  uint8_t node_index;
  bool any_cpu;
} iree_sysfs_numa_ingest_t;

// Check if the list names any CPU we can index.
static bool iree_sysfs_numa_probe_cb(uint32_t start_cpu, uint32_t end_cpu,
                                     void* user_data) {
  iree_sysfs_numa_ingest_t* ctx = (iree_sysfs_numa_ingest_t*)user_data;
  if (start_cpu < IREE_TASK_TOPOLOGY_SYSFS_MAX_CPU_ID && end_cpu > start_cpu) {
    ctx->any_cpu = true;
  }
  return true;
}

static bool iree_sysfs_numa_commit_cb(uint32_t start_cpu, uint32_t end_cpu,
                                      void* user_data) {
  iree_sysfs_numa_ingest_t* ctx = (iree_sysfs_numa_ingest_t*)user_data;
  for (uint32_t cpu = start_cpu; cpu < end_cpu; ++cpu) {
    if (cpu >= IREE_TASK_TOPOLOGY_SYSFS_MAX_CPU_ID) continue;
    ctx->table->cpu_node_index[cpu] = ctx->node_index;
  }
  return true;  // continue
}

// Ingests one node id: reads its cpulist and records CPU ownership. Nodes with
// a readable but empty cpulist (memory-only nodes: CXL, Optane, some SNC
// configs) are skipped so they never become topologies with zero worker groups.
static void iree_sysfs_numa_table_ingest_node(iree_sysfs_numa_table_t* table,
                                              uint32_t node_id) {
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/node/node%u/cpulist", iree_sysfs_get_root_path(), node_id));
  char cpulist_buffer[IREE_TASK_TOPOLOGY_SYSFS_NODE_LIST_BUFFER_SIZE];
  iree_host_size_t length = 0;
  iree_status_t status = iree_sysfs_read_small_file(
      path, cpulist_buffer, sizeof(cpulist_buffer), &length);
  // Truncating a node list would silently report the wrong nodes, so a file
  // that exists but does not fit is fatal. Absent or unreadable falls back.
  if (iree_status_is_out_of_range(status)) {
    IREE_CHECK_OK(status);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }
  const iree_string_view_t cpulist =
      iree_make_string_view(cpulist_buffer, length);

  iree_sysfs_numa_ingest_t ctx = {
      .table = table,
      .node_index = (uint8_t)table->node_count,
      .any_cpu = false,
  };
  if (iree_sysfs_try_parse_cpu_list(cpulist, iree_sysfs_numa_probe_cb, &ctx) &&
      ctx.any_cpu) {  // only CPU-bearing nodes are recorded
    ++table->total_node_count;
    if (table->node_count < IREE_TASK_TOPOLOGY_MAX_NODES) {
      ctx.node_index = (uint8_t)table->node_count;
      iree_sysfs_try_parse_cpu_list(cpulist, iree_sysfs_numa_commit_cb, &ctx);
      table->node_ids[table->node_count++] = node_id;
    }  // else counted but not storable; callers reject on the total
  }
}

static bool iree_sysfs_numa_online_cb(uint32_t start_id, uint32_t end_id,
                                      void* user_data) {
  iree_sysfs_numa_table_t* table = (iree_sysfs_numa_table_t*)user_data;
  for (uint32_t id = start_id; id < end_id; ++id) {
    iree_sysfs_numa_table_ingest_node(table, id);
  }
  return true;
}

// Builds the NUMA table from /sys/devices/system/node/.
//
// "Node" is the memory-locality domain that --task_topology_nodes selects on
// and that the affinity group hint feeds to set_mempolicy.
// When the hierarchy is unavailable callers fall back to physical_package_id.
static void iree_sysfs_numa_table_initialize(iree_sysfs_numa_table_t* table) {
  memset(table, 0, sizeof(*table));
  memset(table->cpu_node_index, IREE_TASK_TOPOLOGY_SYSFS_NODE_UNKNOWN,
         sizeof(table->cpu_node_index));
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(&builder, "%s/node/online",
                                                  iree_sysfs_get_root_path()));
  char online[IREE_TASK_TOPOLOGY_SYSFS_NODE_LIST_BUFFER_SIZE];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_sysfs_read_small_file(path, online, sizeof(online), &length);
  // Truncating a node list would silently report the wrong nodes, so a file
  // that exists but does not fit is fatal. Absent or unreadable falls back.
  if (iree_status_is_out_of_range(status)) {
    IREE_CHECK_OK(status);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;  // no node hierarchy; table stays invalid
  }
  iree_sysfs_try_parse_cpu_list(iree_make_string_view(online, length),
                                iree_sysfs_numa_online_cb, table);
  table->valid = table->node_count > 0;
}

// Resolves the kernel NUMA node id for a logical processor from |table|.
// Returns false when |table| is unusable or does not cover |processor|.
//
// Never substitutes a package id: callers that bind memory must only ever see
// ids that came from the node hierarchy.
static bool iree_sysfs_try_query_numa_node(const iree_sysfs_numa_table_t* table,
                                           uint32_t processor,
                                           uint32_t* out_node_id) {
  *out_node_id = UINT32_MAX;
  if (!table || !table->valid) return false;
  if (processor >= IREE_TASK_TOPOLOGY_SYSFS_MAX_CPU_ID) return false;
  const uint8_t index = table->cpu_node_index[processor];
  if (index == IREE_TASK_TOPOLOGY_SYSFS_NODE_UNKNOWN ||
      index >= table->node_count) {
    return false;
  }
  *out_node_id = table->node_ids[index];
  return true;
}

// Reads the L2 cluster id (topology/cluster_id) for a logical processor. This
// is the sysfs meaning of "cluster" (see topology.h): the L2 cache module.
// Returns false if unavailable.
static bool iree_sysfs_try_query_cluster_id(uint32_t processor,
                                            uint32_t* out_cluster_id) {
  *out_cluster_id = UINT32_MAX;
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/cpu/cpu%u/topology/cluster_id", iree_sysfs_get_root_path(),
      processor));
  return iree_sysfs_try_read_uint32(path, out_cluster_id);
}

// Assigns the NUMA node hint carried in iree_thread_affinity_t::group for
// |processor|, leaving the initialized default (node 0) when no usable node is
// available, and returns the kernel NUMA node id for the processor (see
// iree_task_topology_t::numa_node_id) or IREE_TASK_TOPOLOGY_NODE_ID_ANY.
//
// The hint field is only 8 bits wide (see iree_thread_affinity_t), so this
// rejects both the kernel's UINT16_MAX/UINT32_MAX "unavailable" sentinels and
// any legitimate node id past 255: truncation would hand set_mempolicy a bogus
// node (65535 -> 255) instead of leaving the hint unset.
static iree_task_topology_node_id_t iree_sysfs_apply_numa_affinity_group(
    const iree_sysfs_numa_table_t* numa_table, uint32_t processor,
    iree_thread_affinity_t* out_affinity) {
  uint32_t node_id = 0;
  if (!iree_sysfs_try_query_numa_node(numa_table, processor, &node_id) ||
      !iree_sysfs_is_valid_node(node_id)) {
    return IREE_TASK_TOPOLOGY_NODE_ID_ANY;
  }
  if (node_id <= UINT8_MAX) out_affinity->group = node_id;
  return (iree_task_topology_node_id_t)node_id;
}

//===----------------------------------------------------------------------===//
// Physical core identity
//===----------------------------------------------------------------------===//

// Tracks the lowest CPU id named by a cpulist.
typedef struct {
  uint32_t min_cpu;
  bool any;
} iree_sysfs_min_cpu_t;
static bool iree_sysfs_min_cpu_cb(uint32_t start_cpu, uint32_t end_cpu,
                                  void* user_data) {
  iree_sysfs_min_cpu_t* ctx = (iree_sysfs_min_cpu_t*)user_data;
  if (end_cpu > start_cpu && (!ctx->any || start_cpu < ctx->min_cpu)) {
    ctx->min_cpu = start_cpu;
    ctx->any = true;
  }
  return true;  // continue
}

// Set on keys derived from the sibling list so they can never collide with the
// package/core_id-derived fallback encoding below.
#define IREE_TASK_TOPOLOGY_SYSFS_CORE_KEY_SIBLING_TAG (1ull << 63)

// Derives a core key from the SMT sibling list. Every logical CPU of a physical
// core names the same set, so the lowest member is a machine-wide unique core
// identity by construction. core_cpus_list is the modern name (kernel 5.3+);
// thread_siblings_list is the older one.
static bool iree_sysfs_try_query_core_siblings_key(uint32_t processor,
                                                   uint64_t* out_key) {
  static const char* const kSiblingFiles[] = {"core_cpus_list",
                                              "thread_siblings_list"};
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  char buffer[256];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kSiblingFiles); ++i) {
    iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
    IREE_CHECK_OK(iree_string_builder_append_format(
        &builder, "%s/cpu/cpu%u/topology/%s", iree_sysfs_get_root_path(),
        processor, kSiblingFiles[i]));
    iree_host_size_t length = 0;
    if (!iree_sysfs_try_read_small_file(path, buffer, sizeof(buffer),
                                        &length)) {
      continue;
    }
    iree_sysfs_min_cpu_t ctx = {.min_cpu = 0, .any = false};
    if (iree_sysfs_try_parse_cpu_list(iree_make_string_view(buffer, length),
                                      iree_sysfs_min_cpu_cb, &ctx) &&
        ctx.any) {
      *out_key =
          IREE_TASK_TOPOLOGY_SYSFS_CORE_KEY_SIBLING_TAG | (uint64_t)ctx.min_cpu;
      return true;
    }
  }
  return false;
}

// Returns a key uniquely identifying the physical core |processor| belongs to,
// across the WHOLE machine. Returns false if the core cannot be identified.
//
// topology/core_id alone is not such a key: the kernel documents it as
// "architecture and platform dependent" and on x86 it is only unique within a
// package, so core 0 of socket 0 collides with core 0 of socket 1.
// Deduplicating cores on it discards every core of the second socket whenever
// the scan is not already restricted to a single node -- exactly what happens
// under IREE_TASK_TOPOLOGY_NODE_ID_ANY.
static bool iree_sysfs_try_query_core_key(uint32_t processor,
                                          uint64_t* out_key) {
  *out_key = 0;
  if (iree_sysfs_try_query_core_siblings_key(processor, out_key)) return true;
  uint32_t core_id = 0;
  if (!iree_sysfs_try_query_core_id(processor, &core_id)) return false;
  // No sibling list exposed: qualify core_id with the package so cross-socket
  // collisions are still avoided. Both halves are 32-bit so the pair fits.
  uint32_t package_id = 0;
  if (iree_sysfs_try_query_package_id(processor, &package_id) &&
      iree_sysfs_is_valid_node(package_id)) {
    *out_key = ((uint64_t)package_id << 32) | (uint64_t)core_id;
  } else {
    *out_key = (uint64_t)core_id;
  }
  return true;
}

// Reads the CPU capacity for a specific logical processor.
// Used for ARM big.LITTLE detection and performance level classification.
// Returns 0 if not available (x86 systems or older kernels).
static uint32_t iree_sysfs_query_cpu_capacity(uint32_t processor) {
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(
      iree_string_builder_append_format(&builder, "%s/cpu/cpu%u/cpu_capacity",
                                        iree_sysfs_get_root_path(), processor));
  uint32_t capacity = 0;
  iree_sysfs_try_read_uint32(path, &capacity);
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
// Returns false if the cache index doesn't exist or can't be parsed.
static bool iree_sysfs_try_query_cache_level(
    uint32_t processor, uint32_t cache_index,
    iree_sysfs_cache_info_t* out_cache) {
  // Read cache type (Data, Instruction, or Unified).
  // If this fails the cache index doesn't exist.
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/cpu/cpu%u/cache/index%u/type", iree_sysfs_get_root_path(),
      processor, cache_index));
  char buffer[64];
  iree_host_size_t length = 0;
  if (!iree_sysfs_try_read_small_file(path, buffer, sizeof(buffer), &length)) {
    return false;
  }

  iree_string_view_t type_str =
      iree_string_view_trim(iree_make_string_view(buffer, length));
  out_cache->is_data_cache =
      iree_string_view_starts_with(type_str, IREE_SV("Data")) ||
      iree_string_view_starts_with(type_str, IREE_SV("Unified"));

  // Read cache level (optional - ignore failures).
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/cpu/cpu%u/cache/index%u/level", iree_sysfs_get_root_path(),
      processor, cache_index));
  uint32_t level = 0;
  iree_sysfs_try_read_uint32(path, &level);
  out_cache->level = level;

  // Read cache size (optional - ignore failures).
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/cpu/cpu%u/cache/index%u/size", iree_sysfs_get_root_path(),
      processor, cache_index));
  uint64_t size = 0;
  iree_sysfs_try_read_size(path, &size);
  out_cache->size = size;

  return true;
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
    if (!iree_sysfs_try_query_cache_level(processor, cache_index, &cache)) {
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

void iree_task_topology_query_default_caches(
    iree_task_topology_caches_t* out_caches) {
  memset(out_caches, 0, sizeof(*out_caches));
  // Query cache sizes for the CPU we happen to be running on. Since this is
  // used for unpinned groups we can't know which CPU they'll end up on, but
  // the current CPU's cache sizes are a representative sample of the hardware.
  iree_task_topology_group_t temp_group;
  iree_task_topology_group_initialize(0, &temp_group);
  iree_sysfs_populate_cache_info(iree_sysfs_query_current_cpu(), &temp_group);
  *out_caches = temp_group.caches;
}

iree_status_t iree_task_topology_set_snapshot_path(const char* path) {
  if (path) {
    char probe_path[IREE_SYSFS_MAX_PATH];
    iree_string_builder_t builder;
    iree_string_builder_initialize_with_storage(probe_path, sizeof(probe_path),
                                                &builder);
    IREE_CHECK_OK(
        iree_string_builder_append_format(&builder, "%s/cpu/present", path));
    char probe_buffer[256];
    iree_host_size_t probe_length = 0;
    if (!iree_sysfs_try_read_small_file(probe_path, probe_buffer,
                                        sizeof(probe_buffer), &probe_length)) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "'%s' has no readable cpu/present; expected a "
                              "directory /sys/devices/system",
                              path);
    }
  }
  iree_sysfs_set_root_path(path);
  return iree_ok_status();
}

iree_host_size_t iree_task_topology_format_processor_debug_ids(
    uint32_t processor, iree_host_size_t buffer_capacity, char* buffer) {
  if (!buffer || buffer_capacity == 0) return 0;
  uint32_t cluster_id = 0;
  const bool have_cluster =
      iree_sysfs_try_query_cluster_id(processor, &cluster_id);
  uint32_t package_id = 0;
  const bool have_package =
      iree_sysfs_try_query_package_id(processor, &package_id);
  int length = 0;
  if (have_cluster && have_package) {
    length = iree_snprintf(buffer, buffer_capacity,
                           "cluster_id=%u, physical_package_id=%u", cluster_id,
                           package_id);
  } else if (have_cluster) {
    length =
        iree_snprintf(buffer, buffer_capacity,
                      "cluster_id=%u, physical_package_id=n/a", cluster_id);
  } else if (have_package) {
    length =
        iree_snprintf(buffer, buffer_capacity,
                      "cluster_id=n/a, physical_package_id=%u", package_id);
  } else {
    return 0;
  }
  return length > 0 ? (iree_host_size_t)length : 0;
}

iree_host_size_t iree_task_topology_query_node_ids(
    iree_host_size_t capacity, iree_task_topology_node_id_t* out_ids) {
  iree_sysfs_numa_table_t numa_table;
  iree_sysfs_numa_table_initialize(&numa_table);
  if (!numa_table.valid) {
    // A kernel built without CONFIG_NUMA exposes no node hierarchy: the whole
    // machine is one node.
    if (out_ids && capacity >= 1) out_ids[0] = 0;
    return 1;
  }

  for (iree_host_size_t i = 0; i < numa_table.node_count && i < capacity; ++i) {
    if (out_ids) out_ids[i] = numa_table.node_ids[i];
  }
  return numa_table.total_node_count;
}

iree_task_topology_node_id_t iree_task_topology_query_current_node(void) {
  // getcpu reports the calling thread's CPU *and* NUMA node in a single
  // syscall, so the common case needs no sysfs walk at all. It only answers for
  // the live host, hence the fallback below when the root is redirected.
  uint32_t current_cpu = 0;
  uint32_t current_node = 0;
  if (iree_sysfs_try_query_current_cpu_and_node(&current_cpu, &current_node)) {
    return current_node;
  }

  iree_sysfs_numa_table_t numa_table;
  iree_sysfs_numa_table_initialize(&numa_table);
  uint32_t node_id = 0;
  if (iree_sysfs_try_query_numa_node(&numa_table, current_cpu, &node_id) &&
      iree_sysfs_is_valid_node(node_id)) {
    return node_id;
  }
  return 0;  // Fallback to node 0.
}

//===----------------------------------------------------------------------===//
// Constructive sharing mask utilities
//===----------------------------------------------------------------------===//

// Context for building a topology group mask directly from a CPU list.
// For each CPU range in the list, we scan the topology's groups to find which
// ones have a processor_index in the range, and set their bit in group_mask.
// This avoids the intermediate cpu_set_t (limited to CPU_SETSIZE=1024) and
// works for arbitrary processor IDs.
typedef struct {
  const iree_task_topology_t* topology;
  iree_task_topology_group_mask_t group_mask;
} iree_sysfs_sharing_context_t;

// Callback for iree_sysfs_parse_cpu_list that maps CPU ranges to group indices.
// O(ranges_in_list x group_count) per group — both are small.
static bool iree_sysfs_accumulate_sharing_groups(uint32_t start_cpu,
                                                 uint32_t end_cpu,
                                                 void* user_data) {
  iree_sysfs_sharing_context_t* ctx = (iree_sysfs_sharing_context_t*)user_data;
  for (iree_host_size_t i = 0; i < ctx->topology->group_count; ++i) {
    uint32_t processor = ctx->topology->groups[i].processor_index;
    if (processor >= start_cpu && processor < end_cpu) {
      iree_task_affinity_set_set_index(&ctx->group_mask,
                                       ctx->topology->groups[i].group_index);
    }
  }
  return true;  // Continue enumeration.
}

// Reads shared_cpu_list for a given cache index and builds a group mask
// directly from the topology (no intermediate cpu_set_t).
// Returns true if successful, false if the file doesn't exist or can't be
// parsed.
static bool iree_sysfs_read_cache_shared_cpu_list(
    uint32_t processor, uint32_t cache_index,
    const iree_task_topology_t* topology,
    iree_task_topology_group_mask_t* out_group_mask) {
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/cpu/cpu%u/cache/index%u/shared_cpu_list",
      iree_sysfs_get_root_path(), processor, cache_index));

  char buffer[256];
  iree_host_size_t length = 0;
  if (!iree_sysfs_try_read_small_file(path, buffer, sizeof(buffer), &length)) {
    return false;
  }

  // Parse CPU list directly into group mask.
  iree_sysfs_sharing_context_t ctx = {
      .topology = topology,
      .group_mask = iree_task_affinity_set_empty(),
  };
  const bool valid =
      iree_sysfs_try_parse_cpu_list(iree_make_string_view(buffer, length),
                                    iree_sysfs_accumulate_sharing_groups, &ctx);
  *out_group_mask = ctx.group_mask;
  return valid;
}

// Finds the best cache level for constructive sharing and returns the group
// mask directly. Prefers L3 Data/Unified, falls back to L2 Data/Unified.
// Returns true if a mask was found, false otherwise.
static bool iree_sysfs_find_sharing_cache_mask(
    uint32_t processor, const iree_task_topology_t* topology,
    iree_task_topology_group_mask_t* out_group_mask) {
  iree_task_topology_group_mask_t l3_mask = iree_task_affinity_set_empty();
  iree_task_topology_group_mask_t l2_mask = iree_task_affinity_set_empty();
  bool found_l3 = false;
  bool found_l2 = false;

  // Scan cache indices looking for L3 (preferred) and L2 (fallback).
  for (uint32_t cache_index = 0; cache_index < IREE_SYSFS_MAX_CACHE_INDICES;
       ++cache_index) {
    iree_sysfs_cache_info_t cache = {0};
    if (!iree_sysfs_try_query_cache_level(processor, cache_index, &cache)) {
      break;  // No more cache levels.
    }

    // Only consider Data or Unified caches.
    if (!cache.is_data_cache) {
      continue;
    }

    iree_task_topology_group_mask_t shared_mask;
    if (iree_sysfs_read_cache_shared_cpu_list(processor, cache_index, topology,
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
    *out_group_mask = l3_mask;
    return true;
  } else if (found_l2) {
    *out_group_mask = l2_mask;
    return true;
  }
  return false;
}

// Builds constructive sharing masks based on cache sharing.
// We parse shared_cpu_list from cache/index*/shared_cpu_list to determine
// which processors share cache levels. We prefer L3 cache sharing, falling
// back to L2 if L3 is not available.
//
// The group mask is built directly from the CPU list without an intermediate
// cpu_set_t, so there is no limit on processor IDs (unlike glibc's
// CPU_SETSIZE=1024 which overflows on machines with >1024 logical CPUs).
iree_status_t iree_task_topology_fixup_constructive_sharing_masks(
    iree_task_topology_t* topology) {
  for (iree_host_size_t i = 0; i < topology->group_count; ++i) {
    iree_task_topology_group_t* group = &topology->groups[i];

    // Find groups that share L3 (or L2 as fallback) cache with this group.
    iree_task_topology_group_mask_t group_mask = iree_task_affinity_set_empty();
    iree_sysfs_find_sharing_cache_mask(group->processor_index, topology,
                                       &group_mask);

    group->constructive_sharing_mask = group_mask;
  }

  return iree_ok_status();
}

iree_status_t iree_task_topology_initialize_from_logical_cpu_set(
    iree_host_size_t cpu_count, const uint32_t* cpu_ids,
    iree_task_topology_t* out_topology) {
  // Validate input.
  if (cpu_count > IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many CPUs specified (%" PRIhsz
                            " provided for a max capacity of %d)",
                            cpu_count, IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT);
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

  // Built once for all groups; see the NUMA node table section.
  iree_sysfs_numa_table_t numa_table;
  iree_sysfs_numa_table_initialize(&numa_table);

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

    // Query NUMA node for the affinity group hint.
    iree_sysfs_apply_numa_affinity_group(&numa_table, cpu_ids[i],
                                         &group->ideal_thread_affinity);
  }

  iree_status_t status =
      iree_task_topology_fixup_constructive_sharing_masks(out_topology);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Cache domain enumeration
//===----------------------------------------------------------------------===//

// Context for building a processor bitmask from a CPU list.
// Used by the cache domain enumeration path which needs cpu_set_t for domain
// grouping. Note: cpu_set_t is limited to CPU_SETSIZE (glibc 1024) — this is
// acceptable for domain enumeration since IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT
// bounds the number of cores we enumerate, but processor IDs themselves could
// exceed 1024 on large machines. The constructive sharing mask path above
// avoids cpu_set_t entirely.
typedef struct {
  cpu_set_t processor_mask;
} iree_sysfs_processor_mask_context_t;

// Callback to accumulate processor IDs into a cpu_set_t bitmask.
static bool iree_sysfs_accumulate_processor_mask(uint32_t start_cpu,
                                                 uint32_t end_cpu,
                                                 void* user_data) {
  iree_sysfs_processor_mask_context_t* ctx =
      (iree_sysfs_processor_mask_context_t*)user_data;
  for (uint32_t cpu = start_cpu; cpu < end_cpu; ++cpu) {
    if (cpu < CPU_SETSIZE) {
      CPU_SET(cpu, &ctx->processor_mask);
    }
  }
  return true;  // Continue enumeration.
}

// Reads shared_cpu_list for a given cache index into a processor bitmask.
// Returns true if successful, false if the file doesn't exist or can't be
// parsed.
static bool iree_sysfs_read_cache_shared_processor_mask(uint32_t processor,
                                                        uint32_t cache_index,
                                                        cpu_set_t* out_mask) {
  char path[IREE_SYSFS_MAX_PATH];
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(path, sizeof(path), &builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &builder, "%s/cpu/cpu%u/cache/index%u/shared_cpu_list",
      iree_sysfs_get_root_path(), processor, cache_index));

  char buffer[256];
  iree_host_size_t length = 0;
  if (!iree_sysfs_try_read_small_file(path, buffer, sizeof(buffer), &length)) {
    return false;
  }

  // Parse CPU list into bitmask.
  iree_sysfs_processor_mask_context_t ctx;
  CPU_ZERO(&ctx.processor_mask);
  const bool valid_bitmask =
      iree_sysfs_try_parse_cpu_list(iree_make_string_view(buffer, length),
                                    iree_sysfs_accumulate_processor_mask, &ctx);
  *out_mask = ctx.processor_mask;
  return valid_bitmask;
}

// Finds the best cache level for domain grouping and returns a processor mask.
// Prefers L3 Data/Unified, falls back to L2 Data/Unified.
// Returns true if a mask was found, false otherwise.
static bool iree_sysfs_find_sharing_processor_mask(uint32_t processor,
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
    if (!iree_sysfs_try_query_cache_level(processor, cache_index, &cache)) {
      break;  // No more cache levels.
    }

    // Only consider Data or Unified caches.
    if (!cache.is_data_cache) {
      continue;
    }

    cpu_set_t shared_mask;
    if (iree_sysfs_read_cache_shared_processor_mask(processor, cache_index,
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
        iree_sysfs_find_sharing_processor_mask(processor, &sharing_mask);

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

  max_core_count = iree_min(max_core_count, IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT);

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

  // Sysfs describes the host machine, not necessarily the processor set this
  // process is allowed to run on. Constrain topology discovery with the current
  // affinity mask so cgroups, cpusets, taskset, and qemu-user test runners do
  // not create worker groups that can never execute.
  cpu_set_t current_affinity;
  const cpu_set_t* current_affinity_ptr =
      iree_sysfs_query_current_affinity(&current_affinity) ? &current_affinity
                                                           : NULL;

  // Built once for the whole scan; see the NUMA node table section.
  iree_sysfs_numa_table_t numa_table;
  iree_sysfs_numa_table_initialize(&numa_table);

  // Find unique cores by enumerating processors and grouping by core key. We
  // build a simple map of core -> first processor in that core, keeping the
  // keys alongside so the duplicate check does not re-read sysfs per pair. The
  // key is machine-wide unique (see iree_sysfs_try_query_core_key); a raw
  // core_id is not, and would collapse the second socket into the first.
  uint32_t core_map[IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT];
  uint64_t core_keys[IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT];
  iree_host_size_t core_count = 0;
  for (uint32_t cpu = 0; cpu < processor_count && core_count < max_core_count;
       ++cpu) {
    if (!iree_sysfs_is_processor_available(cpu, current_affinity_ptr)) {
      continue;
    }

    uint64_t core_key = 0;
    if (!iree_sysfs_try_query_core_key(cpu, &core_key)) {
      continue;  // Skip CPUs we can't query.
    }

    if (node_id != IREE_TASK_TOPOLOGY_NODE_ID_ANY) {
      // Only filter when node info is valid and doesn't match. When invalid we
      // skip filtering to avoid removing all cores on systems that don't expose
      // node ids.
      // NOTE: a CPU whose node cannot be determined is therefore kept for
      // EVERY node, so --task_topology_nodes=numa double-books such CPUs across
      // topologies rather than leaving those cores unused.
      uint32_t cpu_node_id = 0;
      if (iree_sysfs_try_query_numa_node(&numa_table, cpu, &cpu_node_id) &&
          iree_sysfs_is_valid_node(cpu_node_id) &&
          cpu_node_id != (uint32_t)node_id) {
        continue;  // Wrong node.
      }
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

    // Check if we've already seen this core (compared against the cached keys
    // so this stays O(cores) in memory rather than re-reading sysfs per pair).
    bool core_seen = false;
    for (iree_host_size_t i = 0; i < core_count; ++i) {
      if (core_keys[i] == core_key) {
        core_seen = true;
        break;
      }
    }
    if (!core_seen) {
      // First processor in this core.
      core_keys[core_count] = core_key;
      core_map[core_count++] = cpu;
    }
  }

  // Reorder cores according to distribution strategy across cache domains.
  // COMPACT fills cache domains sequentially, SCATTER distributes round-robin.
  // For COMPACT or single-domain systems, use cores in original order.
  if (core_count > 1 &&
      distribution == IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER) {
    // Enumerate cache domains from the cores we found.
    iree_sysfs_cache_domain_t domains[IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT];
    const iree_host_size_t domain_count = iree_sysfs_enumerate_cache_domains(
        core_count, core_map, domains, IREE_ARRAYSIZE(domains));
    if (domain_count > 1) {
      // SCATTER: Distribute cores evenly across domains using round-robin.
      // This maximizes memory bandwidth by utilizing multiple controllers.
      uint32_t new_core_map[IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT];
      iree_host_size_t new_core_count = 0;

      // Track next CPU to check for each domain.
      int domain_next_cpu[IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT];
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

    const iree_task_topology_node_id_t core_node_id =
        iree_sysfs_apply_numa_affinity_group(&numa_table, processor,
                                             &group->ideal_thread_affinity);
    // Fold the selected cores down to the one node they all share, or ANY if
    // they disagree.
    if (i == 0) {
      out_topology->numa_node_id = core_node_id;
    } else if (out_topology->numa_node_id != core_node_id) {
      out_topology->numa_node_id = IREE_TASK_TOPOLOGY_NODE_ID_ANY;
    }
  }

  iree_status_t status =
      iree_task_topology_fixup_constructive_sharing_masks(out_topology);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // !IREE_TASK_USE_CPUINFO && IREE_PLATFORM_LINUX &&
        // !IREE_PLATFORM_WASM
