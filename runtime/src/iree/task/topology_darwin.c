// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "iree/task/topology.h"

#if defined(IREE_PLATFORM_APPLE)

#include <sys/sysctl.h>
#include <sys/types.h>

//===----------------------------------------------------------------------===//
// Platform utilities
//===----------------------------------------------------------------------===//

// Apple really doesn't want to let applications control things and hides nearly
// all query and control of threads on the system. The best we can do here is
// estimate counts and cache information but it'll never be correct and barely
// do what the user intends. Such is life in Apple land. Think Different(tm).
// Unfortunately this lack of APIs means we can't do much of anything besides
// request a QoS level and hope the system puts our workers in the right place.
// This makes reliable benchmarking near impossible and results in users having
// wildly different performance based on the whims of their current scheduler.
//
// Meager documentation:
// https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_system_capabilities

static bool iree_task_sysctlbyname_int32(const char* name, int32_t* out_value) {
  *out_value = 0;
  size_t sizeof_value = sizeof(*out_value);
  return sysctlbyname(name, out_value, &sizeof_value, NULL, 0) == 0;
}

static bool iree_task_sysctlbyname_int64(const char* name, int64_t* out_value) {
  *out_value = 0;
  size_t sizeof_value = sizeof(*out_value);
  return sysctlbyname(name, out_value, &sizeof_value, NULL, 0) == 0;
}

static bool iree_task_sysctlbyname_perflevel_int32(int level, const char* key,
                                                   int32_t* out_value) {
  char name[64];
  sprintf(name, "hw.perflevel%d.%s", level, key);
  return iree_task_sysctlbyname_int32(name, out_value);
}

//===----------------------------------------------------------------------===//
// NUMA queries
//===----------------------------------------------------------------------===//

iree_host_size_t iree_task_topology_query_node_count(void) {
  int32_t packages = 1;
#if !defined(IREE_PLATFORM_IOS)
  if (!iree_task_sysctlbyname_int32("hw.packages", &packages) ||
      packages == 0) {
    packages = 1;  // failed to fetch or invalid value
  }
#endif  // !IREE_PLATFORM_IOS
  return packages;
}

iree_task_topology_node_id_t iree_task_topology_query_current_node(void) {
  // AFAICT there's no way to query the system for this information.
  // AFAICT there's also no dual-package systems? Maybe the M2 Ultra?
  return (iree_task_topology_node_id_t)0;
}

//===----------------------------------------------------------------------===//
// Topology initialization helpers
//===----------------------------------------------------------------------===//

iree_status_t iree_task_topology_fixup_constructive_sharing_masks(
    iree_task_topology_t* topology) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "assignment to individual CPUs is not available on "
                          "Apple platforms due to a lack of APIs");
}

iree_status_t iree_task_topology_initialize_from_logical_cpu_set(
    iree_host_size_t cpu_count, const uint32_t* cpu_ids,
    iree_task_topology_t* out_topology) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "assignment to individual CPUs is not available on "
                          "Apple platforms due to a lack of APIs");
}

typedef struct {
  int32_t physicalcpu_max;
  int32_t logicalcpu_max;
  int32_t l1dcachesize;
  int32_t l2cachesize;
  int32_t l3cachesize;
  int32_t cpusperl2;
  int32_t cpusperl3;
} iree_task_hw_perflevel_t;

#define IREE_TASK_MAX_HW_PERF_LEVELS 2

static iree_task_hw_perflevel_t iree_task_query_hw_perflevel_default(void) {
  iree_task_hw_perflevel_t perflevel = {0};

  iree_task_sysctlbyname_int32("hw.physicalcpu_max",
                               &perflevel.physicalcpu_max);
  iree_task_sysctlbyname_int32("hw.logicalcpu_max", &perflevel.logicalcpu_max);
  iree_task_sysctlbyname_int32("hw.l1dcachesize", &perflevel.l1dcachesize);
  iree_task_sysctlbyname_int32("hw.l2cachesize", &perflevel.l2cachesize);
  iree_task_sysctlbyname_int32("hw.l3cachesize", &perflevel.l3cachesize);

  // cpusperX, with [main memory, l1d, l2, l3, ...]
  size_t sizeof_cacheconfig = 0;
  sysctlbyname("hw.cacheconfig", NULL, &sizeof_cacheconfig, NULL, 0);
  int64_t* cacheconfig = (int64_t*)iree_alloca(sizeof_cacheconfig);
  sysctlbyname("hw.cacheconfig", cacheconfig, &sizeof_cacheconfig, NULL, 0);
  size_t ncacheconfig = sizeof_cacheconfig / sizeof(cacheconfig[0]);
  perflevel.cpusperl2 = ncacheconfig >= 3 ? cacheconfig[2] : 0;
  perflevel.cpusperl3 = ncacheconfig >= 4 ? cacheconfig[3] : 0;

  return perflevel;
}

static iree_task_hw_perflevel_t iree_task_query_hw_perflevel(int level) {
  iree_task_hw_perflevel_t perflevel = {0};
  iree_task_sysctlbyname_perflevel_int32(level, "physicalcpu_max",
                                         &perflevel.physicalcpu_max);
  iree_task_sysctlbyname_perflevel_int32(level, "logicalcpu_max",
                                         &perflevel.logicalcpu_max);
  iree_task_sysctlbyname_perflevel_int32(level, "l1dcachesize",
                                         &perflevel.l1dcachesize);
  iree_task_sysctlbyname_perflevel_int32(level, "l2cachesize",
                                         &perflevel.l2cachesize);
  iree_task_sysctlbyname_perflevel_int32(level, "l3cachesize",
                                         &perflevel.l3cachesize);
  iree_task_sysctlbyname_perflevel_int32(level, "cpusperl2",
                                         &perflevel.cpusperl2);
  iree_task_sysctlbyname_perflevel_int32(level, "cpusperl3",
                                         &perflevel.cpusperl3);
  return perflevel;
}

iree_status_t iree_task_topology_initialize_from_physical_cores(
    iree_task_topology_node_id_t node_id,
    iree_task_topology_performance_level_t performance_level,
    iree_host_size_t max_core_count, iree_task_topology_t* out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)node_id);

  iree_task_topology_initialize(out_topology);

  // Total number of physical cores in the system of all types.
  int32_t total_physicalcpu_max = 0;
  if (!iree_task_sysctlbyname_int32("hw.physicalcpu_max",
                                    &total_physicalcpu_max) ||
      total_physicalcpu_max == 0) {
    total_physicalcpu_max = 1;  // failed to fetch or invalid value
  }

  // Query CPU info per performance level type.
  // NOTE: older systems will report nperflevels=0 and we instead use the
  // default non-perflevel keys.
  // NOTE: when present perflevels[0] is performance.
  int32_t nperflevels = 0;
  iree_task_sysctlbyname_int32("hw.nperflevels", &nperflevels);
  nperflevels = iree_min(nperflevels, IREE_TASK_MAX_HW_PERF_LEVELS);
  iree_task_hw_perflevel_t perflevels[IREE_TASK_MAX_HW_PERF_LEVELS];
  if (nperflevels > 0) {
    // System has multiple perflevels (AMP / asymmetric multiprocessing).
    for (int32_t i = 0; i < nperflevels; ++i) {
      perflevels[i] = iree_task_query_hw_perflevel(i);
    }
  } else {
    // Only one perflevel (homogeneous cores).
    nperflevels = 1;
    perflevels[0] = iree_task_query_hw_perflevel_default();
  }
  int32_t physicalcpu_max = total_physicalcpu_max;
  if (nperflevels > 0) {
    switch (performance_level) {
      default:
      case IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY:
        physicalcpu_max = total_physicalcpu_max;
        break;
      case IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_LOW:
        physicalcpu_max = perflevels[/*efficiency=*/1].physicalcpu_max;
        break;
      case IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_HIGH:
        physicalcpu_max = perflevels[/*performance=*/0].physicalcpu_max;
        break;
    }
  }

  iree_host_size_t core_count = iree_min(physicalcpu_max, max_core_count);
  iree_task_topology_initialize_from_group_count(core_count, out_topology);
  for (iree_host_size_t i = 0; i < out_topology->group_count; ++i) {
    iree_task_topology_group_t* group = &out_topology->groups[i];
    group->processor_index = i;

    // Assign attributes based on the perflevel of the group; we can't pin cores
    // on Apple platforms so instead we just treat the first N groups as
    // perflevel[0], the next as perflevel[1], etc.
    int perflevel = 0;
    if (nperflevels > 1) {
      perflevel = i < perflevels[0].physicalcpu_max ? 0 : 1;
    }
    group->caches.l1_data = perflevels[perflevel].l1dcachesize;
    group->caches.l2_data = perflevels[perflevel].l2cachesize;
    group->caches.l3_data = perflevels[perflevel].l3cachesize;

    // We make stuff up as Apple doesn't want us to have nice things.
    // See iree_thread_affinity_t for more information about how we use the
    // affinity info. Note that we pack "use efficiency cores only" into the SMT
    // bit and use that to force a QoS level that ensures only efficiency cores
    // are used when present. Probably.
    group->ideal_thread_affinity.specified = 1;
    group->ideal_thread_affinity.group = (uint32_t)node_id;
    group->ideal_thread_affinity.id = i;
    switch (performance_level) {
      default:
      case IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY:
        // If heterogeneous then put the first N groups anywhere and the rest on
        // efficiency cores.
        group->ideal_thread_affinity.smt = perflevel > 0 ? 1 : 0;
        break;
      case IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_HIGH:
        // Try to avoid efficiency cores (but no way to do that).
        group->ideal_thread_affinity.smt = 0;
        break;
      case IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_LOW:
        // Force onto efficiency cores.
        group->ideal_thread_affinity.smt = 1;
        break;
    }

    // We don't set any sharing mask here as we have no idea where the groups
    // we be placed by the magical mystical completely unpredictable XNU
    // scheduler. Cool.
    // We could use cpusperl2/l3 to at least know how many groups may share a
    // particular cache but without control it's useless info.
    // group->constructive_sharing_mask = ...;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_APPLE
