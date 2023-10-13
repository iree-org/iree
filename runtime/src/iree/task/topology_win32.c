// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/task/topology.h"

#if defined(IREE_PLATFORM_WINDOWS)

//===----------------------------------------------------------------------===//
// NUMA queries
//===----------------------------------------------------------------------===//

iree_host_size_t iree_task_topology_query_node_count(void) {
  ULONG highest_number = 0;
  return GetNumaHighestNodeNumber(&highest_number)
             ? (iree_host_size_t)(highest_number + 1)
             : 1;
}

iree_task_topology_node_id_t iree_task_topology_query_current_node(void) {
  PROCESSOR_NUMBER processor_number;
  GetCurrentProcessorNumberEx(&processor_number);
  USHORT node_number = 0;
  GetNumaProcessorNodeEx(&processor_number, &node_number);
  return (iree_task_topology_node_id_t)node_number;
}

//===----------------------------------------------------------------------===//
// Topology initialization helpers
//===----------------------------------------------------------------------===//

static inline int iree_task_count_trailing_zeros_kaffinity(
    KAFFINITY affinity_mask) {
#if defined(_WIN64)
  return iree_math_count_trailing_zeros_u64(affinity_mask);
#else
  return iree_math_count_trailing_zeros_u32(affinity_mask);
#endif  // _WIN64
}

static inline int iree_task_count_kaffinity_bits(KAFFINITY affinity_mask) {
#if defined(_WIN64)
  return iree_math_count_ones_u64(affinity_mask);
#else
  return iree_math_count_ones_u32(affinity_mask);
#endif  // _WIN64
}

// Sets |out_affinity| to be pinned to |processor|.
static void iree_task_topology_set_affinity_from_processor(
    const PROCESSOR_RELATIONSHIP* processor,
    iree_thread_affinity_t* out_affinity) {
  memset(out_affinity, 0, sizeof(*out_affinity));
  out_affinity->specified = 1;

  // Special bit to indicate that (if required) we want the entire core.
  out_affinity->smt = (processor->Flags & LTP_PC_SMT) == LTP_PC_SMT;

  out_affinity->group = processor->GroupMask[0].Group;
  out_affinity->id =
      iree_task_count_trailing_zeros_kaffinity(processor->GroupMask[0].Mask);
}

// Uses |group_mask| to assign constructive sharing masks to all topology groups
// that constructively share some level of the cache hierarchy.
static void iree_task_topology_assign_constructive_sharing(
    iree_task_topology_t* topology, GROUP_AFFINITY group_mask) {
  // NOTE: O(n^2) but should always be small (~number of NUMA nodes).
  for (iree_host_size_t group_i = 0; group_i < topology->group_count;
       ++group_i) {
    iree_task_topology_group_t* group = &topology->groups[group_i];
    if (group->ideal_thread_affinity.group == group_mask.Group &&
        (group_mask.Mask & (1ull << group->ideal_thread_affinity.id))) {
      for (iree_host_size_t group_j = 0; group_j < topology->group_count;
           ++group_j) {
        iree_task_topology_group_t* other = &topology->groups[group_j];
        if (other->ideal_thread_affinity.group == group_mask.Group &&
            (group_mask.Mask & (1ull << other->ideal_thread_affinity.id))) {
          group->constructive_sharing_mask |= 1ull << group_j;
        }
      }
    }
  }
}

// Assigns constructive sharing masks to each topology group. These indicate
// which other topology groups share L3 caches (if any).
static void
iree_task_topology_fixup_constructive_sharing_masks_from_relationships(
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* relationships,
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* relationships_end,
    iree_task_topology_t* topology) {
  for (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p = relationships;
       p < relationships_end;
       p = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)p + p->Size)) {
    if (p->Relationship == RelationCache) {
      if (p->Cache.Level == 3 &&
          (p->Cache.Type == CacheUnified || p->Cache.Type == CacheData)) {
        if (p->Cache.GroupCount == 0) {
          iree_task_topology_assign_constructive_sharing(topology,
                                                         p->Cache.GroupMask);
        } else {
          for (WORD i = 0; i < p->Cache.GroupCount; ++i) {
            iree_task_topology_assign_constructive_sharing(
                topology, p->Cache.GroupMasks[i]);
          }
        }
      }
    }
  }
}

iree_status_t iree_task_topology_fixup_constructive_sharing_masks(
    iree_task_topology_t* topology) {
  // Query the total size required for just cache information and allocate
  // storage for it on the stack - it's generally just a few KB.
  DWORD cache_relationships_size = 0;
  if (!GetLogicalProcessorInformationEx(RelationCache, NULL,
                                        &cache_relationships_size) &&
      GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "failed to query logical processor information size (%08X)",
        GetLastError());
  }
  if (cache_relationships_size > 64 * 1024) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "logical processor information size overflow (got "
                            "%u which is large for a stack alloc)",
                            cache_relationships_size);
  }
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* cache_relationships =
      (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)iree_alloca(
          cache_relationships_size);

  // Query again to populate the storage with cache relationship information.
  if (!GetLogicalProcessorInformationEx(RelationCache, cache_relationships,
                                        &cache_relationships_size)) {
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "failed to query logical processor information (%08X)", GetLastError());
  }
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* cache_relationships_end =
      (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)
                                                     cache_relationships +
                                                 cache_relationships_size);

  // Perform the assignment.
  iree_task_topology_fixup_constructive_sharing_masks_from_relationships(
      cache_relationships, cache_relationships_end, topology);
  return iree_ok_status();
}

iree_status_t iree_task_topology_initialize_from_logical_cpu_set(
    iree_host_size_t cpu_count, const uint32_t* cpu_ids,
    iree_task_topology_t* out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)cpu_count);

  iree_task_topology_initialize(out_topology);

  // Query the total size required for all information and allocate storage for
  // it on the stack - it's generally just a few KB.
  DWORD all_relationships_size = 0;
  if (!GetLogicalProcessorInformationEx(RelationAll, NULL,
                                        &all_relationships_size) &&
      GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "failed to query logical processor information size (%08X)",
        GetLastError());
  }
  if (all_relationships_size > 64 * 1024) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "logical processor information size overflow (got "
                            "%u which is large for a stack alloc)",
                            all_relationships_size);
  }
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* all_relationships =
      (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)iree_alloca(
          all_relationships_size);

  // Query again to populate the storage with all relationship information.
  if (!GetLogicalProcessorInformationEx(RelationAll, all_relationships,
                                        &all_relationships_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "failed to query logical processor information (%08X)", GetLastError());
  }
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* all_relationships_end =
      (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)all_relationships +
                                                 all_relationships_size);

  // Count up the total number of logical processors (bits in each core group).
  uint32_t total_processor_count = 0;
  for (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p = all_relationships;
       p < all_relationships_end;
       p = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)p + p->Size)) {
    if (p->Relationship == RelationProcessorCore) {
      assert(p->Processor.GroupCount == 1);
      total_processor_count +=
          iree_task_count_kaffinity_bits(p->Processor.GroupMask[0].Mask);
    }
  }

  // Validate the CPU IDs provided and build a lookup table of processors we
  // have selected. This could be a bitmap but it's not worth the code today.
  uint8_t* included_processors =
      (uint8_t*)iree_alloca(total_processor_count * sizeof(uint8_t));
  memset(included_processors, 0, total_processor_count * sizeof(uint8_t));
  for (iree_host_size_t i = 0; i < cpu_count; ++i) {
    if (cpu_ids[i] >= total_processor_count) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "cpu_ids[%" PRIhsz
          "] %u out of bounds, only %u logical processors available",
          i, cpu_ids[i], total_processor_count);
    }
    included_processors[cpu_ids[i]] = 1;
  }

  // Build an on-stack table for random access into all logical processors.
  // This isn't strictly required but makes it easier to walk the CPU table.
  PROCESSOR_RELATIONSHIP** all_processors =
      iree_alloca(sizeof(PROCESSOR_RELATIONSHIP*) * total_processor_count);
  iree_host_size_t global_processor_count = 0;
  for (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p = all_relationships;
       p < all_relationships_end;
       p = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)p + p->Size)) {
    if (p->Relationship != RelationProcessorCore) continue;
    assert(p->Processor.GroupCount == 1);
    KAFFINITY mask = p->Processor.GroupMask[0].Mask;
    int group_offset = 0;
    while (mask) {
      int bit_offset = iree_task_count_trailing_zeros_kaffinity(mask);
      mask = mask >> (bit_offset + 1);
      iree_host_size_t global_processor_index = global_processor_count++;
      if (included_processors[global_processor_index]) {
        // Setup the group for the processor.
        uint8_t group_index = (uint8_t)out_topology->group_count++;
        iree_task_topology_group_t* group = &out_topology->groups[group_index];
        iree_task_topology_group_initialize(group_index, group);
        group->processor_index = (uint32_t)global_processor_index;
        group->constructive_sharing_mask = 0;  // set below

        // Pin group to the processor.
        iree_thread_affinity_t* affinity = &group->ideal_thread_affinity;
        memset(affinity, 0, sizeof(*affinity));
        affinity->specified = 1;
        affinity->smt = (p->Processor.Flags & LTP_PC_SMT) == LTP_PC_SMT;
        affinity->group = p->Processor.GroupMask[0].Group;
        affinity->id = group_offset + bit_offset;
      }
      group_offset += bit_offset + 1;
      if (out_topology->group_count >= cpu_count) break;
    }
    if (out_topology->group_count >= cpu_count) break;
  }

  // Assign constructive sharing masks to each topology group.
  iree_task_topology_fixup_constructive_sharing_masks_from_relationships(
      all_relationships, all_relationships_end, out_topology);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_task_topology_initialize_from_physical_cores(
    iree_task_topology_node_id_t node_id, iree_host_size_t max_core_count,
    iree_task_topology_t* out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)node_id);

  iree_task_topology_initialize(out_topology);

  // Query the total size required for all information and allocate storage for
  // it on the stack - it's generally just a few KB.
  DWORD all_relationships_size = 0;
  if (!GetLogicalProcessorInformationEx(RelationAll, NULL,
                                        &all_relationships_size) &&
      GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "failed to query logical processor information size (%08X)",
        GetLastError());
  }
  if (all_relationships_size > 64 * 1024) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "logical processor information size overflow (got "
                            "%u which is large for a stack alloc)",
                            all_relationships_size);
  }
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* all_relationships =
      (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)iree_alloca(
          all_relationships_size);

  // Query again to populate the storage with all relationship information.
  if (!GetLogicalProcessorInformationEx(RelationAll, all_relationships,
                                        &all_relationships_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "failed to query logical processor information (%08X)", GetLastError());
  }
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* all_relationships_end =
      (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)all_relationships +
                                                 all_relationships_size);

  // Allocate an on-stack table of Windows group information.
  // This will let us easily look up information by PROCESSOR_NUMBER::Group and
  // KAFFINITY.
  iree_host_size_t max_group_count = 0;
  for (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p = all_relationships;
       p < all_relationships_end;
       p = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)p + p->Size)) {
    if (p->Relationship == RelationGroup) {
      max_group_count += p->Group.MaximumGroupCount;
    }
  }
  typedef struct group_info_t {
    // 1 if the group is included in the current filter.
    // If 0 then all processors in the group are to be ignored.
    uint32_t selected : 1;
    // Total number of available cores in the group.
    uint32_t core_count : 8;
  } group_info_t;
  group_info_t* group_table =
      iree_alloca(sizeof(group_info_t) * max_group_count);
  memset(group_table, 0, sizeof(group_info_t) * max_group_count);

  // Filter out groups selected by the NUMA node filter and populate the group
  // table with information.
  for (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p = all_relationships;
       p < all_relationships_end;
       p = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)p + p->Size)) {
    if (p->Relationship == RelationNumaNode ||
        p->Relationship == RelationNumaNodeEx) {
      if (node_id == IREE_TASK_TOPOLOGY_NODE_ID_ANY ||
          p->NumaNode.NodeNumber == node_id) {
        if (p->NumaNode.GroupCount == 0) {
          group_table[p->NumaNode.GroupMask.Group].selected =
              p->NumaNode.GroupMask.Mask != 0;
        } else {
          for (WORD i = 0; i < p->NumaNode.GroupCount; ++i) {
            group_table[p->NumaNode.GroupMasks[i].Group].selected =
                p->NumaNode.GroupMasks[i].Mask != 0;
          }
        }
      }
    }
  }
  iree_host_size_t total_core_count = 0;
  iree_host_size_t selected_core_count = 0;
  for (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p = all_relationships;
       p < all_relationships_end;
       p = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)p + p->Size)) {
    if (p->Relationship == RelationProcessorCore) {
      assert(p->Processor.GroupCount == 1);
      if (group_table[p->Processor.GroupMask[0].Group].selected) {
        ++group_table[p->Processor.GroupMask[0].Group].core_count;
        ++selected_core_count;
      }
      ++total_core_count;
    }
  }
  if (!selected_core_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no processors found with NUMA node ID %u",
                            node_id);
  }

  // Build an on-stack table for random access into all cores.
  PROCESSOR_RELATIONSHIP** all_cores =
      iree_alloca(sizeof(PROCESSOR_RELATIONSHIP*) * total_core_count);
  iree_host_size_t global_core_index = 0;
  for (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p = all_relationships;
       p < all_relationships_end;
       p = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((uintptr_t)p + p->Size)) {
    if (p->Relationship == RelationProcessorCore) {
      assert(p->Processor.GroupCount == 1);
      all_cores[global_core_index++] = &p->Processor;
    }
  }

  // Clamp the total number of cores available to the max provided.
  // This is the number of topology groups we'll create.
  iree_host_size_t used_core_count =
      iree_min(selected_core_count, max_core_count);

  // Check if the current (base) processor is part of the filtered groups.
  // If so we perform rotation to favor cores other than the current one.
  // Were we to just use any core there's a high likelihood that we'd pick the
  // current one and compete for resources.
  PROCESSOR_NUMBER base_processor_number;
  GetCurrentProcessorNumberEx(&base_processor_number);
  group_info_t* base_group = &group_table[base_processor_number.Group];
  iree_host_size_t base_core_index = 0;
  for (iree_host_size_t core_index = 0; core_index < total_core_count;
       ++core_index) {
    if (all_cores[core_index]->GroupMask[0].Mask &
        (1ull << base_processor_number.Number)) {
      base_core_index = core_index;
      break;
    }
  }

  // TODO(benvanik): round up to the next cache aligned chunk of cores instead
  // of wherever the current base_core_index is. Today if there's 0,1,2,3 and
  // 4,5,6,7 and the base_core_index is 2 we'll split things if the user asks
  // for 4 workers: 3 + 4,5,6. Ideally we'd jump up to 4 and return 4,5,6,7
  // instead. How much this matters needs to be measured but it'd at least make
  // sense vs being random as it is now.

  // Initialize all topology groups from the selected cores.
  for (iree_host_size_t used_core_index = 0; used_core_index < used_core_count;
       ++used_core_index) {
    iree_host_size_t adjusted_core_index = used_core_index;
    if (base_group->selected) {
      // Rotate the starting core index by the base core such that we only use
      // the base core if all other available cores are utilized.
      adjusted_core_index =
          (((base_core_index + 1) % total_core_count) + used_core_index) %
          total_core_count;
    }
    uint8_t group_index = (uint8_t)out_topology->group_count++;
    iree_task_topology_group_t* group = &out_topology->groups[group_index];
    iree_task_topology_group_initialize(group_index, group);
    group->processor_index = (uint32_t)adjusted_core_index;
    group->constructive_sharing_mask = 0;  // set below
    iree_task_topology_set_affinity_from_processor(
        all_cores[adjusted_core_index], &group->ideal_thread_affinity);
  }

  // Assign constructive sharing masks to each topology group.
  iree_task_topology_fixup_constructive_sharing_masks_from_relationships(
      all_relationships, all_relationships_end, out_topology);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS
