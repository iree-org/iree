// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Processor/NUMA affinity description for thread placement.
//
// This is a lightweight value type that can be used without pulling in the
// full threading library (iree_thread_t, pthreads, etc.). Separated from
// thread.h so that subsystems needing only topology descriptions (e.g., async
// proactor pools, NUMA-aware allocators) can depend on :threading without
// requiring :thread and its platform threading dependencies.

#ifndef IREE_BASE_THREADING_AFFINITY_H_
#define IREE_BASE_THREADING_AFFINITY_H_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Specifies the processor affinity for a particular thread.
// Each platform handles this differently (if at all).
//
// macOS/iOS:
//   Only affinity tags are supported; the ID will be used by the kernel to
//   group threads that having matching values together and (hopefully) schedule
//   them on cores that may share some level of the cache hierarchy. The API is
//   effectively just asking nicely and hoping the kernel is on the same
//   wavelength. It rarely is. As a workaround for the lack of specific pinning
//   we use the smt bit of 1 to indicate that _only_ efficiency cores should be
//   used by effectively changing the QoS of the thread to one in the range
//   where only efficiency cores will be used.
//
//   Mapping:
//    group: (unused)
//       id: used for THREAD_AFFINITY_POLICY to request exclusive cores.
//      smt: 1 if only efficiency cores should be used (QOS_CLASS_BACKGROUND).
//
// Linux/Android:
//   sched_setaffinity is used to pin the thread to the core with the given ID.
//   There are, naturally, issues on Android where if the governor has turned
//   off some cores (such as powering down big cores in an ARM big.LITTLE
//   configuration) the affinity request will be dropped on the floor even if
//   the cores are later enabled. This is one of the reasons why we note in
//   iree_thread_request_affinity that requests may need to be made at
//   ¯\_(ツ)_/¯ intervals. In the future we can try to hook into power
//   management infra to see if we can tell when we need to do this.
//
//   Mapping:
//    group: NUMA node passed to set_mempolicy.
//       id: CPU_SET bit indicating which CPU to run on.
//      smt: whether to CPU_SET both the base ID and the subsequent ID.
//
// Windows:
//   Stuff just works. Love it.
//
//   Mapping:
//    group: GROUP_AFFINITY::Group/PROCESSOR_NUMBER::Group.
//       id: GROUP_AFFINITY::Mask bit/PROCESSOR_NUMBER::Number.
//      smt: whether to set both the base ID and the subsequent ID in Mask.
typedef struct iree_thread_affinity_t {
  // When 1 the processor ID will be ignored and the platform will choose any
  // processor associated with the specified group (NUMA node ID).
  uint32_t group_any : 1;
  // Processor group the thread should be assigned to, aka NUMA node, cluster,
  // etc depending on platform. On platforms where the processor ID is unique
  // for the purposes of scheduling (e.g. Linux) this is used for related APIs
  // like mbind/set_mempolicy. If group_any is set and id_assigned is not then
  // any processor associated with the group will be used.
  uint32_t group : 8;

  uint32_t reserved : 23;

  // When 0 the affinity is undefined and the system may place the thread
  // anywhere and migrate it as much as it likes. In practice it may do that
  // even when specified.
  uint32_t id_assigned : 1;
  // Processor ID the thread should be scheduled on. The interpretation and
  // efficacy of this request varies per platform.
  uint32_t id : 30;
  // When 1 and the specified processor ID is part of an SMT set all logical
  // cores in the set should be reserved for the thread to avoid contention.
  uint32_t smt : 1;
} iree_thread_affinity_t;

// Sets |out_thread_affinity| to match with any processor in the system.
IREE_API_EXPORT void iree_thread_affinity_set_any(
    iree_thread_affinity_t* out_thread_affinity);

// Returns true if |thread_affinity| does not specify any particular processor.
static inline bool iree_thread_affinity_is_unspecified(
    iree_thread_affinity_t thread_affinity) {
  return !thread_affinity.group_any && !thread_affinity.id_assigned;
}

// Sets |out_thread_affinity| to match all processors associated with the given
// processor group (aka NUMA node ID). Any processor within the group may be
// selected by the platform.
IREE_API_EXPORT void iree_thread_affinity_set_group_any(
    uint32_t group, iree_thread_affinity_t* out_thread_affinity);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_AFFINITY_H_
