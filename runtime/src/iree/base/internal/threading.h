// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_THREADING_H_
#define IREE_BASE_INTERNAL_THREADING_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// iree_thread_t
//==============================================================================

typedef struct iree_thread_t iree_thread_t;

// Specifies a thread's priority class.
// These translate roughly to the same thing across all platforms, though they
// are just a hint and the schedulers on various platforms may behave very
// differently. When in doubt prefer to write code that works at the extremes
// of the classes.
typedef enum iree_thread_priority_class_e {
  // Lowest possible priority used for background/idle work.
  // Maps to QOS_CLASS_BACKGROUND.
  IREE_THREAD_PRIORITY_CLASS_LOWEST = -2,
  // Low priority work but still something the user expects to complete soon.
  // Maps to QOS_CLASS_UTILITY.
  IREE_THREAD_PRIORITY_CLASS_LOW = -1,
  // Normal/default priority for the system.
  // Maps to QOS_CLASS_DEFAULT.
  IREE_THREAD_PRIORITY_CLASS_NORMAL = 0,
  // High priority work for operations the user is waiting on.
  // Maps to QOS_CLASS_USER_INITIATED.
  IREE_THREAD_PRIORITY_CLASS_HIGH = 1,
  // Highest possible priority used for interactive work.
  // Maps to QOS_CLASS_USER_INTERACTIVE.
  IREE_THREAD_PRIORITY_CLASS_HIGHEST = 2,
} iree_thread_priority_class_t;

// Specifies the processor affinity for a particular thread.
// Each platform handles this differently (if at all).
//
// macOS/iOS:
//   Only affinity tags are supported; the ID will be used by the kernel to
//   group threads that having matching values together and (hopefully) schedule
//   them on cores that may share some level of the cache hierarchy. The API is
//   effectively just asking nicely and hoping the kernel is on the same
//   wavelength.
//
//   Mapping:
//    group: (unused)
//       id: used for THREAD_AFFINITY_POLICY to request exclusive cores.
//      smt: (unused)
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
  // When 0 the affinity is undefined and the system may place the thread
  // anywhere and migrate it as much as it likes. In practice it may do that
  // even when specified.
  uint32_t specified : 1;
  // When 1 and the specified processor is part of an SMT set all logical cores
  // in the set should be reserved for the thread to avoid contention.
  uint32_t smt : 1;
  // Processor group the thread should be assigned to, aka NUMA node, cluster,
  // etc depending on platform. On platforms where the processor ID is unique
  // for the purposes of scheduling (e.g. Linux) this is used for related APIs
  // like mbind/set_mempolicy.
  uint32_t group : 7;
  // Processor ID the thread should be scheduled on. The interpretation and
  // efficacy of this request varies per platform.
  uint32_t id : 23;
} iree_thread_affinity_t;

// Sets |thread_affinity| to match with any processor in the system.
void iree_thread_affinity_set_any(iree_thread_affinity_t* out_thread_affinity);

// Thread creation parameters.
// All are optional and the entire struct can safely be zero-initialized.
typedef struct iree_thread_create_params_t {
  // Developer-visible name for the thread displayed in tooling.
  // May be omitted for the system-default name (usually thread ID).
  iree_string_view_t name;

  // Stack size of the new thread, in bytes. If omitted a platform-defined
  // default system stack size will be used.
  size_t stack_size;

  // Whether to create the thread in a suspended state. The thread will be
  // initialized but not call the entry routine until it is resumed with
  // iree_thread_resume. This can be useful to avoid a thundering herd upon
  // creation of many threads.
  bool create_suspended;

  // Initial priority class.
  // This may be changed later via iree_thread_priority_class_override_begin;
  // see that for more information.
  iree_thread_priority_class_t priority_class;

  // Initial thread affinity.
  // This may be changed later via iree_thread_request_affinity; see that for
  // more information.
  iree_thread_affinity_t initial_affinity;
} iree_thread_create_params_t;

typedef int (*iree_thread_entry_t)(void* entry_arg);

// Creates a new thread and calls |entry| with |entry_arg|.
// |params| can be used to specify additional thread creation parameters but can
// also be zero-initialized to use defaults.
//
// The thread will be created and configured prior to returning from the
// function. If the create_suspended parameter is set the thread will be
// suspended and must be resumed with iree_thread_resume. Otherwise, the thread
// may already be inside of the |entry| function by the time the function
// returns.
//
// |entry_arg| lifetime is not managed and unless the caller is waiting for the
// thread to start must not be stack-allocated.
iree_status_t iree_thread_create(iree_thread_entry_t entry, void* entry_arg,
                                 iree_thread_create_params_t params,
                                 iree_allocator_t allocator,
                                 iree_thread_t** out_thread);

// Retains the given |thread| for the caller.
void iree_thread_retain(iree_thread_t* thread);

// Releases the given |thread| from the caller.
void iree_thread_release(iree_thread_t* thread);

// Returns a platform-defined thread ID for the given |thread|.
uintptr_t iree_thread_id(iree_thread_t* thread);

typedef struct iree_thread_override_t iree_thread_override_t;

// Begins overriding the priority class of the given |thread|.
// The priority of the thread will be the max of the base priority and the
// overridden priority. Callers must pass the returned override token to
// iree_thread_override_end.
//
// This is only a hint to the OS and may be ignored. Implementations may
// non-deterministically return NULL and callers must gracefully handle that.
// It's safe to pass NULL to iree_thread_override_end and in most cases as
// callers aren't checking the returned value they won't notice.
iree_thread_override_t* iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class);

// Ends a priority class override that was began for a thread with
// iree_thread_priority_class_override_begin.
void iree_thread_override_end(iree_thread_override_t* override_token);

// Updates the thread affinity of the given |thread|.
// Affinities are not sticky and may need to be refreshed over time as CPUs are
// enabled/disabled by the OS (such as power mode changes, governor adjustments,
// etc). Users wanting to ensure threads have specific affinities may want to
// request updates whenever new large amounts of work are about to be performed.
//
// NOTE: thread affinities are just a hint. The OS scheduler is free to do
// whatever it wants up to and including entirely ignoring the specified
// affinity. In many cases where cores are oversubscribed setting an affinity
// mask can pessimize battery/thermals/performance as the OS will sometimes try
// to shuffle around threads to disable physical cores/etc.
//
// Compatibility warning: Apple/darwin only support affinity groups, with each
// unique affinity sharing time with all others of the same value. This means
// that trying to get clever with several thread sets with overlapping
// affinities will likely not work as expected. Try to stick with threads that
// run only on a single processor.
void iree_thread_request_affinity(iree_thread_t* thread,
                                  iree_thread_affinity_t affinity);

// Resumes |thread| if it was created suspended.
// This has no effect if the thread is not suspended.
void iree_thread_resume(iree_thread_t* thread);

void iree_thread_yield(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_THREADING_H_
