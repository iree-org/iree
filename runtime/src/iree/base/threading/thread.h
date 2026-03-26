// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_THREADING_THREAD_H_
#define IREE_BASE_THREADING_THREAD_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/threading/affinity.h"

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
IREE_API_EXPORT iree_status_t
iree_thread_create(iree_thread_entry_t entry, void* entry_arg,
                   iree_thread_create_params_t params,
                   iree_allocator_t allocator, iree_thread_t** out_thread);

// Retains the given |thread| for the caller.
IREE_API_EXPORT void iree_thread_retain(iree_thread_t* thread);

// Releases the given |thread| from the caller.
IREE_API_EXPORT void iree_thread_release(iree_thread_t* thread);

// Returns a platform-defined thread ID for the given |thread|.
IREE_API_EXPORT uintptr_t iree_thread_id(iree_thread_t* thread);

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
IREE_API_EXPORT iree_thread_override_t*
iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class);

// Ends a priority class override that was began for a thread with
// iree_thread_priority_class_override_begin.
IREE_API_EXPORT void iree_thread_override_end(
    iree_thread_override_t* override_token);

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
IREE_API_EXPORT void iree_thread_request_affinity(
    iree_thread_t* thread, iree_thread_affinity_t affinity);

// Resumes |thread| if it was created suspended.
// This has no effect if the thread is not suspended.
IREE_API_EXPORT void iree_thread_resume(iree_thread_t* thread);

// Blocks the current thread until |thread| has finished its execution.
IREE_API_EXPORT void iree_thread_join(iree_thread_t* thread);

IREE_API_EXPORT void iree_thread_yield(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_THREAD_H_
