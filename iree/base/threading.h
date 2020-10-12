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

#ifndef IREE_BASE_THREADING_H_
#define IREE_BASE_THREADING_H_

#include <stdbool.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// iree_thread_t
//==============================================================================

typedef struct iree_thread_s iree_thread_t;

// Specifies a thread's priority class.
// These translate roughly to the same thing across all platforms, though they
// are just a hint and the schedulers on various platforms may behave very
// differently. When in doubt prefer to write code that works at the extremes
// of the classes.
enum iree_thread_priority_class_e {
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
};
typedef int32_t iree_thread_priority_class_t;

// Thread creation parameters.
// All are optional and the entire struct can safely be zero-initialized.
typedef struct {
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
  // This may be changed later via iree_thread_set_priority_class; see that for
  // more information.
  iree_thread_priority_class_t priority_class;
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

typedef struct iree_thread_override_s iree_thread_override_t;

// Begins overriding the priority class of the given |thread|.
// The priority of the thread will be the max of the base priority and the
// overridden priority. Callers must pass the returned override token to
// iree_thread_override_end.
iree_thread_override_t* iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class);

// Ends a priority class override that was began for a thread with
// iree_thread_priority_class_override_begin.
void iree_thread_override_end(iree_thread_override_t* override_token);

// Resumes |thread| if it was created suspended.
// This has no effect if the thread is not suspended.
void iree_thread_resume(iree_thread_t* thread);

//==============================================================================
// iree_fpu_state_*
//==============================================================================

// Flags controlling FPU features.
enum iree_fpu_state_flags_e {
  // Platform default.
  IREE_FPU_STATE_DEFAULT = 0,

  // Denormals can cause some serious slowdowns in certain ISAs where they may
  // be implemented in microcode. Flushing them to zero instead of letting them
  // propagate ensures that the slow paths aren't hit. This is a fast-math style
  // optimization (and is often part of all compiler's fast-math set of flags).
  //
  // https://en.wikipedia.org/wiki/Denormal_number
  // https://carlh.net/plugins/denormals.php
  // https://www.xspdf.com/resolution/50507310.html
  IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO = 1 << 0,
};
typedef uint32_t iree_fpu_state_flags_t;

// Opaque FPU state vector manipulated with iree_fpu_* functions.
typedef struct {
  uint64_t previous_value;
  uint64_t current_value;
} iree_fpu_state_t;

// Pushes a new floating-point unit (FPU) state for the current thread.
// May lead to a pipeline flush; avoid if possible.
iree_fpu_state_t iree_fpu_state_push(iree_fpu_state_flags_t flags);

// Restores the FPU state of the thread to its original value.
// May lead to a pipeline flush; avoid if possible.
void iree_fpu_state_pop(iree_fpu_state_t state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_H_
