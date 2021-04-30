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

#ifndef IREE_BASE_INTERNAL_THREADING_H_
#define IREE_BASE_INTERNAL_THREADING_H_

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
// Linux/Android:
//   sched_setaffinity is used to pin the thread to the core with the given ID.
//   There are, naturally, issues on Android where if the governer has turned
//   off some cores (such as powering down big cores in an ARM big.LITTLE
//   configuration) the affinity request will be dropped on the floor even if
//   the cores are later enabled. This is one of the reasons why we note in
//   iree_thread_request_affinity that requests may need to be made at
//   ¯\_(ツ)_/¯ intervals. In the future we can try to hook into power
//   management infra to see if we can tell when we need to do this.
//
// Windows:
//   Stuff just works. Love it.
typedef struct {
  uint32_t specified : 1;
  uint32_t smt : 1;
  uint32_t group : 7;
  uint32_t id : 23;
} iree_thread_affinity_t;

// Sets |thread_affinity| to match with any processor in the system.
void iree_thread_affinity_set_any(iree_thread_affinity_t* out_thread_affinity);

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

// Updates the thread affinity of the given |thread|.
// Affinities are not sticky and may need to be refreshed over time as CPUs are
// enabled/disabled by the OS (such as power mode changes, governer adjustments,
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

//==============================================================================
// iree_call_once
//==============================================================================
// Emulates the C11 call_once feature as few seem to have it.
// https://en.cppreference.com/w/c/thread/call_once

#if defined(__has_include)
#if __has_include(<thread.h>)
#define IREE_HAS_C11_THREAD_H 1
#endif
#endif

#if defined(IREE_HAS_C11_THREAD_H)

// Always prefer the C11 header if present.
#include <thread.h>
#define IREE_ONCE_FLAG_INIT ONCE_FLAG_INIT
#define iree_once_flag ONCE_FLAG
#define iree_call_once call_once

#elif defined(IREE_PLATFORM_WINDOWS)

// Windows fallback using the native InitOnceExecuteOnce:
// https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-initonceexecuteonce

// Expands to a value that can be used to initialize an object of type
// iree_once_flag.
#define IREE_ONCE_FLAG_INIT INIT_ONCE_STATIC_INIT

// Complete object type capable of holding a flag used by iree_call_once.
typedef INIT_ONCE iree_once_flag;

typedef struct {
  void (*func)(void);
} iree_call_once_impl_params_t;
static BOOL CALLBACK iree_call_once_callback_impl(PINIT_ONCE InitOnce,
                                                  PVOID Parameter,
                                                  PVOID* Context) {
  // https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nc-synchapi-pinit_once_fn
  iree_call_once_impl_params_t* param =
      (iree_call_once_impl_params_t*)Parameter;
  (param->func)();
  ((void)InitOnce);
  ((void)Context);  // suppress warning
  return TRUE;
}

// Calls |func| exactly once, even if invoked from several threads.
// The completion of the function synchronizes with all previous or subsequent
// calls to call_once with the same flag variable.
static inline void iree_call_once(iree_once_flag* flag, void (*func)(void)) {
  iree_call_once_impl_params_t param;
  param.func = func;
  InitOnceExecuteOnce(flag, iree_call_once_callback_impl, (PVOID)&param, NULL);
}

#else

// Fallback using pthread_once:
// https://pubs.opengroup.org/onlinepubs/007908775/xsh/pthread_once.html

#include <pthread.h>

// Expands to a value that can be used to initialize an object of type
// iree_once_flag.
#define IREE_ONCE_FLAG_INIT PTHREAD_ONCE_INIT

// Complete object type capable of holding a flag used by iree_call_once.
typedef pthread_once_t iree_once_flag;

// Calls |func| exactly once, even if invoked from several threads.
// The completion of the function synchronizes with all previous or subsequent
// calls to call_once with the same flag variable.
static inline void iree_call_once(iree_once_flag* flag, void (*func)(void)) {
  pthread_once(flag, func);
}

#endif  // IREE_HAS_C11_THREAD_H / fallbacks

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_THREADING_H_
