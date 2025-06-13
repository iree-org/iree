// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: must be first to ensure that we can define settings for all includes.
#include "iree/base/internal/threading_impl.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_EMSCRIPTEN) || \
    defined(IREE_PLATFORM_LINUX)

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/call_once.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"

#if defined(IREE_PLATFORM_EMSCRIPTEN)
#include <emscripten/threading.h>
#endif  // IREE_PLATFORM_EMSCRIPTEN

struct iree_thread_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  char name[32];
  pthread_t handle;

  iree_thread_entry_t entry;
  void* entry_arg;

  iree_atomic_int32_t suspend_count;
  iree_notification_t suspend_barrier;

  // Thread-safe (has its own synchronization).
  iree_thread_override_list_t qos_override_list;
};

static void iree_thread_set_priority_class(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class);

static bool iree_thread_resumed_predicate(void* arg) {
  iree_thread_t* thread = (iree_thread_t*)arg;
  return iree_atomic_load(&thread->suspend_count, iree_memory_order_acquire) ==
         0;
}

#if defined(IREE_PLATFORM_EMSCRIPTEN)

static int iree_thread_set_name(pthread_t handle, const char* name) {
  emscripten_set_thread_name(handle, name);
  return 0;
}

#else

typedef int (*pthread_setname_np_fn_t)(pthread_t thread, const char* name);

static pthread_setname_np_fn_t iree_pthread_setname_np_fn = NULL;
static void iree_thread_try_query_setname_fn(void) {
  iree_pthread_setname_np_fn =
      (pthread_setname_np_fn_t)dlsym(RTLD_DEFAULT, "pthread_setname_np");
}

static int iree_thread_set_name(pthread_t handle, const char* name) {
  IREE_TRACE_ZONE_BEGIN(z0);
  static iree_once_flag fn_query_flag = IREE_ONCE_FLAG_INIT;
  iree_call_once(&fn_query_flag, iree_thread_try_query_setname_fn);
  int rc;
  if (iree_pthread_setname_np_fn) {
    rc = iree_pthread_setname_np_fn(handle, name);
  } else {
    rc = EINVAL;
  }
  IREE_TRACE_ZONE_END(z0);
  return rc;
}

#endif  // IREE_PLATFORM_EMSCRIPTEN

static void* iree_thread_start_routine(void* param) {
  // NOTE: we own a reference to the thread handle so that the creation
  // thread can't delete this out from under us.
  iree_thread_t* thread = (iree_thread_t*)param;

  // Set the thread name used by debuggers and tracy (which must be called on
  // the thread).
  iree_thread_set_name(thread->handle, thread->name);
  IREE_TRACE_SET_THREAD_NAME(thread->name);

  // Wait until we resume if we were created suspended.
  while (iree_atomic_load(&thread->suspend_count, iree_memory_order_acquire) >
         0) {
    iree_notification_await(&thread->suspend_barrier,
                            iree_thread_resumed_predicate, thread,
                            iree_infinite_timeout());
  }

  // "Consume" the entry info so that we don't see it again (as we don't own
  // its lifetime).
  iree_thread_entry_t entry = thread->entry;
  void* entry_arg = thread->entry_arg;
  thread->entry = NULL;
  thread->entry_arg = NULL;

  // Call the user thread entry point function.
  // Note that this can be a tail-call which saves a stack frame in all threads
  // (which is really just to make call stacks in debuggers much cleaner).
  return (void*)((uintptr_t)entry(entry_arg));
}

iree_status_t iree_thread_create(iree_thread_entry_t entry, void* entry_arg,
                                 iree_thread_create_params_t params,
                                 iree_allocator_t allocator,
                                 iree_thread_t** out_thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate our thread struct; we'll use it to shuttle params into the thread
  // (including the user-specified entry_arg).
  iree_thread_t* thread = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*thread), (void**)&thread);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_atomic_ref_count_init(&thread->ref_count);
  thread->allocator = allocator;
  thread->entry = entry;
  thread->entry_arg = entry_arg;
  iree_strncpy_s(thread->name, IREE_ARRAYSIZE(thread->name), params.name.data,
                 iree_min(params.name.size, IREE_ARRAYSIZE(thread->name) - 1));
  thread->suspend_count = IREE_ATOMIC_VAR_INIT(params.create_suspended ? 1 : 0);
  iree_notification_initialize(&thread->suspend_barrier);
  iree_thread_override_list_initialize(iree_thread_set_priority_class,
                                       params.priority_class, thread->allocator,
                                       &thread->qos_override_list);

  pthread_attr_t thread_attr;
  pthread_attr_init(&thread_attr);
  pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
  if (params.stack_size) {
    pthread_attr_setstacksize(&thread_attr, params.stack_size);
  }

  *out_thread = thread;

  // Unfortunately we can't create the thread suspended (no API). This means
  // that we are likely to incur some thrashing here as the thread gets spun up
  // immediately. We emulate the create_suspended behavior by waiting in the
  // thread until iree_thread_resume is called which at least gives us the same
  // execution order guarantee across all platforms.
  int rc;
  {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "pthread_create");
    rc = pthread_create(&thread->handle, &thread_attr,
                        &iree_thread_start_routine, thread);
    IREE_TRACE_ZONE_END(z1);
  }
  pthread_attr_destroy(&thread_attr);
  if (rc != 0) {
    iree_thread_release(thread);  // for caller
    *out_thread = NULL;
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "thread creation failed with %d", rc);
  }

  if (params.priority_class != IREE_THREAD_PRIORITY_CLASS_NORMAL) {
    iree_thread_set_priority_class(thread, params.priority_class);
  }
  if (!iree_thread_affinity_is_unspecified(params.initial_affinity)) {
    iree_thread_request_affinity(thread, params.initial_affinity);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_thread_delete(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_thread_resume(thread);
  pthread_join(thread->handle, NULL);

  iree_notification_deinitialize(&thread->suspend_barrier);
  iree_thread_override_list_deinitialize(&thread->qos_override_list);
  iree_allocator_free(thread->allocator, thread);

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_retain(iree_thread_t* thread) {
  if (thread) {
    iree_atomic_ref_count_inc(&thread->ref_count);
  }
}

void iree_thread_release(iree_thread_t* thread) {
  if (thread && iree_atomic_ref_count_dec(&thread->ref_count) == 1) {
    iree_thread_delete(thread);
  }
}

uintptr_t iree_thread_id(iree_thread_t* thread) {
  return (uintptr_t)thread->handle;
}

// Maps an IREE iree_thread_priority_class_t value to a pthreads priority param.
// The min/max ranges of the priority are implementation dependent so we need to
// do this at runtime.
static struct sched_param iree_thread_sched_param_for_priority_class(
    int policy, iree_thread_priority_class_t priority_class) {
  struct sched_param param;
  memset(&param, 0, sizeof(param));
  int min_priority = sched_get_priority_min(policy);
  int max_priority = sched_get_priority_max(policy);
  int normal_priority = (max_priority - min_priority) / 2 + min_priority;
  switch (priority_class) {
    case IREE_THREAD_PRIORITY_CLASS_LOWEST:
      param.sched_priority = min_priority;
      break;
    case IREE_THREAD_PRIORITY_CLASS_LOW:
      param.sched_priority =
          (normal_priority - min_priority) / 2 + min_priority;
      break;
    case IREE_THREAD_PRIORITY_CLASS_NORMAL:
      param.sched_priority = normal_priority;
      break;
    case IREE_THREAD_PRIORITY_CLASS_HIGH:
      param.sched_priority =
          (max_priority - normal_priority) / 2 + normal_priority;
      break;
    case IREE_THREAD_PRIORITY_CLASS_HIGHEST:
      param.sched_priority = max_priority;
      break;
  }
  return param;
}

// Sets the thread priority to the given |priority_class|, resetting any
// previous value.
//
// NOTE: this probably doesn't work on Android, because Android.
// They seem to use linux LWPs and setpriority/nice on the tid will actually
// change the priority. It doesn't seem possible to elevate priority above
// normal (without root), but it would at least be useful to be able to
// indicate background threads.
//
// See:
// https://stackoverflow.com/questions/17398075/change-native-thread-priority-on-android-in-c-c
// https://android.googlesource.com/platform/frameworks/native/+/android-4.2.2_r1/include/utils/ThreadDefs.h
//
// TODO(benvanik): try this from filament:
// https://github.com/google/filament/blob/56682794d398236c4caa5be40d80acdb73a13bc8/libs/utils/src/JobSystem.cpp
static void iree_thread_set_priority_class(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  IREE_TRACE_ZONE_BEGIN(z0);

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_EMSCRIPTEN)
  // TODO(benvanik): Some sort of solution on Android, if possible (see above)
  // TODO(benvanik): Some sort of solution on Emscripten, if possible
#else
  int policy = 0;
  struct sched_param param;
  pthread_getschedparam(thread->handle, &policy, &param);
  param = iree_thread_sched_param_for_priority_class(policy, priority_class);
  pthread_setschedparam(thread->handle, policy, &param);
#endif  // IREE_PLATFORM_ANDROID

  IREE_TRACE_ZONE_END(z0);
}

iree_thread_override_t* iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_thread_override_t* override = iree_thread_override_list_add(
      &thread->qos_override_list, thread, priority_class);
  IREE_TRACE_ZONE_END(z0);
  return override;
}

void iree_thread_override_end(iree_thread_override_t* override) {
  if (!override) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_thread_override_remove_self(override);
  IREE_TRACE_ZONE_END(z0);
}

// Sets all CPU bits in the given |out_set|.
// The platform is allowed to place the thread on any CPU.
static void iree_thread_make_cpu_set_all(cpu_set_t* out_set) {
  for (uint32_t i = 0; i < CPU_SETSIZE; ++i) {
    CPU_SET(i, out_set);
  }
}

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

// Sets CPU bits associated with the given NUMA node ID.
// If the platform query fails then all CPU bits are set.
static void iree_thread_make_cpu_set_from_node_id(uint32_t node_id,
                                                  cpu_set_t* out_set) {
  // e.g. /sys/devices/system/node/node0/cpumap
  char cpumap_path[256];
  snprintf(cpumap_path, sizeof(cpumap_path),
           "/sys/devices/system/node/node%u/cpumap", node_id);

  // Open file for reading. This should succeed under hypervisors/lockdown.
  FILE* file = fopen(cpumap_path, "r");
  if (!file) {
    // Permission denied or not found (not a conformant Linux kernel).
    iree_thread_make_cpu_set_all(out_set);
    return;
  }

  // Read the entire file to EOF and get the cpumap line.
  // After trimming we expect |line| to be something like:
  // 'ffffffff,ffffffff,ffffffff,00000000,00000000,00000000'
  char line_buffer[512];
  const size_t read_length = fread(line_buffer, 1, sizeof(line_buffer), file);
  if (ferror(file)) {
    // Read should never fail, but may if the CPU set grows to thousands. We'd
    // probably want to then query the file length and allocate a heap buffer.
    // For now all systems we can observe easily fit into our stack buffer.
    iree_thread_make_cpu_set_all(out_set);
    return;
  }
  iree_string_view_t line =
      iree_string_view_trim(iree_make_string_view(line_buffer, read_length));

  // Parse each comma-delimited segment. Segments are a base-16 encoded uint32_t
  // value. Each segment contains 32 CPU bits and we track the current index
  // as we walk them to get the absolute cpu_set_t index.
  intptr_t split_index = 0;
  iree_host_size_t cpu_index = 0;
  do {
    iree_string_view_t segment_str;
    split_index = iree_string_view_split(line, ',', &segment_str, &line);
    uint32_t segment = 0;
    if (!iree_string_view_atoi_uint32_base(segment_str, 16, &segment)) {
      // Failed to parse segment as an integer.
      iree_thread_make_cpu_set_all(out_set);
      return;
    }
    for (iree_host_size_t i = 0; i < 32; ++i) {
      if (segment & (1ull << i)) {
        CPU_SET(cpu_index + i, out_set);
      }
    }
    cpu_index += 32;
  } while (split_index != -1);

  fclose(file);
}

#else

// No implementation available. BSD may have some equivalent to the Linux
// cpumap we could use.
static void iree_thread_make_cpu_set_from_node_id(uint32_t node_id,
                                                  cpu_set_t* out_set) {
  iree_thread_make_cpu_set_all(out_set);
}

#endif  // IREE_PLATFORM_EMSCRIPTEN

static void iree_thread_make_cpu_set_from_affinity(
    iree_thread_affinity_t affinity, cpu_set_t* out_set) {
  CPU_ZERO(out_set);

  // Assign to any processor in the group.
  if (affinity.group_any) {
    iree_thread_make_cpu_set_from_node_id(affinity.group, out_set);
    return;
  }

  // Specific processors can be set directly and optionally we also set its
  // paired SMT processor. Note that we don't check whether SMT is enabled and
  // assume the smt field is only assigned if it is.
  if (affinity.id_assigned) {
    CPU_SET(affinity.id, out_set);
    if (affinity.smt) {
      CPU_SET(affinity.id + 1, out_set);
    }
    return;
  }

  // No specific affinity specified; use any CPU.
  iree_thread_make_cpu_set_all(out_set);
}

void iree_thread_request_affinity(iree_thread_t* thread,
                                  iree_thread_affinity_t affinity) {
  IREE_TRACE_ZONE_BEGIN(z0);
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  char affinity_desc[64];
  int affinity_desc_length =
      snprintf(affinity_desc, IREE_ARRAYSIZE(affinity_desc),
               "group_any=%u, group=%u, id_assigned=%u, id=%u, smt=%u",
               affinity.group_any, affinity.group, affinity.id_assigned,
               affinity.id, affinity.smt);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, affinity_desc, affinity_desc_length);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  cpu_set_t cpu_set;
  iree_thread_make_cpu_set_from_affinity(affinity, &cpu_set);

#if defined(IREE_PLATFORM_ANDROID)
  // `pthread_gettid_np` is only available on API 21+ and it is needed to set
  // affinity so skip it for older API versions.
#if __ANDROID_API__ >= 21
  // Android doesn't have pthread_setaffinity_np but that's usually just
  // implemented as this sequence anyway:
  pid_t tid = pthread_gettid_np(thread->handle);
  sched_setaffinity(tid, sizeof(cpu_set), &cpu_set);
#endif  // __ANDROID_API__ >= 21
#elif defined(IREE_PLATFORM_EMSCRIPTEN)
  // TODO(benvanik): Some sort of solution on Emscripten, if possible
#else
  pthread_setaffinity_np(thread->handle, sizeof(cpu_set), &cpu_set);
#endif  // IREE_PLATFORM_*

  // TODO(benvanik): make a set_mempolicy syscall where available to set the
  // NUMA allocation pinning for the calling thread. We'll likely want an
  // iree_allocator_t control operation for this to allow users to plug in
  // their own implementations and also let the HAL pin allocated buffers.

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_resume(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_atomic_exchange(&thread->suspend_count, 0,
                           iree_memory_order_acq_rel) == 1) {
    iree_notification_post(&thread->suspend_barrier, IREE_ALL_WAITERS);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_join(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);
  pthread_join(thread->handle, NULL);
  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_yield(void) { sched_yield(); }

#endif  // IREE_PLATFORM_*
