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

#include "iree/base/atomics.h"
#include "iree/base/threading.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_APPLE)

#include <errno.h>
#include <mach/mach.h>
#include <mach/thread_act.h>
#include <pthread.h>
#include <string.h>

// Useful to see how pthreads is implemented on (old) darwin:
// https://opensource.apple.com/source/Libc/Libc-825.40.1/pthreads/pthread.c.auto.html

struct iree_thread_s {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  char name[16];
  pthread_t handle;
  mach_port_t mach_port;

  iree_thread_entry_t entry;
  void* entry_arg;

  iree_atomic_int32_t is_suspended;
};

static qos_class_t iree_thread_qos_class_for_priority_class(
    iree_thread_priority_class_t priority_class);

static void* iree_thread_start_routine(void* param) {
  // NOTE: we own a reference to the thread handle so that the creation
  // thread can't delete this out from under us.
  iree_thread_t* thread = (iree_thread_t*)param;

  // Set the thread name used by debuggers and tracy (which must be called on
  // the thread).
  pthread_setname_np(thread->name);
  IREE_TRACE_SET_THREAD_NAME(thread->name);

  // "Consume" the entry info so that we don't see it again (as we don't own
  // its lifetime).
  iree_thread_entry_t entry = thread->entry;
  void* entry_arg = thread->entry_arg;
  thread->entry = NULL;
  thread->entry_arg = NULL;

  // Release our ownership of the thread handle. If the creating thread doesn't
  // want it this will free the memory and fully detach the thread.
  iree_thread_release(thread);

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
      iree_allocator_malloc(allocator, sizeof(iree_thread_t), (void**)&thread);
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
  iree_atomic_store_int32(&thread->is_suspended, 1, iree_memory_order_relaxed);

  pthread_attr_t thread_attr;
  pthread_attr_init(&thread_attr);
  pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_DETACHED);
  if (params.stack_size) {
    pthread_attr_setstacksize(&thread_attr, params.stack_size);
  }

  // Ensure we start with the right QoS class.
  qos_class_t qos_class =
      iree_thread_qos_class_for_priority_class(priority_class);
  pthread_attr_set_qos_class_np(&thread_attr, qos_class, 0);

  // Always create the thread suspended.
  // If we didn't do this it's possible the OS could schedule the thread
  // immediately inside of CreateThread and we wouldn't be able to prepare
  // it (and even weirder, it's possible the thread would have exited and
  // the handle would be closed before we even do anything with it!).
  int rc = pthread_create_suspended_np(&thread->handle, &thread_attr,
                                       &iree_thread_start_routine, thread);
  pthread_attr_destroy(&thread_attr);
  if (rc != 0) {
    iree_allocator_free(allocator, thread);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "thread creation failed with %d", rc);
  }

  thread->mach_port = pthread_mach_thread_np(thread->handle);

  // Retain the thread for the thread itself; this way if the caller immediately
  // releases the iree_thread_t handle the thread won't explode.
  iree_thread_retain(thread);

  // If the thread is being created unsuspended then resume now. Otherwise the
  // caller must resume when they want it spun up.
  if (!params.create_suspended) {
    iree_thread_resume(thread);
  }

  IREE_TRACE_ZONE_END(z0);
  *out_thread = thread;
  return iree_ok_status();
}

void iree_thread_retain(iree_thread_t* thread) {
  if (thread) {
    iree_atomic_ref_count_inc(&thread->ref_count);
  }
}

void iree_thread_release(iree_thread_t* thread) {
  if (thread && iree_atomic_ref_count_dec(&thread->ref_count) == 1) {
    iree_allocator_free(thread->allocator, thread);
  }
}

uintptr_t iree_thread_id(iree_thread_t* thread) {
  return (uintptr_t)thread->handle;
}

// Maps an IREE iree_thread_priority_class_t value to a QoS type.
// https://developer.apple.com/library/archive/documentation/Performance/Conceptual/EnergyGuide-iOS/PrioritizeWorkWithQoS.html
static qos_class_t iree_thread_qos_class_for_priority_class(
    iree_thread_priority_class_t priority_class) {
  switch (priority_class) {
    case IREE_THREAD_PRIORITY_CLASS_LOWEST:
      return QOS_CLASS_BACKGROUND;
    case IREE_THREAD_PRIORITY_CLASS_LOW:
      return QOS_CLASS_UTILITY;
    default:
    case IREE_THREAD_PRIORITY_CLASS_NORMAL:
      return QOS_CLASS_DEFAULT;
    case IREE_THREAD_PRIORITY_CLASS_HIGH:
      return QOS_CLASS_USER_INITIATED;
    case IREE_THREAD_PRIORITY_CLASS_HIGHEST:
      return QOS_CLASS_USER_INTERACTIVE;
  }
}

iree_thread_override_t* iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  IREE_TRACE_ZONE_BEGIN(z0);

  qos_class_t qos_class =
      iree_thread_qos_class_for_priority_class(priority_class);
  pthread_override_t override =
      pthread_override_qos_class_start_np(thread->handle, qos_class, 0);

  IREE_TRACE_ZONE_END(z0);
  return (iree_thread_override_t*)override;
}

void iree_thread_override_end(iree_thread_override_t* override) {
  IREE_TRACE_ZONE_BEGIN(z0);

  pthread_override_qos_class_end_np((pthread_override_t)override);

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_resume(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: we don't track the suspend/resume depth here because we don't
  // expose suspend as an operation (yet). If we did we'd want to make sure we
  // always balance suspend/resume or else we'll mess with any
  // debuggers/profilers that may be suspending threads for their own uses.
  int32_t expected = 1;
  if (iree_atomic_compare_exchange_strong_int32(
          &thread->is_suspended, &expected, 0, iree_memory_order_seq_cst,
          iree_memory_order_seq_cst)) {
    thread_resume(thread->mach_port);
  }

  IREE_TRACE_ZONE_END(z0);
}

#endif  // IREE_PLATFORM_APPLE
