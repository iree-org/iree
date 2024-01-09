// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: must be first to ensure that we can define settings for all includes.
#include "iree/base/internal/threading_impl.h"

#if defined(IREE_PLATFORM_APPLE)

#include <errno.h>
#include <mach/mach.h>
#include <mach/thread_act.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/threading.h"

// Useful to see how pthreads is implemented on (old) darwin:
// https://opensource.apple.com/source/Libc/Libc-825.40.1/pthreads/pthread.c.auto.html

struct iree_thread_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  char name[16];
  pthread_t handle;
  mach_port_t mach_port;

  iree_thread_entry_t entry;
  void* entry_arg;

  iree_atomic_int32_t is_suspended;
};

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
    case IREE_THREAD_PRIORITY_CLASS_HIGH:
      return QOS_CLASS_USER_INITIATED;
    case IREE_THREAD_PRIORITY_CLASS_HIGHEST:
      return QOS_CLASS_USER_INTERACTIVE;
  }
}

static void iree_thread_set_name(const char* name) {
  IREE_TRACE_ZONE_BEGIN(z0);
  pthread_setname_np(name);
  IREE_TRACE_SET_THREAD_NAME(name);
  IREE_TRACE_ZONE_END(z0);
}

static void* iree_thread_start_routine(void* param) {
  // NOTE: we own a reference to the thread handle so that the creation
  // thread can't delete this out from under us.
  iree_thread_t* thread = (iree_thread_t*)param;

  // Set the thread name used by debuggers and tracy (which must be called on
  // the thread).
  iree_thread_set_name(thread->name);

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
  iree_atomic_store_int32(&thread->is_suspended,
                          params.create_suspended ? 1 : 0,
                          iree_memory_order_relaxed);

  pthread_attr_t thread_attr;
  pthread_attr_init(&thread_attr);
  pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
  if (params.stack_size) {
    pthread_attr_setstacksize(&thread_attr, params.stack_size);
  }

  // Ensure we start with the right QoS class.
  qos_class_t qos_class;
  if (params.initial_affinity.specified && params.initial_affinity.smt) {
    qos_class = QOS_CLASS_BACKGROUND;
  } else {
    qos_class = iree_thread_qos_class_for_priority_class(params.priority_class);
  }
  pthread_attr_set_qos_class_np(&thread_attr, qos_class, 0);

  // Retain the thread for the thread itself; this way if the caller immediately
  // releases the iree_thread_t handle the thread won't explode.
  iree_thread_retain(thread);
  *out_thread = thread;

  // Create the thread either suspended or running as the user requested.
  int rc;
  if (params.create_suspended) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "pthread_create_suspended_np");
    rc = pthread_create_suspended_np(&thread->handle, &thread_attr,
                                     &iree_thread_start_routine, thread);
    IREE_TRACE_ZONE_END(z1);
  } else {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "pthread_create");
    rc = pthread_create(&thread->handle, &thread_attr,
                        &iree_thread_start_routine, thread);
    IREE_TRACE_ZONE_END(z1);
  }
  pthread_attr_destroy(&thread_attr);
  if (rc != 0) {
    iree_thread_release(thread);  // for self
    iree_thread_release(thread);  // for caller
    *out_thread = NULL;
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "thread creation failed with %d", rc);
  }

  thread->mach_port = pthread_mach_thread_np(thread->handle);
  if (params.initial_affinity.specified) {
    iree_thread_request_affinity(thread, params.initial_affinity);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_thread_delete(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_thread_resume(thread);
  pthread_join(thread->handle, NULL);

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

iree_thread_override_t* iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  IREE_TRACE_ZONE_BEGIN(z0);

  qos_class_t qos_class =
      iree_thread_qos_class_for_priority_class(priority_class);
  pthread_override_t override =
      pthread_override_qos_class_start_np(thread->handle, qos_class, 0);

  IREE_TRACE_ZONE_END(z0);
  return (iree_thread_override_t*) override;
}

void iree_thread_override_end(iree_thread_override_t* override) {
  // pthread_override_qos_class_start_np() in the begin call can fail, in
  // which case we must not attempt to end it.
  if (!override) return;

  IREE_TRACE_ZONE_BEGIN(z0);

  pthread_override_qos_class_end_np((pthread_override_t) override);

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_request_affinity(iree_thread_t* thread,
                                  iree_thread_affinity_t affinity) {
  if (!affinity.specified) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Use mach_task_self when the caller requesting the affinity change is the
  // thread being changed.
  mach_port_t thread_port =
      thread->handle == pthread_self() ? mach_task_self() : thread->mach_port;

  // See:
  // https://gist.github.com/Coneko/4234842
  // https://fergofrog.com/code/cbowser/xnu/osfmk/mach/thread_policy.h.html
  // http://www.hybridkernel.com/2015/01/18/binding_threads_to_cores_osx.html
  thread_affinity_policy_data_t policy_data = {affinity.id};
  thread_policy_set(thread_port, THREAD_AFFINITY_POLICY,
                    (thread_policy_t)(&policy_data),
                    THREAD_AFFINITY_POLICY_COUNT);

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
          &thread->is_suspended, &expected, 0, iree_memory_order_acq_rel,
          iree_memory_order_relaxed /* expected is unused */)) {
    thread_resume(thread->mach_port);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_thread_yield(void) { sched_yield(); }

#endif  // IREE_PLATFORM_APPLE
