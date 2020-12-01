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

// NOTE: must be first to ensure that we can define settings for all includes.
#include "iree/base/threading_impl.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include "iree/base/atomics.h"
#include "iree/base/threading.h"
#include "iree/base/threading_impl.h"
#include "iree/base/tracing.h"

// Great documentation:
// https://www.microsoftpressstore.com/articles/article.aspx?p=2233328

struct iree_thread_s {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;

  char name[16];
  HANDLE handle;
  DWORD id;

  iree_thread_entry_t entry;
  void* entry_arg;

  iree_atomic_int32_t is_suspended;

  // Thread-safe (has its own synchronization).
  iree_thread_override_list_t qos_override_list;
};

static void iree_thread_set_priority_class(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class);

// Sets the thread's name to the given NUL-terminated string.
//
// See:
// https://docs.microsoft.com/en-us/visualstudio/debugger/how-to-set-a-thread-name-in-native-code
static void iree_thread_set_name(HANDLE handle, const char* name) {
  // Try first to use the modern SetThreadDescription API.
  // This will work even if a debugger is not attached meaning that tools that
  // don't use the debugger API can still query thread names. It's only
  // available on Win10+.
  typedef HRESULT(WINAPI * SetThreadDescriptionFn)(HANDLE hThread,
                                                   PCWSTR lpThreadDescription);
  SetThreadDescriptionFn pSetThreadDescription =
      (SetThreadDescriptionFn)GetProcAddress(GetModuleHandleW(L"Kernel32.dll"),
                                             "SetThreadDescription");
  if (pSetThreadDescription) {
    wchar_t name_wide[16] = {0};
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name, -1, name_wide,
                        IREE_ARRAYSIZE(name_wide) - 1);
    pSetThreadDescription(handle, name_wide);
    return;
  }

  if (!IsDebuggerPresent()) {
    // The name is only captured if a debugger is attached so we can avoid
    // doing any of the work if none is present. This means that a debugger
    // attached to the process after thread creation won't see thread names but
    // that's a rare case anyway.
    return;
  }

#pragma pack(push, 8)
  struct THREADNAME_INFO {
    DWORD dwType;      // Must be 0x1000.
    LPCSTR szName;     // Pointer to name (in user addr space).
    DWORD dwThreadID;  // Thread ID (-1=caller thread).
    DWORD dwFlags;     // Reserved for future use, must be zero.
  };
#pragma pack(pop)

#pragma warning(push)
#pragma warning(disable : 6320 6322)
  struct THREADNAME_INFO info;
  info.dwType = 0x1000;
  info.szName = name;
  info.dwThreadID = GetThreadId(handle);
  info.dwFlags = 0;
  __try {
    RaiseException(0x406D1388u, 0, sizeof(info) / sizeof(ULONG_PTR),
                   (ULONG_PTR*)(&info));
  } __except (EXCEPTION_EXECUTE_HANDLER) {
  }
#pragma warning(pop)
}

static DWORD WINAPI iree_thread_start_routine(LPVOID param) {
  // NOTE: we own a reference to the thread handle so that the creation
  // thread can't delete this out from under us.
  iree_thread_t* thread = (iree_thread_t*)param;

  // Set the thread name used by tracy (which must be called on the thread).
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
  return (DWORD)entry(entry_arg);
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
  strncpy_s(thread->name, IREE_ARRAYSIZE(thread->name), params.name.data,
            min(params.name.size, IREE_ARRAYSIZE(thread->name) - 1));
  iree_atomic_store_int32(&thread->is_suspended, 1, iree_memory_order_relaxed);
  iree_thread_override_list_initialize(iree_thread_set_priority_class,
                                       params.priority_class, thread->allocator,
                                       &thread->qos_override_list);

  // Always create the thread suspended.
  // If we didn't do this it's possible the OS could schedule the thread
  // immediately inside of CreateThread and we wouldn't be able to prepare it
  // (and even weirder, it's possible the thread would have exited and the
  // handle would be closed before we even do anything with it!).
  thread->handle =
      CreateThread(NULL, params.stack_size, iree_thread_start_routine, thread,
                   CREATE_SUSPENDED, &thread->id);
  if (thread->handle == INVALID_HANDLE_VALUE) {
    iree_allocator_free(allocator, thread);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "thread creation failed with %lu", GetLastError());
  }

  // Immediately set thread properties before resuming (so that we don't
  // start on the wrong core/at the wrong priority).
  if (!iree_string_view_is_empty(params.name)) {
    iree_thread_set_name(thread->handle, thread->name);
  }
  if (params.priority_class != IREE_THREAD_PRIORITY_CLASS_NORMAL) {
    iree_thread_set_priority_class(thread, params.priority_class);
  }
  if (params.initial_affinity.specified) {
    iree_thread_request_affinity(thread, params.initial_affinity);
  }

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

static void iree_thread_delete(iree_thread_t* thread) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_thread_resume(thread);

  CloseHandle(thread->handle);
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
  return (uintptr_t)thread->id;
}

// Sets the thread priority to the given |priority_class| immediately.
static void iree_thread_set_priority_class(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  IREE_TRACE_ZONE_BEGIN(z0);

  DWORD priority = THREAD_PRIORITY_NORMAL;
  switch (priority_class) {
    case IREE_THREAD_PRIORITY_CLASS_LOWEST:
      priority = THREAD_PRIORITY_LOWEST;
      break;
    case IREE_THREAD_PRIORITY_CLASS_LOW:
      priority = THREAD_PRIORITY_BELOW_NORMAL;
      break;
    case IREE_THREAD_PRIORITY_CLASS_NORMAL:
      priority = THREAD_PRIORITY_NORMAL;
      break;
    case IREE_THREAD_PRIORITY_CLASS_HIGH:
      priority = THREAD_PRIORITY_ABOVE_NORMAL;
      break;
    case IREE_THREAD_PRIORITY_CLASS_HIGHEST:
      priority = THREAD_PRIORITY_HIGHEST;
      break;
  }
  SetThreadPriority(thread->handle, priority);

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

void iree_thread_request_affinity(iree_thread_t* thread,
                                  iree_thread_affinity_t affinity) {
  if (!affinity.specified) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  PROCESSOR_NUMBER ideal_processor;
  memset(&ideal_processor, 0, sizeof(ideal_processor));
  ideal_processor.Group = affinity.group;
  ideal_processor.Number = affinity.id;
  SetThreadIdealProcessorEx(thread->handle, &ideal_processor, NULL);

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
    ResumeThread(thread->handle);
  }

  IREE_TRACE_ZONE_END(z0);
}

#endif  // IREE_PLATFORM_WINDOWS
