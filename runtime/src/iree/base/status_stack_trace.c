// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_APPLE)
#include <dlfcn.h>
#include <execinfo.h>
#define IREE_HAVE_BACKTRACE 1
#define IREE_STATUS_HAVE_STACK_TRACE_SUPPORT 1
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
// Currently disabled because we can't get meaningful stacks on Linux.
// #define _GNU_SOURCE 1
// #include <dlfcn.h>
// #include <execinfo.h>
// #define IREE_HAVE_BACKTRACE 1
// #define IREE_STATUS_HAVE_STACK_TRACE_SUPPORT 1
#elif defined(IREE_PLATFORM_WINDOWS)
#pragma warning(disable : 4091)
#include <dbghelp.h>
#define IREE_STATUS_HAVE_STACK_TRACE_SUPPORT 1
#endif  // IREE_PLATFORM_*

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

#include "iree/base/alignment.h"
#include "iree/base/allocator.h"
#include "iree/base/assert.h"
#include "iree/base/config.h"
#include "iree/base/status.h"
#include "iree/base/status_payload.h"
#include "iree/base/tracing.h"

#if defined(IREE_STATUS_HAVE_STACK_TRACE_SUPPORT) && \
    ((IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) != 0)

// TODO(benvanik): make a string_view utility to share with console tracing.
static iree_string_view_t iree_status_trim_file_path(const char* file_name) {
  size_t file_name_length = strlen(file_name);
  for (int i = (int)file_name_length - 1; i >= 0; --i) {
    char c = file_name[i];
    if (c == '/' || c == '\\') {
      return iree_make_string_view(file_name + i + 1, file_name_length - i - 1);
    }
  }
  return iree_make_string_view(file_name, file_name_length);
}

static iree_host_size_t iree_string_buffer_append_cstr(
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t buffer_length, const char* str) {
  iree_host_size_t n =
      snprintf(buffer ? buffer + buffer_length : NULL,
               buffer ? buffer_capacity - buffer_length : 0, "%s", str);
  return IREE_UNLIKELY(n < 0) ? 0 : buffer_length + n;
}

static iree_host_size_t IREE_PRINTF_ATTRIBUTE(4, 5)
    iree_string_buffer_append_format(iree_host_size_t buffer_capacity,
                                     char* buffer,
                                     iree_host_size_t buffer_length,
                                     const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  iree_host_size_t n =
      vsnprintf(buffer ? buffer + buffer_length : NULL,
                buffer ? buffer_capacity - buffer_length : 0, format, varargs);
  va_end(varargs);
  return IREE_UNLIKELY(n < 0) ? 0 : buffer_length + n;
}

#if defined(IREE_PLATFORM_WINDOWS)

typedef BOOL(WINAPI* PFN_SymInitialize)(HANDLE, PCSTR, BOOL);
typedef BOOL(WINAPI* PFN_SymCleanup)(HANDLE);
typedef BOOL(WINAPI* PFN_SymGetModuleInfo64)(HANDLE, DWORD64,
                                             PIMAGEHLP_MODULE64);
typedef BOOL(WINAPI* PFN_SymFromAddr)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);
typedef BOOL(WINAPI* PFN_SymGetLineFromAddr64)(HANDLE, DWORD64, PDWORD,
                                               PIMAGEHLP_LINE);
typedef struct iree_symbol_resolver_t {
  SRWLOCK mutex;
  HMODULE library;
  PFN_SymInitialize SymInitialize;
  PFN_SymCleanup SymCleanup;
  PFN_SymGetModuleInfo64 SymGetModuleInfo64;
  PFN_SymFromAddr SymFromAddr;
  PFN_SymGetLineFromAddr64 SymGetLineFromAddr64;
} iree_symbol_resolver_t;

// If tracy is enabled then it has its own dbghelp lock we need to use.
// If not enabled then we
#if defined(TRACY_ENABLE)
void IREEDbgHelpInit(void);
void IREEDbgHelpLock(void);
void IREEDbgHelpUnlock(void);
static void iree_symbol_resolver_initialize_mutex(
    iree_symbol_resolver_t* resolver) {
  IREEDbgHelpInit();
}
static void iree_symbol_resolver_lock(iree_symbol_resolver_t* resolver) {
  IREEDbgHelpLock();
}
static void iree_symbol_resolver_unlock(iree_symbol_resolver_t* resolver) {
  IREEDbgHelpUnlock();
}
#else
static void iree_symbol_resolver_initialize_mutex(
    iree_symbol_resolver_t* resolver) {
  InitializeSRWLock(&resolver->mutex);
}
static void iree_symbol_resolver_lock(iree_symbol_resolver_t* resolver) {
  AcquireSRWLockExclusive(&resolver->mutex);
}
static void iree_symbol_resolver_unlock(iree_symbol_resolver_t* resolver) {
  ReleaseSRWLockExclusive(&resolver->mutex);
}
#endif  // TRACY_ENABLE

static void iree_symbol_resolver_initialize(
    iree_symbol_resolver_t* out_resolver) {
  memset(out_resolver, 0, sizeof(*out_resolver));
  iree_symbol_resolver_initialize_mutex(out_resolver);
  out_resolver->library = LoadLibraryA("dbghelp.dll");
  if (!out_resolver->library) return;
  out_resolver->SymInitialize =
      (PFN_SymInitialize)GetProcAddress(out_resolver->library, "SymInitialize");
  out_resolver->SymCleanup =
      (PFN_SymCleanup)GetProcAddress(out_resolver->library, "SymCleanup");
  out_resolver->SymGetModuleInfo64 = (PFN_SymGetModuleInfo64)GetProcAddress(
      out_resolver->library, "SymGetModuleInfo64");
  out_resolver->SymFromAddr =
      (PFN_SymFromAddr)GetProcAddress(out_resolver->library, "SymFromAddr");
  out_resolver->SymGetLineFromAddr64 = (PFN_SymGetLineFromAddr64)GetProcAddress(
      out_resolver->library, "SymGetLineFromAddr64");
  if (!out_resolver->SymInitialize || !out_resolver->SymCleanup ||
      !out_resolver->SymGetModuleInfo64 || !out_resolver->SymFromAddr ||
      !out_resolver->SymGetLineFromAddr64) {
    FreeLibrary(out_resolver->library);
    memset(out_resolver, 0, sizeof(*out_resolver));
    return;
  }
  out_resolver->SymInitialize(GetCurrentProcess(), /*UserSearchPath=*/NULL,
                              /*fInvadeProcess=*/TRUE);
}

static void iree_symbol_resolver_deinitialize(
    iree_symbol_resolver_t* resolver) {
  if (resolver->SymCleanup) {
    resolver->SymCleanup(GetCurrentProcess());
  }
  if (resolver->library) {
    FreeLibrary(resolver->library);
    resolver->library = NULL;
  }
  memset(resolver, 0, sizeof(*resolver));
}

static BOOL CALLBACK iree_symbol_resolver_setup(PINIT_ONCE InitOnce,
                                                PVOID Parameter,
                                                PVOID* Context) {
  ((void)InitOnce);
  ((void)Context);
  iree_symbol_resolver_initialize((iree_symbol_resolver_t*)Parameter);
  return TRUE;
}

static INIT_ONCE instance_flag = INIT_ONCE_STATIC_INIT;
static iree_symbol_resolver_t instance;
static iree_symbol_resolver_t* iree_symbol_resolver_get(void) {
  InitOnceExecuteOnce(&instance_flag, iree_symbol_resolver_setup,
                      (PVOID)&instance, NULL);
  return &instance;
}

static bool iree_symbol_resolver_format_frame(iree_symbol_resolver_t* resolver,
                                              void* address,
                                              iree_host_size_t buffer_capacity,
                                              char* buffer,
                                              iree_host_size_t* buffer_length) {
  if (IREE_UNLIKELY(!resolver->library)) return false;

  HANDLE process = GetCurrentProcess();

  // Query generic information like what module the address is located in and
  // what the offset of the address is within its parent symbol.
  char symbol_buffer[512];
  SYMBOL_INFO* symbol_info = (SYMBOL_INFO*)symbol_buffer;
  symbol_info->SizeOfStruct = sizeof(*symbol_info);
  symbol_info->MaxNameLen = sizeof(symbol_buffer) - sizeof(*symbol_info);
  DWORD64 displacement64 = 0;
  iree_symbol_resolver_lock(resolver);
  BOOL result = resolver->SymFromAddr(process, (DWORD64)address,
                                      &displacement64, symbol_info);
  iree_symbol_resolver_unlock(resolver);
  if (!result) {
    // Failed to get any information; bail.
    return false;
  }

  IMAGEHLP_MODULE64 module;
  module.SizeOfStruct = sizeof(module);
  if (resolver->SymGetModuleInfo64(process, symbol_info->ModBase, &module)) {
    *buffer_length = iree_string_buffer_append_cstr(
        buffer_capacity, buffer, *buffer_length, module.ModuleName);
  } else {
    *buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                    *buffer_length, "???");
  }

  *buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                  *buffer_length, " <");
  if (symbol_info->NameLen > 0) {
    *buffer_length = iree_string_buffer_append_format(
        buffer_capacity, buffer, *buffer_length, "%.*s",
        (int)symbol_info->NameLen, symbol_info->Name);
  } else {
    *buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                    *buffer_length, "???");
  }
  if (displacement64) {
    *buffer_length = iree_string_buffer_append_format(
        buffer_capacity, buffer, *buffer_length, "+0x%0" PRIx64,
        displacement64);
  }
  *buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                  *buffer_length, ">");

  // Note that the returned name pointer is only valid while the lock is held.
  IMAGEHLP_LINE64 line;
  line.SizeOfStruct = sizeof(line);
  DWORD displacement32 = 0;
  iree_symbol_resolver_lock(resolver);
  if (resolver->SymGetLineFromAddr64(process, (DWORD64)address, &displacement32,
                                     &line)) {
    *buffer_length = iree_string_buffer_append_format(
        buffer_capacity, buffer, *buffer_length, " (%s:%" PRIu32 ")",
        line.FileName, (uint32_t)line.LineNumber);
  }
  iree_symbol_resolver_unlock(resolver);

  return true;
}

#endif  // IREE_PLATFORM_WINDOWS

static iree_host_size_t iree_status_payload_stack_trace_format_frame(
    void* address, iree_host_size_t buffer_capacity, char* buffer) {
  iree_host_size_t buffer_length = iree_string_buffer_append_format(
      buffer_capacity, buffer, 0, "  0x%016" PRIx64 " ",
      (uint64_t)(uintptr_t)address);
#if defined(IREE_HAVE_BACKTRACE)
  Dl_info info;
  if (dladdr(address, &info) != 0) {
    if (info.dli_fname) {
      iree_string_view_t fname = iree_status_trim_file_path(info.dli_fname);
      buffer_length = iree_string_buffer_append_format(
          buffer_capacity, buffer, buffer_length, "%.*s", (int)fname.size,
          fname.data);
    } else {
      buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                     buffer_length, "???");
    }
    buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                   buffer_length, " <");
    if (info.dli_sname) {
      iree_string_view_t sname = iree_make_cstring_view(info.dli_sname);
      buffer_length = iree_string_buffer_append_format(
          buffer_capacity, buffer, buffer_length, "%.*s", (int)sname.size,
          sname.data);
    } else {
      buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                     buffer_length, "???");
    }
    void* saddr = info.dli_saddr ? info.dli_saddr : address;
    ptrdiff_t diff = (ptrdiff_t)address - (ptrdiff_t)saddr;
    if (diff) {
      buffer_length = iree_string_buffer_append_format(
          buffer_capacity, buffer, buffer_length, "+0x%0" PRIx32,
          (int32_t)diff);
    }
    buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                   buffer_length, ">");
  } else {
    buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                   buffer_length, "???");
  }
#elif defined(IREE_PLATFORM_WINDOWS)
  if (!iree_symbol_resolver_format_frame(iree_symbol_resolver_get(), address,
                                         buffer_capacity, buffer,
                                         &buffer_length)) {
    buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                   buffer_length, "???");
  }
#else
  // Symbol resolution not implemented on the platform.
  buffer_length = iree_string_buffer_append_cstr(buffer_capacity, buffer,
                                                 buffer_length, "???");
#endif
  return iree_string_buffer_append_cstr(buffer_capacity, buffer, buffer_length,
                                        "\n");
}

static void iree_status_payload_stack_trace_formatter(
    const iree_status_payload_t* base_payload, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  iree_status_payload_stack_trace_t* payload =
      (iree_status_payload_stack_trace_t*)base_payload;
  if (payload->frame_count - payload->skip_frames == 0) return;
  iree_host_size_t buffer_length =
      iree_string_buffer_append_cstr(buffer_capacity, buffer, 0, "stack:\n");
  for (iree_host_size_t i = payload->skip_frames + 1; i < payload->frame_count;
       ++i) {
    buffer_length += iree_status_payload_stack_trace_format_frame(
        (void*)payload->addresses[i],
        buffer ? buffer_capacity - buffer_length : 0,
        buffer ? buffer + buffer_length : NULL);
    if (buffer_length > buffer_capacity) buffer = NULL;
  }
  *out_buffer_length = buffer_length;
}

// Captures the current stack and attaches it to the status storage.
// A count of |skip_frames| will be skipped from the top of the stack.
// Setting |skip_frames|=0 will include the caller in the stack while
// |skip_frames|=1 will exclude it.
iree_status_t iree_status_attach_stack_trace(iree_status_t status,
                                             iree_status_storage_t* storage,
                                             int skip_frames) {
  // Reserve storage for the number of stack frames so we can capture directly
  // into the storage even if we don't need them all. At the point we are
  // mallocing the exact size doesn't really matter.
  iree_status_payload_stack_trace_t* payload = NULL;
  iree_host_size_t total_size =
      sizeof(*payload) +
      sizeof(payload->addresses[0]) * IREE_STATUS_MAX_STACK_TRACE_FRAMES;

  iree_allocator_t allocator = iree_allocator_system();
  iree_status_ignore(
      iree_allocator_malloc(allocator, total_size, (void**)&payload));
  if (IREE_UNLIKELY(!payload)) return status;
  memset(payload, 0, sizeof(*payload));
  payload->header.type = IREE_STATUS_PAYLOAD_TYPE_STACK_TRACE;
  payload->header.allocator = allocator;
  payload->header.formatter = iree_status_payload_stack_trace_formatter;

#if defined(IREE_HAVE_BACKTRACE)
  // Capture up to the max frame count and skip some frames when formatting -
  // this means that our actual backtrace() max frame count is smaller than the
  // defined value. We could instead overallocate by skip_frames to waste a bit
  // of memory but keep the processing simpler.
  payload->skip_frames = skip_frames;
  payload->frame_count = backtrace((void**)&payload->addresses,
                                   IREE_STATUS_MAX_STACK_TRACE_FRAMES);
#elif defined(IREE_PLATFORM_WINDOWS)
  // NOTE: Win32 supports skip frames by default so we don't lose any storage.
  payload->frame_count =
      CaptureStackBackTrace(skip_frames, IREE_STATUS_MAX_STACK_TRACE_FRAMES,
                            (void**)&payload->addresses, NULL);
#endif

  return iree_status_append_payload(status, storage,
                                    (iree_status_payload_t*)payload);
}

#else

iree_status_t iree_status_attach_stack_trace(iree_status_t status,
                                             iree_status_storage_t* storage,
                                             int skip_frames) {
  return status;
}

#endif  // has IREE_STATUS_FEATURE_STACK_TRACE
