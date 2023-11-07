// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/status.h"

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
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/allocator.h"
#include "iree/base/assert.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// C11 aligned_alloc compatibility shim
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc
#define iree_aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define iree_aligned_free(p) _aligned_free(p)
#elif defined(_ISOC11_SOURCE)
// https://en.cppreference.com/w/c/memory/aligned_alloc
#define iree_aligned_alloc(alignment, size) aligned_alloc(alignment, size)
#define iree_aligned_free(p) free(p)
#elif _POSIX_C_SOURCE >= 200112L
// https://pubs.opengroup.org/onlinepubs/9699919799/functions/posix_memalign.html
IREE_ATTRIBUTE_UNUSED static inline void* iree_aligned_alloc(size_t alignment,
                                                             size_t size) {
  void* ptr = NULL;
  return posix_memalign(&ptr, alignment, size) == 0 ? ptr : NULL;
}
#define iree_aligned_free(p) free(p)
#else
// Emulates alignment with normal malloc. We overallocate by at least the
// alignment + the size of a pointer, store the base pointer at p[-1], and
// return the aligned pointer. This lets us easily get the base pointer in free
// to pass back to the system.
static inline void* iree_aligned_alloc(size_t alignment, size_t size) {
  void* base_ptr = malloc(size + alignment + sizeof(uintptr_t));
  if (!base_ptr) return NULL;
  uintptr_t* aligned_ptr = (uintptr_t*)iree_host_align(
      (uintptr_t)base_ptr + sizeof(uintptr_t), alignment);
  aligned_ptr[-1] = (uintptr_t)base_ptr;
  return aligned_ptr;
}
static inline void iree_aligned_free(void* p) {
  if (IREE_UNLIKELY(!p)) return;
  uintptr_t* aligned_ptr = (uintptr_t*)p;
  void* base_ptr = (void*)aligned_ptr[-1];
  free(base_ptr);
}
#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// iree_status_t canonical errors
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_code_t
iree_status_code_from_errno(int error_number) {
  switch (error_number) {
    case 0:
      return IREE_STATUS_OK;
    case EINVAL:        // Invalid argument
    case ENAMETOOLONG:  // Filename too long
    case E2BIG:         // Argument list too long
    case EDESTADDRREQ:  // Destination address required
    case EDOM:          // Mathematics argument out of domain of function
    case EFAULT:        // Bad address
    case EILSEQ:        // Illegal byte sequence
    case ENOPROTOOPT:   // Protocol not available
    case ENOSTR:        // Not a STREAM
    case ENOTSOCK:      // Not a socket
    case ENOTTY:        // Inappropriate I/O control operation
    case EPROTOTYPE:    // Protocol wrong type for socket
    case ESPIPE:        // Invalid seek
      return IREE_STATUS_INVALID_ARGUMENT;
    case ETIMEDOUT:  // Connection timed out
    case ETIME:      // Timer expired
      return IREE_STATUS_DEADLINE_EXCEEDED;
    case ENODEV:  // No such device
    case ENOENT:  // No such file or directory
#ifdef ENOMEDIUM
    case ENOMEDIUM:  // No medium found
#endif
    case ENXIO:  // No such device or address
    case ESRCH:  // No such process
      return IREE_STATUS_NOT_FOUND;
    case EEXIST:         // File exists
    case EADDRNOTAVAIL:  // Address not available
    case EALREADY:       // Connection already in progress
#ifdef ENOTUNIQ
    case ENOTUNIQ:  // Name not unique on network
#endif
      return IREE_STATUS_ALREADY_EXISTS;
    case EPERM:   // Operation not permitted
    case EACCES:  // Permission denied
#ifdef ENOKEY
    case ENOKEY:  // Required key not available
#endif
    case EROFS:  // Read only file system
      return IREE_STATUS_PERMISSION_DENIED;
    case ENOTEMPTY:   // Directory not empty
    case EISDIR:      // Is a directory
    case ENOTDIR:     // Not a directory
    case EADDRINUSE:  // Address already in use
    case EBADF:       // Invalid file descriptor
#ifdef EBADFD
    case EBADFD:  // File descriptor in bad state
#endif
    case EBUSY:    // Device or resource busy
    case ECHILD:   // No child processes
    case EISCONN:  // Socket is connected
#ifdef EISNAM
    case EISNAM:  // Is a named type file
#endif
#ifdef ENOTBLK
    case ENOTBLK:  // Block device required
#endif
    case ENOTCONN:  // The socket is not connected
    case EPIPE:     // Broken pipe
#ifdef ESHUTDOWN
    case ESHUTDOWN:  // Cannot send after transport endpoint shutdown
#endif
    case ETXTBSY:  // Text file busy
#ifdef EUNATCH
    case EUNATCH:  // Protocol driver not attached
#endif
      return IREE_STATUS_FAILED_PRECONDITION;
    case ENOSPC:  // No space left on device
#ifdef EDQUOT
    case EDQUOT:  // Disk quota exceeded
#endif
    case EMFILE:   // Too many open files
    case EMLINK:   // Too many links
    case ENFILE:   // Too many open files in system
    case ENOBUFS:  // No buffer space available
    case ENODATA:  // No message is available on the STREAM read queue
    case ENOMEM:   // Not enough space
    case ENOSR:    // No STREAM resources
#ifdef EUSERS
    case EUSERS:  // Too many users
#endif
      return IREE_STATUS_RESOURCE_EXHAUSTED;
#ifdef ECHRNG
    case ECHRNG:  // Channel number out of range
#endif
    case EFBIG:      // File too large
    case EOVERFLOW:  // Value too large to be stored in data type
    case ERANGE:     // Result too large
      return IREE_STATUS_OUT_OF_RANGE;
#ifdef ENOPKG
    case ENOPKG:  // Package not installed
#endif
    case ENOSYS:        // Function not implemented
    case ENOTSUP:       // Operation not supported
    case EAFNOSUPPORT:  // Address family not supported
#ifdef EPFNOSUPPORT
    case EPFNOSUPPORT:  // Protocol family not supported
#endif
    case EPROTONOSUPPORT:  // Protocol not supported
#ifdef ESOCKTNOSUPPORT
    case ESOCKTNOSUPPORT:  // Socket type not supported
#endif
    case EXDEV:  // Improper link
      return IREE_STATUS_UNIMPLEMENTED;
    case EAGAIN:  // Resource temporarily unavailable
#ifdef ECOMM
    case ECOMM:  // Communication error on send
#endif
    case ECONNREFUSED:  // Connection refused
    case ECONNABORTED:  // Connection aborted
    case ECONNRESET:    // Connection reset
    case EINTR:         // Interrupted function call
#ifdef EHOSTDOWN
    case EHOSTDOWN:  // Host is down
#endif
    case EHOSTUNREACH:  // Host is unreachable
    case ENETDOWN:      // Network is down
    case ENETRESET:     // Connection aborted by network
    case ENETUNREACH:   // Network unreachable
    case ENOLCK:        // No locks available
    case ENOLINK:       // Link has been severed
#ifdef ENONET
    case ENONET:  // Machine is not on the network
#endif
      return IREE_STATUS_UNAVAILABLE;
    case EDEADLK:  // Resource deadlock avoided
#ifdef ESTALE
    case ESTALE:  // Stale file handle
#endif
      return IREE_STATUS_ABORTED;
    case ECANCELED:  // Operation cancelled
      return IREE_STATUS_CANCELLED;
    default:
      return IREE_STATUS_UNKNOWN;
  }
}

#if defined(IREE_PLATFORM_WINDOWS)
IREE_API_EXPORT iree_status_code_t
iree_status_code_from_win32_error(uint32_t error) {
  switch (error) {
    case ERROR_SUCCESS:
      return IREE_STATUS_OK;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      return IREE_STATUS_NOT_FOUND;
    case ERROR_TOO_MANY_OPEN_FILES:
    case ERROR_OUTOFMEMORY:
    case ERROR_HANDLE_DISK_FULL:
    case ERROR_HANDLE_EOF:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
    case ERROR_ACCESS_DENIED:
      return IREE_STATUS_PERMISSION_DENIED;
    case ERROR_INVALID_HANDLE:
    case ERROR_INVALID_PARAMETER:
      return IREE_STATUS_INVALID_ARGUMENT;
    case ERROR_NOT_READY:
    case ERROR_READ_FAULT:
      return IREE_STATUS_UNAVAILABLE;
    case ERROR_WRITE_FAULT:
      return IREE_STATUS_DATA_LOSS;
    case ERROR_NOT_SUPPORTED:
      return IREE_STATUS_UNIMPLEMENTED;
    default:
      return IREE_STATUS_UNKNOWN;
  }
}
#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// iree_status_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT const char* iree_status_code_string(iree_status_code_t code) {
  switch (code) {
    case IREE_STATUS_OK:
      return "OK";
    case IREE_STATUS_CANCELLED:
      return "CANCELLED";
    case IREE_STATUS_UNKNOWN:
      return "UNKNOWN";
    case IREE_STATUS_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case IREE_STATUS_DEADLINE_EXCEEDED:
      return "DEADLINE_EXCEEDED";
    case IREE_STATUS_NOT_FOUND:
      return "NOT_FOUND";
    case IREE_STATUS_ALREADY_EXISTS:
      return "ALREADY_EXISTS";
    case IREE_STATUS_PERMISSION_DENIED:
      return "PERMISSION_DENIED";
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    case IREE_STATUS_FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case IREE_STATUS_ABORTED:
      return "ABORTED";
    case IREE_STATUS_OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case IREE_STATUS_UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case IREE_STATUS_INTERNAL:
      return "INTERNAL";
    case IREE_STATUS_UNAVAILABLE:
      return "UNAVAILABLE";
    case IREE_STATUS_DATA_LOSS:
      return "DATA_LOSS";
    case IREE_STATUS_UNAUTHENTICATED:
      return "UNAUTHENTICATED";
    case IREE_STATUS_DEFERRED:
      return "DEFERRED";
    default:
      return "";
  }
}

// TODO(#55): move payload methods/types to header when API is stabilized.

struct iree_status_handle_t {
  uintptr_t value;
};

// Defines the type of an iree_status_payload_t.
typedef enum iree_status_payload_type_e {
  // Opaque; payload may still be formatted by a formatter but is not possible
  // to retrieve by the programmatic APIs.
  IREE_STATUS_PAYLOAD_TYPE_OPAQUE = 0,
  // A string message annotation of type iree_status_payload_message_t.
  IREE_STATUS_PAYLOAD_TYPE_MESSAGE = 1,
  // Platform-dependent stack trace in iree_status_payload_stack_trace_t.
  IREE_STATUS_PAYLOAD_TYPE_STACK_TRACE = 2,
  // Starting type ID for user payloads. IREE reserves all payloads with types
  // less than this.
  IREE_STATUS_PAYLOAD_TYPE_MIN_USER = 0x70000000u,
} iree_status_payload_type_t;

typedef struct iree_status_payload_t iree_status_payload_t;

// Function that formats a payload into a human-readable string form for logs.
typedef void(IREE_API_PTR* iree_status_payload_formatter_t)(
    const iree_status_payload_t* payload, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

// Header for optional status payloads.
// Each status may have zero or more payloads associated with it that can later
// be used to produce more detailed logging or programmatically query
// information about an error.
struct iree_status_payload_t {
  // Next payload in the status payload linked list.
  struct iree_status_payload_t* next;
  // Payload type identifier used for programmatic access to payloads. May be
  // IREE_STATUS_PAYLOAD_TYPE_OPAQUE if the payload cannot be accessed directly.
  iree_status_payload_type_t type;
  // Allocator used for the payload and associated resources.
  iree_allocator_t allocator;
  // String formatter callback used to write the payload into a string buffer.
  // If not present then the payload will be mentioned but not dumped when the
  // status is logged.
  iree_status_payload_formatter_t formatter;
};

// A string message (IREE_STATUS_PAYLOAD_TYPE_MESSAGE).
typedef struct iree_status_payload_message_t {
  iree_status_payload_t header;
  // String data reference. May point to an address immediately following this
  // struct (if copied) or a constant string reference in rodata.
  iree_string_view_t message;
} iree_status_payload_message_t;

// A platform-dependent stack trace (IREE_STATUS_PAYLOAD_TYPE_STACK_TRACE).
typedef struct iree_status_payload_stack_trace_t {
  iree_status_payload_t header;
  uint16_t skip_frames;
  uint16_t frame_count;
  uintptr_t addresses[];
} iree_status_payload_stack_trace_t;

// Allocated storage for an iree_status_t.
// Only statuses that have either source information or payloads will have
// storage allocated for them.
typedef struct iree_status_storage_t {
  // Optional doubly-linked list of payloads associated with the status.
  // Head = first added, tail = last added.
  iree_status_payload_t* payload_head;
  iree_status_payload_t* payload_tail;

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  // __FILE__ of the originating status allocation.
  const char* file;
  // __LINE__ of the originating status allocation.
  uint32_t line;
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  // Optional message that is allocated either as a constant string in rodata or
  // present as a suffix on the storage.
  iree_string_view_t message;
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
} iree_status_storage_t;

#define iree_status_storage(status) \
  ((iree_status_storage_t*)(((uintptr_t)(status) & ~IREE_STATUS_CODE_MASK)))

// Appends a payload to the storage doubly-linked list.
IREE_ATTRIBUTE_UNUSED static iree_status_t iree_status_append_payload(
    iree_status_t status, iree_status_storage_t* storage,
    iree_status_payload_t* payload) {
  if (!storage->payload_tail) {
    storage->payload_head = payload;
  } else {
    storage->payload_tail->next = payload;
  }
  storage->payload_tail = payload;
  return status;
}

// Formats an iree_status_payload_message_t to the given output |buffer|.
// |out_buffer_length| will be set to the number of characters written excluding
// NUL. If |buffer| is omitted then |out_buffer_length| will be set to the
// total number of characters in |buffer_capacity| required to contain the
// entire message.
IREE_ATTRIBUTE_UNUSED static void iree_status_payload_message_formatter(
    const iree_status_payload_t* base_payload, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  iree_status_payload_message_t* payload =
      (iree_status_payload_message_t*)base_payload;
  if (!buffer) {
    *out_buffer_length = payload->message.size;
    return;
  }
  iree_host_size_t n = buffer_capacity < payload->message.size
                           ? buffer_capacity
                           : payload->message.size;
  memcpy(buffer, payload->message.data, n);
  buffer[n] = '\0';
  *out_buffer_length = n;
}

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
static iree_status_t iree_status_attach_stack_trace(
    iree_status_t status, iree_status_storage_t* storage, int skip_frames) {
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

IREE_ATTRIBUTE_UNUSED static iree_status_t iree_status_attach_stack_trace(
    iree_status_t status, iree_status_storage_t* storage, int skip_frames) {
  return status;
}

#endif  // has IREE_STATUS_FEATURE_STACK_TRACE

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_status_allocate(iree_status_code_t code, const char* file, uint32_t line,
                     iree_string_view_t message) {
#if IREE_STATUS_FEATURES == 0
  // More advanced status code features like source location and messages are
  // disabled. All statuses are just the codes.
  return iree_status_from_code(code);
#else
  // No-op for OK statuses; we won't get these from the macros but may be called
  // with this from marshaling code.
  if (IREE_UNLIKELY(code == IREE_STATUS_OK)) return iree_ok_status();

  // Allocate storage with the appropriate alignment such that we can pack the
  // code in the lower bits of the pointer. Since failed statuses are rare and
  // likely have much larger costs (like string formatting) the extra bytes for
  // alignment are worth being able to avoid pointer dereferences and other
  // things during the normal code paths that just check codes.
  //
  // Note that we are using the CRT allocation function here, as we can't trust
  // our allocator system to work when we are throwing errors (as we may be
  // allocating this error from a failed allocation!).
  size_t storage_alignment = (IREE_STATUS_CODE_MASK + 1);
  size_t storage_size =
      iree_host_align(sizeof(iree_status_storage_t), storage_alignment);
  iree_status_storage_t* storage = (iree_status_storage_t*)iree_aligned_alloc(
      storage_alignment, storage_size);
  if (IREE_UNLIKELY(!storage)) return iree_status_from_code(code);
  memset(storage, 0, sizeof(*storage));

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  storage->file = file;
  storage->line = line;
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  // NOTE: messages are rodata strings here and not retained.
  storage->message = message;
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

  return iree_status_attach_stack_trace(
      (iree_status_t)((uintptr_t)storage | (code & IREE_STATUS_CODE_MASK)),
      storage, /*skip_frames=*/1);
#endif  // has any IREE_STATUS_FEATURES
}

IREE_MUST_USE_RESULT static iree_status_t iree_status_allocate_vf_impl(
    iree_status_code_t code, const char* file, uint32_t line, int skip_frames,
    const char* format, va_list varargs_0, va_list varargs_1) {
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
  // Annotations disabled; ignore the format string/args.
  return iree_status_allocate(code, file, line, iree_string_view_empty());
#else
  // No-op for OK statuses; we won't get these from the macros but may be called
  // with this from marshaling code.
  if (IREE_UNLIKELY(code == IREE_STATUS_OK)) return iree_ok_status();

  // Compute the total number of bytes (including NUL) required to store the
  // message.
  int message_size =
      vsnprintf(/*buffer=*/NULL, /*buffer_count=*/0, format, varargs_0);
  if (message_size < 0) return iree_status_from_code(code);
  ++message_size;  // NUL byte

  // Allocate storage with the additional room to store the formatted message.
  // This avoids additional allocations for the common case of a message coming
  // only from the original status error site.
  size_t storage_alignment = (IREE_STATUS_CODE_MASK + 1);
  size_t storage_size = iree_host_align(
      sizeof(iree_status_storage_t) + message_size, storage_alignment);
  iree_status_storage_t* storage = (iree_status_storage_t*)iree_aligned_alloc(
      storage_alignment, storage_size);
  if (IREE_UNLIKELY(!storage)) return iree_status_from_code(code);
  memset(storage, 0, sizeof(*storage));

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  storage->file = file;
  storage->line = line;
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION

  // vsnprintf directly into message buffer.
  storage->message.size = message_size - 1;
  storage->message.data = (const char*)storage + sizeof(iree_status_storage_t);
  int ret =
      vsnprintf((char*)storage->message.data, message_size, format, varargs_1);
  if (IREE_UNLIKELY(ret < 0)) {
    iree_aligned_free(storage);
    return (iree_status_t)code;
  }

  return iree_status_attach_stack_trace(
      (iree_status_t)((uintptr_t)storage | (code & IREE_STATUS_CODE_MASK)),
      storage, skip_frames);
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_status_allocate_f(iree_status_code_t code, const char* file, uint32_t line,
                       const char* format, ...) {
  va_list varargs_0, varargs_1;
  va_start(varargs_0, format);
  va_start(varargs_1, format);
  iree_status_t ret = iree_status_allocate_vf_impl(
      code, file, line, /*skip_frames=*/2, format, varargs_0, varargs_1);
  va_end(varargs_0);
  va_end(varargs_1);
  return ret;
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t iree_status_allocate_vf(
    iree_status_code_t code, const char* file, uint32_t line,
    const char* format, va_list varargs_0, va_list varargs_1) {
  return iree_status_allocate_vf_impl(code, file, line, /*skip_frames=*/1,
                                      format, varargs_0, varargs_1);
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_status_clone(iree_status_t status) {
#if IREE_STATUS_FEATURES == 0
  // Statuses are just codes; nothing to do.
  return status;
#else
  iree_status_storage_t* storage = iree_status_storage(status);
  if (!storage) return status;

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  const char* file = storage->file;
  uint32_t line = storage->line;
#else
  const char* file = NULL;
  uint32_t line = 0;
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  iree_string_view_t message = storage->message;
#else
  iree_string_view_t message = iree_string_view_empty();
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

  // Always copy the message by performing the formatting as we don't know
  // whether the original status has ownership or not.
  return iree_status_allocate_f(iree_status_code(status), file, line, "%.*s",
                                (int)message.size, message.data);
#endif  // has no IREE_STATUS_FEATURES
}

IREE_API_EXPORT void iree_status_free(iree_status_t status) {
#if IREE_STATUS_FEATURES != 0
  iree_status_storage_t* storage = iree_status_storage(status);
  if (!storage) return;
  iree_status_payload_t* payload = storage->payload_head;
  while (payload) {
    iree_status_payload_t* next = payload->next;
    iree_allocator_free(payload->allocator, payload);
    payload = next;
  }
  iree_aligned_free(storage);
#endif  // has any IREE_STATUS_FEATURES
}

IREE_API_EXPORT iree_status_t iree_status_ignore(iree_status_t status) {
  // We can set an 'ignored' flag on the status so that we can otherwise assert
  // in iree_status_free when statuses are freed without this being called.
  // Hoping with the C++ Status wrapper we won't hit that often so that
  // complexity is skipped for now.
  iree_status_free(status);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_status_join(iree_status_t base_status,
                                               iree_status_t new_status) {
  // TODO(benvanik): annotate |base_status| with |new_status| so we see it?
  // This is intended for failure handling and usually the first failure is the
  // root cause and most important to see.
  if (!iree_status_is_ok(base_status)) {
    iree_status_ignore(new_status);
    return base_status;
  }
  return new_status;
}

IREE_API_EXPORT IREE_ATTRIBUTE_NORETURN void iree_status_abort(
    iree_status_t status) {
  iree_status_fprint(stderr, status);
  IREE_ASSERT(!iree_status_is_ok(status),
              "only valid to call with failing status codes");
  iree_status_free(status);
  iree_abort();
}

IREE_API_EXPORT iree_status_code_t
iree_status_consume_code(iree_status_t status) {
  iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);
  return code;
}

#if IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t
iree_status_annotate(iree_status_t base_status, iree_string_view_t message) {
  if (iree_status_is_ok(base_status) || iree_string_view_is_empty(message)) {
    return base_status;
  }

  // If there's no storage yet we can just reuse normal allocation. Both that
  // and this do not copy |message|.
  iree_status_storage_t* storage = iree_status_storage(base_status);
  if (!storage) {
    return iree_status_allocate(iree_status_code(base_status), NULL, 0,
                                message);
  } else if (iree_string_view_is_empty(storage->message)) {
    storage->message = message;
    return base_status;
  }

  iree_allocator_t allocator = iree_allocator_system();
  iree_status_payload_message_t* payload = NULL;
  iree_status_ignore(
      iree_allocator_malloc(allocator, sizeof(*payload), (void**)&payload));
  if (IREE_UNLIKELY(!payload)) return base_status;
  memset(payload, 0, sizeof(*payload));
  payload->header.type = IREE_STATUS_PAYLOAD_TYPE_MESSAGE;
  payload->header.allocator = allocator;
  payload->header.formatter = iree_status_payload_message_formatter;
  payload->message = message;
  return iree_status_append_payload(base_status, storage,
                                    (iree_status_payload_t*)payload);
}

IREE_MUST_USE_RESULT static iree_status_t iree_status_annotate_vf(
    iree_status_t base_status, const char* format, va_list varargs_0,
    va_list varargs_1) {
  if (iree_status_is_ok(base_status)) return base_status;

  // If there's no storage yet we can just reuse normal allocation. Both that
  // and this do not copy |message|.
  iree_status_storage_t* storage = iree_status_storage(base_status);
  if (!storage) {
    return iree_status_allocate_vf(iree_status_code(base_status), NULL, 0,
                                   format, varargs_0, varargs_1);
  }

  // Compute the total number of bytes (including NUL) required to store the
  // message.
  int message_size =
      vsnprintf(/*buffer=*/NULL, /*buffer_count=*/0, format, varargs_0);
  if (message_size < 0) return base_status;
  ++message_size;  // NUL byte

  // Allocate storage with the additional room to store the formatted message.
  // This avoids additional allocations for the common case of a message coming
  // only from the original status error site.
  iree_allocator_t allocator = iree_allocator_system();
  iree_status_payload_message_t* payload = NULL;
  iree_status_ignore(iree_allocator_malloc(
      allocator, sizeof(*payload) + message_size, (void**)&payload));
  if (IREE_UNLIKELY(!payload)) return base_status;
  memset(payload, 0, sizeof(*payload));
  payload->header.type = IREE_STATUS_PAYLOAD_TYPE_MESSAGE;
  payload->header.allocator = allocator;
  payload->header.formatter = iree_status_payload_message_formatter;

  // vsnprintf directly into message buffer.
  payload->message.size = message_size - 1;
  payload->message.data =
      (const char*)payload + sizeof(iree_status_payload_message_t);
  int ret = vsnprintf((char*)payload->message.data, payload->message.size + 1,
                      format, varargs_1);
  if (IREE_UNLIKELY(ret < 0)) {
    iree_aligned_free(payload);
    return base_status;
  }
  return iree_status_append_payload(base_status, storage,
                                    (iree_status_payload_t*)payload);
}

IREE_API_EXPORT IREE_MUST_USE_RESULT iree_status_t IREE_PRINTF_ATTRIBUTE(2, 3)
    iree_status_annotate_f(iree_status_t base_status, const char* format, ...) {
  // We walk the lists twice as each va_list can only be walked once we need to
  // double-up. iree_status_annotate_vf could use va_copy to clone the single
  // list however the proper management of va_end is trickier and this works.
  va_list varargs_0, varargs_1;
  va_start(varargs_0, format);
  va_start(varargs_1, format);
  iree_status_t ret =
      iree_status_annotate_vf(base_status, format, varargs_0, varargs_1);
  va_end(varargs_0);
  va_end(varargs_1);
  return ret;
}

#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

IREE_API_EXPORT bool iree_status_format(iree_status_t status,
                                        iree_host_size_t buffer_capacity,
                                        char* buffer,
                                        iree_host_size_t* out_buffer_length) {
  *out_buffer_length = 0;

  // Grab storage which may have a message and zero or more payloads.
  iree_status_storage_t* storage IREE_ATTRIBUTE_UNUSED =
      iree_status_storage(status);

  // Prefix with source location and status code string (may be 'OK').
  iree_host_size_t buffer_length = 0;
  iree_status_code_t status_code = iree_status_code(status);
  int n = 0;
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_SOURCE_LOCATION) != 0
  if (storage && storage->file) {
    n = snprintf(buffer ? buffer + buffer_length : NULL,
                 buffer ? buffer_capacity - buffer_length : 0, "%s:%d: %s",
                 storage->file, storage->line,
                 iree_status_code_string(status_code));
  } else {
    n = snprintf(buffer ? buffer + buffer_length : NULL,
                 buffer ? buffer_capacity - buffer_length : 0, "%s",
                 iree_status_code_string(status_code));
  }
#else
  n = snprintf(buffer ? buffer + buffer_length : NULL,
               buffer ? buffer_capacity - buffer_length : 0, "%s",
               iree_status_code_string(status_code));
#endif  // has IREE_STATUS_FEATURE_SOURCE_LOCATION
  if (IREE_UNLIKELY(n < 0)) {
    return false;
  } else if (buffer && n >= buffer_capacity - buffer_length) {
    buffer = NULL;
  }
  buffer_length += n;

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  // Append base storage message.
  if (storage && !iree_string_view_is_empty(storage->message)) {
    n = snprintf(buffer ? buffer + buffer_length : NULL,
                 buffer ? buffer_capacity - buffer_length : 0, "; %.*s",
                 (int)storage->message.size, storage->message.data);
    if (IREE_UNLIKELY(n < 0)) {
      return false;
    } else if (buffer && n >= buffer_capacity - buffer_length) {
      buffer = NULL;
    }
    buffer_length += n;
  }
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS

#if IREE_STATUS_FEATURES != 0
  // Append each payload separated by a newline.
  iree_status_payload_t* payload = storage ? storage->payload_head : NULL;
  while (payload != NULL) {
    // Skip payloads that have no textual representation.
    if (!payload->formatter) {
      payload = payload->next;
      continue;
    }

    // Append newline to join with message above and other payloads.
    if (buffer) {
      if (2 >= buffer_capacity - buffer_length) {
        buffer = NULL;
      } else {
        buffer[buffer_length] = ';';
        buffer[buffer_length + 1] = ' ';
        buffer[buffer_length + 2] = '\0';
      }
    }
    buffer_length += 2;  // '; '

    // Append payload via custom formatter callback.
    iree_host_size_t payload_buffer_length = 0;
    payload->formatter(payload, buffer ? buffer_capacity - buffer_length : 0,
                       buffer ? buffer + buffer_length : NULL,
                       &payload_buffer_length);
    if (buffer && payload_buffer_length >= buffer_capacity - buffer_length) {
      buffer = NULL;
    }
    buffer_length += payload_buffer_length;

    payload = payload->next;
  }
#endif  // has IREE_STATUS_FEATURES

  *out_buffer_length = buffer_length;
  return true;
}

IREE_API_EXPORT bool iree_status_to_string(
    iree_status_t status, const iree_allocator_t* allocator, char** out_buffer,
    iree_host_size_t* out_buffer_length) {
  *out_buffer_length = 0;
  iree_host_size_t buffer_length = 0;
  if (IREE_UNLIKELY(!iree_status_format(status, /*buffer_capacity=*/0,
                                        /*buffer=*/NULL, &buffer_length))) {
    return false;
  }

  // Buffer capacity needs to be +1 for the NUL terminator (see snprintf).
  char* buffer = NULL;
  iree_status_t malloc_status =
      iree_allocator_malloc(*allocator, buffer_length + 1, (void**)&buffer);
  if (!iree_status_is_ok(malloc_status)) {
    iree_status_ignore(malloc_status);
    return false;
  }
  bool ret =
      iree_status_format(status, buffer_length + 1, buffer, out_buffer_length);
  if (ret) {
    *out_buffer = buffer;
    return true;
  } else {
    iree_allocator_free(*allocator, buffer);
    return false;
  }
}

IREE_API_EXPORT void iree_status_fprint(FILE* file, iree_status_t status) {
  // TODO(benvanik): better support for colors/etc - possibly move to logging.
  // TODO(benvanik): do this without allocation by streaming the status.
  iree_allocator_t allocator = iree_allocator_system();
  char* status_buffer = NULL;
  iree_host_size_t status_buffer_length = 0;
  if (iree_status_to_string(status, &allocator, &status_buffer,
                            &status_buffer_length)) {
    fprintf(file, "%.*s\n", (int)status_buffer_length, status_buffer);
    iree_allocator_free(allocator, status_buffer);
  } else {
    fprintf(file, "(?)\n");
  }
  fflush(file);
}
