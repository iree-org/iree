// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Platform-abstracted shared memory primitives.
//
// Provides create/open/close for shared memory regions with handle passing
// support for cross-process sharing. Regions are always mapped read-write.
//
// Platform implementations:
//   Linux:   memfd_create (anonymous) / shm_open (named) + mmap
//   macOS:   shm_open + mmap (shm_unlink for anonymous)
//   Windows: CreateFileMappingW + MapViewOfFile
//
// Anonymous regions have no filesystem-visible name and are shared only by
// passing the handle (fd or HANDLE) to another process via IPC. Named regions
// are visible to any process that knows the name.
//
// All sizes are rounded up to the system page size. Use iree_shm_required_size
// to query the actual allocation size before creating.

#ifndef IREE_BASE_INTERNAL_SHM_H_
#define IREE_BASE_INTERNAL_SHM_H_

#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Platform handle for sharing shared memory between processes.
//
// On POSIX this is a file descriptor. On Windows this is a HANDLE.
// Handles are opaque and must not be interpreted by callers; use
// iree_shm_handle_dup to duplicate and iree_shm_handle_close to release.
typedef struct iree_shm_handle_t {
  uint64_t value;
} iree_shm_handle_t;

// Sentinel value indicating an invalid or closed handle.
#if defined(IREE_PLATFORM_WINDOWS)
// INVALID_HANDLE_VALUE is (HANDLE)(LONG_PTR)-1 on Windows.
#define IREE_SHM_HANDLE_INVALID ((iree_shm_handle_t){(uint64_t)(uintptr_t)-1})
#else
#define IREE_SHM_HANDLE_INVALID ((iree_shm_handle_t){(uint64_t)-1})
#endif  // IREE_PLATFORM_WINDOWS

// Returns true if the handle is valid (not the invalid sentinel).
static inline bool iree_shm_handle_is_valid(iree_shm_handle_t handle) {
  return handle.value != IREE_SHM_HANDLE_INVALID.value;
}

// Maximum length of a shared memory name in bytes (excluding NUL terminator).
// Matches POSIX NAME_MAX (255), which is the most restrictive platform limit.
// Windows kernel object names can be much longer (~32K wide chars) but we cap
// at the POSIX limit for portability.
#define IREE_SHM_MAX_NAME_LENGTH 255

// Options controlling shared memory creation and opening.
//
// Initialize with iree_shm_options_default() to get safe defaults. Pass by
// value to all create and open functions.
typedef struct iree_shm_options_t {
  int reserved;
} iree_shm_options_t;

// Returns default shared memory options. All fields are zero-initialized.
static inline iree_shm_options_t iree_shm_options_default(void) {
  iree_shm_options_t options;
  memset(&options, 0, sizeof(options));
  return options;
}

// A mapped shared memory region.
//
// Contains the mapped address, actual size (page-aligned), and a handle that
// can be duplicated for sharing with other processes. The mapping is valid
// until iree_shm_close is called.
typedef struct iree_shm_mapping_t {
  // Base address of the mapped region, or NULL if not mapped.
  void* base;
  // Actual mapped size in bytes (page-aligned, >= the requested size).
  iree_host_size_t size;
  // Platform handle for sharing this region with other processes.
  iree_shm_handle_t handle;
} iree_shm_mapping_t;

// Returns the actual allocation size for a given requested size.
//
// The result is always a multiple of the system page size and is always at
// least one page. Use this to determine the true size of a region before
// creating it, or to compute the expected size when opening.
iree_host_size_t iree_shm_required_size(iree_host_size_t requested_size);

// Creates an anonymous shared memory region of at least |minimum_size| bytes.
//
// The region has no filesystem-visible name. Sharing requires passing the
// handle from |out_mapping| to another process via IPC (Unix domain sockets,
// inherited file descriptors, DuplicateHandle, etc.).
//
// The actual mapped size may be larger than |minimum_size| due to page
// alignment. The caller can write to [base, base + size) immediately.
//
// |minimum_size| must be > 0. Returns IREE_STATUS_INVALID_ARGUMENT for 0.
iree_status_t iree_shm_create(iree_shm_options_t options,
                              iree_host_size_t minimum_size,
                              iree_shm_mapping_t* out_mapping);

// Creates a named shared memory region of at least |minimum_size| bytes.
//
// The |name| identifies the region for opening by other processes. On POSIX
// this maps to a shm_open name (must start with '/'). On Windows this maps to
// a named file mapping in the Local namespace.
//
// Returns IREE_STATUS_ALREADY_EXISTS if a region with this name already exists.
// Use iree_shm_open_named to attach to an existing named region.
//
// |minimum_size| must be > 0. Returns IREE_STATUS_INVALID_ARGUMENT for 0.
iree_status_t iree_shm_create_named(iree_string_view_t name,
                                    iree_shm_options_t options,
                                    iree_host_size_t minimum_size,
                                    iree_shm_mapping_t* out_mapping);

// Opens and maps an existing shared memory region from a handle.
//
// The |handle| must be a valid handle obtained via iree_shm_handle_dup or
// platform-specific IPC mechanisms. The handle is NOT consumed; the caller
// retains ownership and must close it separately if desired.
//
// |size| specifies the mapping size and must match the original region's size
// (as returned by iree_shm_required_size of the original minimum_size).
iree_status_t iree_shm_open_handle(iree_shm_handle_t handle,
                                   iree_shm_options_t options,
                                   iree_host_size_t size,
                                   iree_shm_mapping_t* out_mapping);

// Opens and maps an existing named shared memory region.
//
// The |name| must match a region previously created with iree_shm_create_named
// that has not been unlinked.
//
// |size| specifies the mapping size and must match the original region's size.
//
// Returns IREE_STATUS_NOT_FOUND if no region with this name exists.
iree_status_t iree_shm_open_named(iree_string_view_t name,
                                  iree_shm_options_t options,
                                  iree_host_size_t size,
                                  iree_shm_mapping_t* out_mapping);

// Unmaps the shared memory region and closes its handle.
//
// After this call, |mapping| is zeroed and the base pointer is invalid. Other
// processes that have mapped the same region are unaffected.
//
// Safe to call on a zeroed/default-initialized mapping (no-op).
void iree_shm_close(iree_shm_mapping_t* mapping);

// Duplicates a shared memory handle for transfer to another process.
//
// The returned handle in |out_handle| is independent of |source| and must be
// closed separately via iree_shm_handle_close. This is used to prepare a
// handle for IPC transfer (e.g., sending over a Unix domain socket or passing
// to DuplicateHandle on Windows).
iree_status_t iree_shm_handle_dup(iree_shm_handle_t source,
                                  iree_shm_handle_t* out_handle);

// Closes a standalone shared memory handle.
//
// Use this for handles obtained from iree_shm_handle_dup that are not
// associated with a mapping. For handles that are part of an iree_shm_mapping_t
// use iree_shm_close instead (which closes the handle along with the mapping).
//
// Safe to call on an invalid handle (no-op). Sets |handle->value| to the
// invalid sentinel after closing.
void iree_shm_handle_close(iree_shm_handle_t* handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_SHM_H_
