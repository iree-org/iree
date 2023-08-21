// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_FILE_H_
#define IREE_HAL_FILE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// A bitfield specifying how a file should be opened and the access allowed.
enum iree_hal_file_mode_bits_t {
  // Opens the file if it exists on the file system.
  IREE_HAL_FILE_MODE_OPEN = 1u << 0,
};
typedef uint32_t iree_hal_file_mode_t;

typedef void(IREE_API_PTR* iree_hal_file_release_fn_t)(void* user_data);

// A callback issued when a file is released.
typedef struct {
  // Callback function pointer.
  iree_hal_file_release_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_hal_file_release_callback_t;

// Returns a no-op file release callback that implies that no cleanup is
// required.
static inline iree_hal_file_release_callback_t
iree_hal_file_release_callback_null(void) {
  iree_hal_file_release_callback_t callback = {NULL, NULL};
  return callback;
}

// Defines the type of an external file handle.
// Each type may only be usable in a subset of implementations and platforms and
// may even vary based on the runtime device properties or file instance.
//
// See the notes on each type for requirements; compatibility often requires
// the handle to check and trying to import/export is the most reliable way to
// check for support.
typedef enum iree_hal_external_file_type_e {
  IREE_HAL_EXTERNAL_FILE_TYPE_NONE = 0,

  // A fixed-size range of host memory.
  // An imported/exported file does not own a reference to the memory and the
  // caller is responsible for ensuring the memory remains live for as long as
  // the iree_hal_file_t referencing it.
  IREE_HAL_EXTERNAL_FILE_TYPE_HOST_ALLOCATION,

  // TODO(benvanik): file descriptor, FILE*, HANDLE, etc.
} iree_hal_external_file_type_t;

// Flags for controlling iree_hal_external_file_t implementation details.
enum iree_hal_external_file_flag_bits_t {
  IREE_HAL_EXTERNAL_FILE_FLAG_NONE = 0u,
};
typedef uint32_t iree_hal_external_file_flags_t;

// Handle to a typed external file.
// This is a non-owning reference and the underlying file contents must remain
// valid for as long as the handle is in use. Some file types support internal
// referencing counting but in general ownership remains with the caller.
// See the type enum for more information.
typedef struct iree_hal_external_file_t {
  // Type of the resource used to interpret the handle.
  iree_hal_external_file_type_t type;
  // Flags indicating file compatibility.
  iree_hal_external_file_flags_t flags;
  union {
    // IREE_HAL_EXTERNAL_FILE_TYPE_HOST_ALLOCATION
    iree_byte_span_t host_allocation;
  } handle;
} iree_hal_external_file_t;

//===----------------------------------------------------------------------===//
// iree_hal_file_t
//===----------------------------------------------------------------------===//

// A file handle usable with asynchronous device transfer operations.
// Files are used for bulk data upload and download and on some implementations
// may have hardware-optimized transfer paths.
//
// Implementations with support:
//  CPU: file descriptors/HANDLEs
//  CUDA: cuFile
//    https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html
//  Direct3D: IDStorageFileX
//    https://learn.microsoft.com/en-us/gaming/gdk/_content/gc/system/overviews/directstorage/directstorage-overview
//  Metal: MTLIOFileHandle
//    https://developer.apple.com/documentation/metal/resource_loading?language=objc
//
// Some implementations may allow additional non-native contents to be wrapped
// in file handles to provide implementation-controlled transfer even if not
// hardware-accelerated. See iree_hal_file_import for more information.
typedef struct iree_hal_file_t iree_hal_file_t;

// TODO(benvanik): support opening files from paths.
// IREE_API_EXPORT iree_status_t iree_hal_file_open(
//     iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
//     iree_hal_file_mode_t mode, iree_hal_memory_access_t access,
//     iree_string_view_t path, iree_hal_file_t** out_file);

// Imports an externally-owned |external_file| handle for use on |device|.
//
// Access checks will be performed against the provided |access| bits and
// callers must ensure the access is accurate (don't allow writes to read-only
// mapped memory, etc).
//
// The provided |external_file| handle is not owned and callers must either
// ensure it remains valid for the lifetime of the handle or retain it prior
// to calling and release it with the provided optional |release_callback|.
// The release callback allows the caller to listen for when the underlying
// resource is no longer in use by the HAL and can be used to perform lifetime
// management of the external file handle, file system synchronization, etc.
//
// |out_file| must be released by the caller.
// Fails with IREE_STATUS_UNAVAILABLE if the allocator cannot import the file.
// This may be due to unavailable device/platform capabilities or the properties
// of the external file handle.
IREE_API_EXPORT iree_status_t iree_hal_file_import(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access,
    iree_hal_external_file_t* IREE_RESTRICT external_file,
    iree_hal_file_release_callback_t release_callback,
    iree_hal_file_t** out_file);

// Retains the given |file| for the caller.
IREE_API_EXPORT void iree_hal_file_retain(iree_hal_file_t* file);

// Releases the given |file| from the caller.
IREE_API_EXPORT void iree_hal_file_release(iree_hal_file_t* file);

//===----------------------------------------------------------------------===//
// iree_hal_file_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_file_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_file_t* IREE_RESTRICT file);
} iree_hal_file_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_file_vtable_t);

IREE_API_EXPORT void iree_hal_file_destroy(iree_hal_file_t* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_FILE_H_
