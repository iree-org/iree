// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_FILE_H_
#define IREE_HAL_FILE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/queue.h"
#include "iree/hal/resource.h"
#include "iree/io/file_handle.h"

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

// Flags for controlling imported file handle implementation details.
enum iree_hal_external_file_flag_bits_t {
  IREE_HAL_EXTERNAL_FILE_FLAG_NONE = 0u,
};
typedef uint32_t iree_hal_external_file_flags_t;

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
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file);

// Retains the given |file| for the caller.
IREE_API_EXPORT void iree_hal_file_retain(iree_hal_file_t* file);

// Releases the given |file| from the caller.
IREE_API_EXPORT void iree_hal_file_release(iree_hal_file_t* file);

// Returns the memory access allowed to the file.
// This may be more strict than the original file handle backing the resource
// if for example we want to prevent particular users from mutating the file.
IREE_API_EXPORT iree_hal_memory_access_t
iree_hal_file_allowed_access(iree_hal_file_t* file);

// Returns the total accessible range of the file.
// This may be a portion of the original file backing this handle.
IREE_API_EXPORT uint64_t iree_hal_file_length(iree_hal_file_t* file);

// Returns an optional device-accessible storage buffer representing the file.
// Available if the implementation is able to perform import/address-space
// mapping/etc such that device-side transfers can directly access the resources
// as if they were a normal device buffer.
IREE_API_EXPORT iree_hal_buffer_t* iree_hal_file_storage_buffer(
    iree_hal_file_t* file);

// TODO(benvanik): truncate/extend? (both can be tricky with async)

// Returns true if the iree_hal_file_read and iree_hal_file_write APIs are
// available for use on the file. Not all implementations support synchronous
// I/O.
IREE_API_EXPORT bool iree_hal_file_supports_synchronous_io(
    iree_hal_file_t* file);

// Synchronously reads a segment of |file| into |buffer|.
// Blocks the caller until completed. Buffers are always host mappable.
// Only available if iree_hal_file_supports_synchronous_io is true.
IREE_API_EXPORT iree_status_t iree_hal_file_read(
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length);

// Synchronously writes a segment of |buffer| into |file|.
// Blocks the caller until completed. Buffers are always host mappable.
// Only available if iree_hal_file_supports_synchronous_io is true.
IREE_API_EXPORT iree_status_t iree_hal_file_write(
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length);

//===----------------------------------------------------------------------===//
// iree_hal_file_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_file_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_file_t* IREE_RESTRICT file);

  iree_hal_memory_access_t(IREE_API_PTR* allowed_access)(iree_hal_file_t* file);

  uint64_t(IREE_API_PTR* length)(iree_hal_file_t* file);

  iree_hal_buffer_t*(IREE_API_PTR* storage_buffer)(iree_hal_file_t* file);

  bool(IREE_API_PTR* supports_synchronous_io)(iree_hal_file_t* file);
  iree_status_t(IREE_API_PTR* read)(iree_hal_file_t* file, uint64_t file_offset,
                                    iree_hal_buffer_t* buffer,
                                    iree_device_size_t buffer_offset,
                                    iree_device_size_t length);
  iree_status_t(IREE_API_PTR* write)(iree_hal_file_t* file,
                                     uint64_t file_offset,
                                     iree_hal_buffer_t* buffer,
                                     iree_device_size_t buffer_offset,
                                     iree_device_size_t length);
} iree_hal_file_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_file_vtable_t);

IREE_API_EXPORT void iree_hal_file_destroy(iree_hal_file_t* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_FILE_H_
