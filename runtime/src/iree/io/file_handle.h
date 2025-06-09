// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_FILE_HANDLE_H_
#define IREE_IO_FILE_HANDLE_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and enums
//===----------------------------------------------------------------------===//

// Bits defining which operations are allowed on a file.
typedef uint32_t iree_io_file_access_t;
enum iree_io_file_access_bits_t {
  // Allows operations that read from the file.
  IREE_IO_FILE_ACCESS_READ = 1u << 0,
  // Allows operations that write to the file.
  IREE_IO_FILE_ACCESS_WRITE = 1u << 1,
};

//===----------------------------------------------------------------------===//
// iree_io_file_handle_primitive_t
//===----------------------------------------------------------------------===//

// Defines the type of a platform file handle primitive.
// Each type may only be usable in a subset of implementations and platforms and
// may even vary based on the runtime device properties or file instance.
//
// See the notes on each type for requirements; compatibility often requires
// the handle to check and trying to import/export is the most reliable way to
// check for support.
typedef enum iree_io_file_handle_type_e {
  // A fixed-size range of host memory.
  // The handle creator is responsible for ensuring the memory remains live for
  // as long as the file handle referencing it.
  IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION = 0u,

  // Platform file descriptor (fd).
  IREE_IO_FILE_HANDLE_TYPE_FD,

  // TODO(benvanik): FILE*, HANDLE, etc.
} iree_io_file_handle_type_t;

// A platform handle to a file primitive.
// Unowned/unreferenced on its own but may be embedded in instances that manage
// lifetime such as iree_io_file_handle_t.
typedef union iree_io_file_handle_primitive_value_t {
  // IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION
  iree_byte_span_t host_allocation;
  // IREE_IO_FILE_HANDLE_TYPE_FD
  int fd;
} iree_io_file_handle_primitive_value_t;

// A (type, value) pair describing a system file primitive handle.
typedef struct iree_io_file_handle_primitive_t {
  iree_io_file_handle_type_t type;
  iree_io_file_handle_primitive_value_t value;
} iree_io_file_handle_primitive_t;

//===----------------------------------------------------------------------===//
// iree_io_file_handle_t
//===----------------------------------------------------------------------===//

typedef void(IREE_API_PTR* iree_io_file_handle_release_fn_t)(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive);

// A callback issued when a file is released.
typedef struct {
  // Callback function pointer.
  iree_io_file_handle_release_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_io_file_handle_release_callback_t;

// Returns a no-op file release callback that implies that no cleanup is
// required.
static inline iree_io_file_handle_release_callback_t
iree_io_file_handle_release_callback_null(void) {
  iree_io_file_handle_release_callback_t callback = {NULL, NULL};
  return callback;
}

// A reference-counted file handle wrapping a platform primitive.
// Handles are stateless (unlike file descriptors) and only used to reference
// files.
typedef struct iree_io_file_handle_t iree_io_file_handle_t;

// Wraps a platform file primitive |handle_primitive| in a reference-counted
// file handle. |allowed_access| declares which operations are allowed on the
// handle and may be more restrictive than the underlying platform primitive.
// The optional provided |release_callback| will be issued when the last
// reference to the handle is released.
IREE_API_EXPORT iree_status_t iree_io_file_handle_wrap(
    iree_io_file_access_t allowed_access,
    iree_io_file_handle_primitive_t handle_primitive,
    iree_io_file_handle_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle);

// Wraps a |host_allocation| in a reference-counted file handle.
// |allowed_access| declares which operations are allowed on the handle and may
// be more restrictive than the underlying memory protection.
// The optional provided |release_callback| will be issued when the last
// reference to the handle is released.
IREE_API_EXPORT iree_status_t iree_io_file_handle_wrap_host_allocation(
    iree_io_file_access_t allowed_access, iree_byte_span_t host_allocation,
    iree_io_file_handle_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle);

// Retains the file |handle| for the caller.
IREE_API_EXPORT void iree_io_file_handle_retain(iree_io_file_handle_t* handle);

// Releases the file |handle| and releases the underlying platform primitive if
// the caller held the last owning reference.
IREE_API_EXPORT void iree_io_file_handle_release(iree_io_file_handle_t* handle);

// Returns the allowed access operations on the file |handle|.
// As handles can specify more restrictive access than the underlying platform
// primitive users of the handles must always verify accesses using this value.
IREE_API_EXPORT iree_io_file_access_t
iree_io_file_handle_access(const iree_io_file_handle_t* handle);

// Returns the underlying platform file |handle| primitive value.
// Only valid for as long as the file handle is retained. Closing or otherwise
// invalidating the platform primitive while the handle still has uses will
// result in undefined behavior.
IREE_API_EXPORT iree_io_file_handle_primitive_t
iree_io_file_handle_primitive(const iree_io_file_handle_t* handle);

// Returns the type of the file |handle|.
static inline iree_io_file_handle_type_t iree_io_file_handle_type(
    const iree_io_file_handle_t* handle) {
  return iree_io_file_handle_primitive(handle).type;
}

// Returns the underlying platform file |handle| primitive value.
// Only valid for as long as the file handle is retained. Closing or otherwise
// invalidating the platform primitive while the handle still has uses will
// result in undefined behavior.
static inline iree_io_file_handle_primitive_value_t iree_io_file_handle_value(
    const iree_io_file_handle_t* handle) {
  return iree_io_file_handle_primitive(handle).value;
}

// Flushes pending writes of |handle| to its backing storage.
IREE_API_EXPORT iree_status_t
iree_io_file_handle_flush(iree_io_file_handle_t* handle);

//===----------------------------------------------------------------------===//
// iree_io_file_handle_t platform files
//===----------------------------------------------------------------------===//

// Bits indicating how a file is opened.
typedef uint64_t iree_io_file_mode_t;
enum iree_io_file_mode_bits_t {
  // Allow reads of both existing and new content.
  IREE_IO_FILE_MODE_READ = 1ull << 0,
  // Allow writes.
  IREE_IO_FILE_MODE_WRITE = 1ull << 1,
  // Hints that the file will be accessed at random (more-so than not).
  // Mutually exclusive with IREE_IO_FILE_MODE_SEQUENTIAL_SCAN. If no access
  // hint is specified the platform will use its default behavior.
  IREE_IO_FILE_MODE_RANDOM_ACCESS = 1ull << 2,
  // Hints that the file will be accessed sequentially (contiguous reads/writes
  // or small skips forward only).
  // Mutually exclusive with IREE_IO_FILE_MODE_RANDOM_ACCESS. If no access
  // hint is specified the platform will use its default behavior.
  IREE_IO_FILE_MODE_SEQUENTIAL_SCAN = 1ull << 3,
  // Hints that the library and system caching are not required. May hurt
  // performance more than it helps unless the file is very large and
  // exclusively accessed as part of bulk transfer operations that are
  // page-aligned.
  IREE_IO_FILE_MODE_DIRECT = 1ull << 4,
  // Ensures the file is deleted when it is closed. Platforms may use this as a
  // hint to avoid writing the file contents when cache is available.
  IREE_IO_FILE_MODE_TEMPORARY = 1ull << 5,
  // Allows subsequent operations to open the file for read access while the
  // file is open by the creator.
  IREE_IO_FILE_MODE_SHARE_READ = 1ull << 6,
  // Allows subsequent operations to open the file for write access while the
  // file is open by the creator.
  IREE_IO_FILE_MODE_SHARE_WRITE = 1ull << 7,
};

// Creates a new platform file at |path| for usage as defined by |mode|.
// The file will be extended to |initial_size| upon creation.
// Returns IREE_STATUS_ALREADY_EXISTS if the file already exists.
// Returns IREE_STATUS_PERMISSION_DENIED if the file cannot be created.
IREE_API_EXPORT iree_status_t iree_io_file_handle_create(
    iree_io_file_mode_t mode, iree_string_view_t path, uint64_t initial_size,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle);

// Opens an existing platform file at |path| for usage as defined by |mode|.
// Returns IREE_STATUS_NOT_FOUND if the file does not exist.
// Returns IREE_STATUS_PERMISSION_DENIED if the specified |mode| is disallowed.
IREE_API_EXPORT iree_status_t iree_io_file_handle_open(
    iree_io_file_mode_t mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle);

// Duplicates an existing platform fd that was already opened as |mode|
// Returns IREE_STATUS_INVALID_ARGUMENT if the fd was not valid.
IREE_API_EXPORT iree_status_t iree_io_file_handle_open_fd(
    iree_io_file_mode_t mode, int fd, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_handle);

//===----------------------------------------------------------------------===//
// iree_io_file_mapping_t
//===----------------------------------------------------------------------===//
// EXPERIMENTAL: this API may change once proper memory objects and views are
// added to iree/base/. This may just end up as a thin wrapper around that
// lower-level API with more fancy features (address placement, commit/decommit,
// etc) left to the lower-level API. We may add new APIs here for flush/sync
// as required.

// Flags used to control the behavior of mapped file views.
typedef uint64_t iree_io_file_mapping_flags_t;
enum iree_io_file_mapping_flag_bits_t {
  IREE_IO_FILE_MAPPING_FLAG_NONE = 0u,

  // Indicates that the memory access pattern of the view is mostly sequential.
  // Hints to the system that an LRU page cache and sequential prefetching are
  // likely to be worth it.
  //
  // Implemented by MADV_SEQUENTIAL.
  IREE_IO_FILE_MAPPING_FLAG_SEQUENTIAL_ACCESS = 1ull << 0,

  // Enables large page support for the given view, if available.
  // Certain mapping modes such as mapping of existing files or opening
  // mappings from another process where the allocation was not made with large
  // pages may not support large pages and the flag will be silently ignored.
  // In either case the memory view will be padded to the
  // iree_memory_info_t::large_page_size regardless of whether the pages are
  // actually large to the system.
  //
  // Use large pages to reduce the overhead involved in accessing
  // hot-but-non-localized memory views that may otherwise spend a significant
  // amount of time/capacity maintaining the TLB. As the platform and
  // machine-dependent large page size is often several orders of magnitude
  // larger than the normal page size (MB vs. KB) care should be used to only
  // apply this to large allocations.
  //
  // Implemented by FILE_MAP_LARGE_PAGES/MAP_HUGETLB, where available.
  IREE_IO_FILE_MAPPING_FLAG_LARGE_PAGES = 1ull << 1,

  // Excludes the view memory from minidumps/coredumps.
  // This is a hint that the memory in the ranges are not useful to include in
  // dumps, such as large chunks of read-only file data (model weights, images,
  // etc).
  //
  // Implemented by WerRegisterExcludedMemoryBlock/MADV_DONTDUMP, where
  // available.
  IREE_IO_FILE_MAPPING_FLAG_EXCLUDE_FROM_DUMPS = 1ull << 2,

  // Privately map the memory into the calling process.
  // Other processes that may hold a reference to the file will not see changes.
  // This is not a guarantee but an optimization to possibly avoid non-trivial
  // kernel overheads.
  //
  // Implemented by MAP_PRIVATE, where available.
  IREE_IO_FILE_MAPPING_FLAG_PRIVATE = 1ull << 3,
};

// A mapped file view into host memory.
//
// Thread-safe; the mapping is immutable and may be accessed from any thread.
// The **contents** of the mapping in the file should be coherent across threads
// within the same process but may not be across threads in different processes.
typedef struct iree_io_file_mapping_t iree_io_file_mapping_t;

// Maps a view of a file into host-accessible memory.
// The provided file |handle| is retained for the lifetime of the view.
// To map the entire file specify a range of [0, IREE_HOST_SIZE_MAX].
//
// If the provided file |handle| is already available for use as a host pointer
// it is returned directly.
IREE_API_EXPORT iree_status_t iree_io_file_map_view(
    iree_io_file_handle_t* handle, iree_io_file_access_t access,
    uint64_t offset, iree_host_size_t length,
    iree_io_file_mapping_flags_t flags, iree_allocator_t host_allocator,
    iree_io_file_mapping_t** out_mapping);

// Retains the file |mapping| for the caller. The backing file handle will be
// retained as well.
IREE_API_EXPORT void iree_io_file_mapping_retain(
    iree_io_file_mapping_t* mapping);

// Releases the file |mapping| and its reference to the backing file handle.
// If the mapping was the last remaining retainer of the handle it will be
// closed.
IREE_API_EXPORT void iree_io_file_mapping_release(
    iree_io_file_mapping_t* mapping);

// Returns the length of the mapped view in bytes.
IREE_API_EXPORT iree_host_size_t
iree_io_file_mapping_length(const iree_io_file_mapping_t* mapping);

// Returns a host-accessible read-only pointer to the file mapping memory.
// Returns iree_const_byte_span_empty if the mapping is not readable.
IREE_API_EXPORT iree_const_byte_span_t
iree_io_file_mapping_contents_ro(const iree_io_file_mapping_t* mapping);

// Returns a host-accessible read-write pointer to the file mapping memory.
// Returns iree_byte_span_empty if the mapping is not writable.
IREE_API_EXPORT iree_byte_span_t
iree_io_file_mapping_contents_rw(iree_io_file_mapping_t* mapping);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_FILE_HANDLE_H_
