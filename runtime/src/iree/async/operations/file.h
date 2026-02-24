// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// File operations for async file I/O.
//
// All operations in this file work with proactor-managed file handles obtained
// via iree_async_file_open_operation_t or iree_async_file_import().
//
// Availability (all operations):
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Performance characteristics:
//   io_uring: Native async I/O via IORING_OP_READ/WRITE.
//   IOCP: Native async I/O via ReadFile/WriteFile overlapped.
//   kqueue: Emulated via thread pool (kqueue doesn't support file I/O).
//   generic: Emulated via thread pool.

#ifndef IREE_ASYNC_OPERATIONS_FILE_H_
#define IREE_ASYNC_OPERATIONS_FILE_H_

#include "iree/async/file.h"
#include "iree/async/operation.h"
#include "iree/async/span.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Open
//===----------------------------------------------------------------------===//

// Open flags for file operations.
//
// Flag support by backend:
//   Flag      generic | io_uring | IOCP  | kqueue
//   ────────────────────────────────────────────────
//   READ      yes     | yes      | yes   | yes
//   WRITE     yes     | yes      | yes   | yes
//   CREATE    yes     | yes      | yes   | yes
//   TRUNCATE  yes     | yes      | yes   | yes
//   APPEND    yes     | yes      | yes   | yes
//   DIRECT    emul*   | yes      | yes   | yes
//
//   * generic: O_DIRECT may be unavailable on some filesystems; falls back
//     to buffered I/O with a warning.
//
enum iree_async_file_open_flag_bits_e {
  IREE_ASYNC_FILE_OPEN_FLAG_NONE = 0u,

  // Open for reading. The file must exist.
  IREE_ASYNC_FILE_OPEN_FLAG_READ = 1u << 0,

  // Open for writing. Combined with CREATE to create if absent.
  IREE_ASYNC_FILE_OPEN_FLAG_WRITE = 1u << 1,

  // Create the file if it does not exist. Requires WRITE.
  IREE_ASYNC_FILE_OPEN_FLAG_CREATE = 1u << 2,

  // Truncate the file to zero length on open. Requires WRITE.
  IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE = 1u << 3,

  // Writes append to the end of the file (offset is ignored for writes).
  IREE_ASYNC_FILE_OPEN_FLAG_APPEND = 1u << 4,

  // Bypass the kernel page cache (O_DIRECT on POSIX, FILE_FLAG_NO_BUFFERING
  // on Windows). Buffers must be aligned to the filesystem block size
  // (typically 512 or 4096 bytes). Enables predictable I/O latency at the
  // cost of requiring the caller to manage buffering.
  //
  // Use this for:
  //   - Database files where you manage your own page cache.
  //   - Large sequential reads/writes where page cache adds overhead.
  //   - Latency-sensitive workloads where page cache eviction is disruptive.
  IREE_ASYNC_FILE_OPEN_FLAG_DIRECT = 1u << 5,
};
typedef uint32_t iree_async_file_open_flags_t;

// Opens a file asynchronously.
//
// On success, |opened_file| is set to the new file handle (caller must
// release it or submit a close operation).
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | 5.6+     | yes  | emul
//
// Path lifetime:
//   The |path| string must remain valid until the callback fires. The
//   proactor does not copy the path—it reads it at submission time (or
//   during the async operation for backends that defer the open).
//
// Flag validation:
//   Invalid flag combinations (e.g., TRUNCATE without WRITE) return
//   IREE_STATUS_INVALID_ARGUMENT synchronously at submit time.
//
// Returns (via callback status):
//   IREE_STATUS_OK: File opened successfully.
//   IREE_STATUS_NOT_FOUND: File does not exist (and CREATE not specified).
//   IREE_STATUS_PERMISSION_DENIED: Insufficient permissions.
//   IREE_STATUS_ALREADY_EXISTS: File exists and exclusive create requested.
typedef struct iree_async_file_open_operation_t {
  iree_async_operation_t base;

  // Path to open. Must be null-terminated and remain valid until the
  // callback fires.
  const char* path;

  // Open mode flags.
  iree_async_file_open_flags_t open_flags;

  // Result: the opened file (new reference, caller must release).
  iree_async_file_t* opened_file;
} iree_async_file_open_operation_t;

//===----------------------------------------------------------------------===//
// Read
//===----------------------------------------------------------------------===//

// Reads data from a file at a specified offset (pread semantics).
//
// On success, |bytes_read| indicates how many bytes were written to the buffer.
// A short read (bytes_read < buffer.data_length) indicates EOF or partial I/O.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | emul
//
// Positional semantics (pread):
//   The read occurs at the specified |offset| independent of any shared file
//   position. Multiple concurrent reads to the same file at different offsets
//   are safe. The file position is not modified.
//
// Short reads:
//   A successful completion with bytes_read < requested indicates:
//     - EOF reached (normal for end of file).
//     - Signal interruption (rare, retryable).
//   The caller should check bytes_read and issue another read if needed.
//
// Buffer alignment (with DIRECT flag):
//   When the file was opened with IREE_ASYNC_FILE_OPEN_FLAG_DIRECT, the
//   buffer address and size must be aligned to the filesystem block size.
//
// Threading model:
//   Callback fires on the poll thread. Buffer contents are valid after
//   the callback—no need to copy during the callback.
typedef struct iree_async_file_read_operation_t {
  iree_async_operation_t base;

  // The file to read from.
  iree_async_file_t* file;

  // File offset to read from (independent of any shared file position).
  uint64_t offset;

  // Buffer to read into.
  iree_async_span_t buffer;

  // Result: number of bytes read.
  iree_host_size_t bytes_read;
} iree_async_file_read_operation_t;

//===----------------------------------------------------------------------===//
// Write
//===----------------------------------------------------------------------===//

// Writes data to a file at a specified offset (pwrite semantics).
//
// On success, |bytes_written| indicates how many bytes were consumed from
// the buffer. Partial writes are possible under disk pressure.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | emul
//
// Positional semantics (pwrite):
//   The write occurs at the specified |offset| independent of any shared
//   file position. Multiple concurrent writes to the same file at different
//   offsets are safe (though overlapping writes have undefined ordering).
//   The file position is not modified.
//
// Append mode:
//   If the file was opened with IREE_ASYNC_FILE_OPEN_FLAG_APPEND, the
//   |offset| field is ignored and writes always append to the end of the
//   file atomically.
//
// Buffer alignment (with DIRECT flag):
//   When the file was opened with IREE_ASYNC_FILE_OPEN_FLAG_DIRECT, the
//   buffer address and size must be aligned to the filesystem block size.
//
// Durability:
//   Write completion means data has been accepted by the kernel, not
//   necessarily persisted to disk. Use fsync/fdatasync (via separate
//   operation or post-write) for durability guarantees.
typedef struct iree_async_file_write_operation_t {
  iree_async_operation_t base;

  // The file to write to.
  iree_async_file_t* file;

  // File offset to write at (independent of any shared file position).
  // Ignored if file was opened with APPEND flag.
  uint64_t offset;

  // Buffer to write from.
  iree_async_span_t buffer;

  // Result: number of bytes written.
  iree_host_size_t bytes_written;
} iree_async_file_write_operation_t;

//===----------------------------------------------------------------------===//
// File close
//===----------------------------------------------------------------------===//

// Closes a file asynchronously.
//
// Consumes the caller's reference: the file must not be accessed after
// submit. On completion, the file is destroyed and the callback reports
// whether the close succeeded.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | 5.6+     | yes  | yes
//
// Error reporting:
//   Close can fail for network filesystems (NFS, CIFS) where the server
//   may report deferred write errors at close time. Local filesystems
//   typically don't fail close, but the status should still be checked.
//
//   Common errors:
//     IREE_STATUS_IO_ERROR: NFS write-back failure or disk error.
//     IREE_STATUS_RESOURCE_EXHAUSTED: Disk full (deferred from write).
//
// Threading model:
//   Callback fires on the poll thread after the file is closed.
//   The file handle is invalid before the callback fires—do not use it.
typedef struct iree_async_file_close_operation_t {
  iree_async_operation_t base;

  // The file to close. Consumed on submit.
  iree_async_file_t* file;
} iree_async_file_close_operation_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_OPERATIONS_FILE_H_
