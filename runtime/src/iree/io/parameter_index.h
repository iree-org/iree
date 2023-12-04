// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_PARAMETER_INDEX_H_
#define IREE_IO_PARAMETER_INDEX_H_

#include "iree/base/api.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Dictates the interpretation of a parameter index entry.
typedef enum iree_io_parameter_index_entry_storage_type_e {
  // Parameter is a read-only splatted value (repeated sequence of bytes to a
  // certain length). No backing storage is required and requests to read the
  // parameter will receive the provided pattern.
  IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT = 0u,
  // Parameter is backed by a range of bytes within a file. Access rights are
  // inherited from the file handle.
  IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE,
} iree_io_parameter_index_entry_storage_type_t;

// Power of two; enough bytes to fit complex128 (complex<f64>).
// Prefer 1, 2, and 4 byte patterns as they can often hit hardware accelerated
// fast paths while 8 and 16 may require emulation.
#define IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH 16

// An entry in an in-memory file index.
typedef struct iree_io_parameter_index_entry_t {
  // Key used to reference this file.
  iree_string_view_t key;
  // Optional metadata.
  iree_const_byte_span_t metadata;
  // Length of the entry in bytes.
  uint64_t length;
  // Type of the parameter dictating how storage is handled.
  iree_io_parameter_index_entry_storage_type_t type;
  // Defines the backing storage of a parameter based on its type.
  union {
    // Describes a synthetic parameter comprised of some repeated sequence of
    // bytes. Sized to allow for up to complex128 (complex<f64>) values.
    // Valid when type is IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT.
    struct {
      // Length of the pattern in bytes defining how many bytes of the pattern
      // field are valid from index 0.
      uint8_t pattern_length;
      // Little-endian pattern used to fill the range in memory.
      // Examples:
      //         0xAA: pattern_length=1 pattern=[0xAA ...]
      //       0xBBAA: pattern_length=2 pattern=[0xAA 0xBB ...]
      //   0xDDCCBBAA: pattern_length=4 pattern=[0xAA 0xBB 0xCC 0xDD ...]
      uint8_t pattern[IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH];
    } splat;
    // Describes a file-backed parameter.
    // Valid when type is IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE.
    struct {
      // File handle backing this entry, retained.
      iree_io_file_handle_t* handle;
      // Offset of the entry in bytes relative to the base file offset.
      uint64_t offset;
    } file;
  } storage;
} iree_io_parameter_index_entry_t;

// An in-memory file index mapping keys to byte ranges in referenced files.
// A single index may contain entries from multiple files. Each parameter is
// backed by a contiguous range in a single file.
//
// Thread-safe due to insert-only behavior. If we ever wanted to allow removal
// from the index we would need to change callers to hold a mutex or design
// a callback-based API to ensure that entries were live for as long as the
// callers were using them.
typedef struct iree_io_parameter_index_t iree_io_parameter_index_t;

// Creates an empty file index.
IREE_API_EXPORT iree_status_t iree_io_parameter_index_create(
    iree_allocator_t host_allocator, iree_io_parameter_index_t** out_index);

// Retains the given |index| for the caller.
IREE_API_EXPORT void iree_io_parameter_index_retain(
    iree_io_parameter_index_t* index);

// Releases the given |index| from the caller.
IREE_API_EXPORT void iree_io_parameter_index_release(
    iree_io_parameter_index_t* index);

// Returns the number of entries in the index at the time the method is called.
// New entries may be added by other threads between when the value is queried
// and when the caller enumerates entries. Use this only for debugging.
IREE_API_EXPORT iree_host_size_t
iree_io_parameter_index_count(iree_io_parameter_index_t* index);

// Reserves storage for at least |new_capacity| entries in the index.
// Ignored if storage capacity is already sufficient.
IREE_API_EXPORT iree_status_t iree_io_parameter_index_reserve(
    iree_io_parameter_index_t* index, iree_host_size_t new_capacity);

// Adds a new entry to the file index.
// The string key and optional metadata will be copied into the index and
// need not remain valid after the call returns. Referenced file handles will
// be retained for the lifetime of the index.
IREE_API_EXPORT iree_status_t
iree_io_parameter_index_add(iree_io_parameter_index_t* index,
                            const iree_io_parameter_index_entry_t* entry);

// Returns the entry at index |i| in [0, iree_io_parameter_index_count).
// The returned |out_entry| is valid for the lifetime of the index.
IREE_API_EXPORT iree_status_t iree_io_parameter_index_get(
    iree_io_parameter_index_t* index, iree_host_size_t i,
    const iree_io_parameter_index_entry_t** out_entry);

// Performs a file entry lookup of |key| in the index and returns it.
// The returned |out_entry| is valid for the lifetime of the index.
IREE_API_EXPORT iree_status_t iree_io_parameter_index_lookup(
    iree_io_parameter_index_t* index, iree_string_view_t key,
    const iree_io_parameter_index_entry_t** out_entry);

// Formats a textual dump of the parameter |index| to |builder|.
// An optional |scope| name can be provided to include in the dump.
IREE_API_EXPORT iree_status_t iree_io_parameter_index_dump(
    iree_string_view_t scope, iree_io_parameter_index_t* index,
    iree_string_builder_t* builder);

// Writes a textual dump of the parameter |index| to |file|.
// An optional |scope| name can be provided to include in the dump.
IREE_API_EXPORT iree_status_t iree_io_parameter_index_fprint(
    FILE* file, iree_string_view_t scope, iree_io_parameter_index_t* index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_PARAMETER_INDEX_H_
