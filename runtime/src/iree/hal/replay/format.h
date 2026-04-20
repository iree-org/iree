// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_FORMAT_H_
#define IREE_HAL_REPLAY_FORMAT_H_

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Magic header bytes for IREE HAL replay files: "IRRP".
#define IREE_HAL_REPLAY_FILE_MAGIC 0x50525249u

// Major version of the IREE HAL replay file format.
#define IREE_HAL_REPLAY_FILE_VERSION_MAJOR 1u

// Minor version of the IREE HAL replay file format.
#define IREE_HAL_REPLAY_FILE_VERSION_MINOR 0u

// Session-local object identifier used by replay records.
typedef uint64_t iree_hal_replay_object_id_t;

// Sentinel object identifier for omitted or not-yet-created objects.
#define IREE_HAL_REPLAY_OBJECT_ID_NONE 0ull

// Type of one top-level record in an IREE HAL replay file.
typedef uint16_t iree_hal_replay_file_record_type_t;
enum iree_hal_replay_file_record_type_e {
  IREE_HAL_REPLAY_FILE_RECORD_TYPE_NONE = 0u,
  // Capture session metadata.
  IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION = 1u,
  // HAL object creation, wrapping, aliasing, or destruction metadata.
  IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT = 2u,
  // HAL API operation in capture sequence order.
  IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION = 3u,
  // Opaque byte payload stored in the replay file blob store.
  IREE_HAL_REPLAY_FILE_RECORD_TYPE_BLOB = 4u,
  // Diagnostic record for unsupported or lossy capture behavior.
  IREE_HAL_REPLAY_FILE_RECORD_TYPE_UNSUPPORTED = 5u,
};

// Bitfield specifying properties of one replay file record.
typedef uint16_t iree_hal_replay_file_record_flags_t;
enum iree_hal_replay_file_record_flag_bits_t {
  IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE = 0u,

  // Record may be skipped by readers that do not understand its type.
  // Unknown records without this flag are required for faithful replay and must
  // fail strict readers.
  IREE_HAL_REPLAY_FILE_RECORD_FLAG_OPTIONAL = 1u << 0,
};

// Session-local HAL object kind.
typedef uint32_t iree_hal_replay_object_type_t;
enum iree_hal_replay_object_type_e {
  IREE_HAL_REPLAY_OBJECT_TYPE_NONE = 0u,
  IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE = 1u,
  IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR = 2u,
  IREE_HAL_REPLAY_OBJECT_TYPE_POOL = 3u,
  IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER = 4u,
  IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER = 5u,
  IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE_CACHE = 6u,
  IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE = 7u,
  IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE = 8u,
  IREE_HAL_REPLAY_OBJECT_TYPE_FILE = 9u,
  IREE_HAL_REPLAY_OBJECT_TYPE_EVENT = 10u,
  IREE_HAL_REPLAY_OBJECT_TYPE_CHANNEL = 11u,
  IREE_HAL_REPLAY_OBJECT_TYPE_HOST_CALL = 12u,
};

// High-level HAL API operation kind.
typedef uint32_t iree_hal_replay_operation_code_t;
enum iree_hal_replay_operation_code_e {
  IREE_HAL_REPLAY_OPERATION_CODE_NONE = 0u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_TRIM = 1u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_I64 = 2u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_CAPABILITIES = 3u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_REFINE_TOPOLOGY_EDGE = 4u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO = 5u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_CHANNEL = 6u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER = 7u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EVENT = 8u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EXECUTABLE_CACHE = 9u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_IMPORT_FILE = 10u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE = 11u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_QUEUE_POOL_BACKEND = 12u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA = 13u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA = 14u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FILL = 15u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_UPDATE = 16u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_COPY = 17u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_READ = 18u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_WRITE = 19u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_HOST_CALL = 20u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DISPATCH = 21u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE = 22u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FLUSH = 23u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_BEGIN = 24u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_FLUSH = 25u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_END = 26u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_EXTERNAL_CAPTURE_BEGIN = 27u,
  IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_EXTERNAL_CAPTURE_END = 28u,

  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_TRIM = 100u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_QUERY_MEMORY_HEAPS = 101u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER = 102u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER = 103u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_EXPORT_BUFFER = 104u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_QUERY_GRANULARITY =
      105u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE = 106u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE = 107u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE = 108u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE = 109u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP = 110u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP = 111u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT = 112u,
  IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE = 113u,

  IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE = 200u,
  IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE = 201u,
  IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_INVALIDATE_RANGE = 202u,
  IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE = 203u,
};

// Producer-defined payload schema stored in record headers.
typedef uint32_t iree_hal_replay_payload_type_t;
enum iree_hal_replay_payload_type_e {
  IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE = 0u,
  IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT = 1u,
  IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_ALLOCATE_BUFFER = 2u,
  IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE = 3u,
};

// Payload describing a captured buffer object.
typedef struct iree_hal_replay_buffer_object_payload_t {
  // Total allocation size, in bytes, reported by the source buffer.
  uint64_t allocation_size;
  // Byte offset of this buffer view within the underlying allocation.
  uint64_t byte_offset;
  // Byte length of this buffer view.
  uint64_t byte_length;
  // Queue affinity from the buffer placement metadata.
  uint64_t queue_affinity;
  // Buffer placement flags from the buffer placement metadata.
  uint32_t placement_flags;
  // Memory type bits assigned to the buffer.
  uint32_t memory_type;
  // Buffer usage bits allowed by the buffer.
  uint32_t allowed_usage;
  // Memory access bits allowed by the buffer.
  uint16_t allowed_access;
  // Reserved for future buffer object metadata; must be zero.
  uint16_t reserved0;
  // Reserved for future buffer object metadata; must be zero.
  uint32_t reserved1;
} iree_hal_replay_buffer_object_payload_t;

// Payload describing an allocator buffer allocation request.
typedef struct iree_hal_replay_allocator_allocate_buffer_payload_t {
  // Requested allocation size, in bytes.
  uint64_t allocation_size;
  // Requested queue affinity from the canonicalized buffer params.
  uint64_t queue_affinity;
  // Requested minimum alignment, in bytes.
  uint64_t min_alignment;
  // Requested buffer usage bits.
  uint32_t usage;
  // Requested memory type bits.
  uint32_t type;
  // Requested memory access bits.
  uint16_t access;
  // Reserved for future allocation request metadata; must be zero.
  uint16_t reserved0;
  // Reserved for future allocation request metadata; must be zero.
  uint32_t reserved1;
} iree_hal_replay_allocator_allocate_buffer_payload_t;

// Payload describing a buffer byte range operation.
typedef struct iree_hal_replay_buffer_range_payload_t {
  // Byte offset within the underlying allocation passed to the driver vtable.
  uint64_t byte_offset;
  // Byte length of the range passed to the driver vtable.
  uint64_t byte_length;
  // Mapping mode bits for map operations, or zero otherwise.
  uint32_t mapping_mode;
  // Memory access bits for map operations, or zero otherwise.
  uint16_t memory_access;
  // Reserved for future buffer range metadata; must be zero.
  uint16_t reserved0;
  // Reserved for future buffer range metadata; must be zero.
  uint32_t reserved1;
} iree_hal_replay_buffer_range_payload_t;

// Compression algorithm used for one replay file byte range.
typedef uint16_t iree_hal_replay_compression_type_t;
enum iree_hal_replay_compression_type_e {
  // Range bytes are stored directly in the replay file.
  IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE = 0u,
};

// Digest algorithm used for one replay file byte range.
typedef uint16_t iree_hal_replay_digest_type_t;
enum iree_hal_replay_digest_type_e {
  // Digest bytes are not populated.
  IREE_HAL_REPLAY_DIGEST_TYPE_NONE = 0u,
  // FNV-1a 64-bit integrity checksum stored in the first 8 digest bytes.
  //
  // This is a cheap corruption check for local replay plumbing, not a security
  // boundary. Stronger content hashes can be added without changing the range
  // record shape.
  IREE_HAL_REPLAY_DIGEST_TYPE_FNV1A_64 = 1u,
};

// Bitfield specifying properties of one replay file byte range.
typedef uint32_t iree_hal_replay_file_range_flags_t;
enum iree_hal_replay_file_range_flag_bits_t {
  IREE_HAL_REPLAY_FILE_RANGE_FLAG_NONE = 0u,
};

// Reference to byte storage inside a replay file.
typedef struct iree_hal_replay_file_range_t {
  // Byte offset in the replay file where the stored range begins.
  uint64_t offset;
  // Stored byte length in the replay file.
  uint64_t length;
  // Uncompressed byte length observed by replay consumers.
  uint64_t uncompressed_length;
  // Flags describing optional range behavior.
  iree_hal_replay_file_range_flags_t flags;
  // Compression algorithm used for the stored bytes.
  iree_hal_replay_compression_type_t compression_type;
  // Digest algorithm used for integrity validation.
  iree_hal_replay_digest_type_t digest_type;
  // Reserved for future range metadata; must be zero.
  uint32_t reserved0;
  // Digest bytes for the uncompressed range contents.
  uint8_t digest[32];
} iree_hal_replay_file_range_t;

// Returns an empty replay file range.
static inline iree_hal_replay_file_range_t iree_hal_replay_file_range_empty(
    void) {
  iree_hal_replay_file_range_t range;
  memset(&range, 0, sizeof(range));
  return range;
}

// File header stored at byte 0 of every IREE HAL replay file.
typedef struct iree_hal_replay_file_header_t {
  // Magic bytes equal to IREE_HAL_REPLAY_FILE_MAGIC.
  uint32_t magic;
  // Major version of the file format.
  uint16_t version_major;
  // Minor version of the file format.
  uint16_t version_minor;
  // Size of this header in bytes.
  uint32_t header_length;
  // Reserved for future file-level flags; must be zero.
  uint32_t flags;
  // Final byte length of the valid replay file contents, or zero while open.
  uint64_t file_length;
} iree_hal_replay_file_header_t;

// Fixed record header followed by |payload_length| bytes.
//
// All multi-byte fields are serialized in little-endian order. The current
// writer only supports little-endian hosts; a future big-endian implementation
// must byte-swap instead of changing the file format.
typedef struct iree_hal_replay_file_record_header_t {
  // Total byte length of this record including header and payload.
  uint64_t record_length;
  // Total payload byte length following this header.
  uint64_t payload_length;
  // Capture-order sequence ordinal assigned by the shared recorder.
  uint64_t sequence_ordinal;
  // Capture-time host thread identifier, or zero when unknown.
  uint64_t thread_id;
  // Session-local device object id associated with this record.
  iree_hal_replay_object_id_t device_id;
  // Primary session-local object id associated with this record.
  iree_hal_replay_object_id_t object_id;
  // Secondary session-local object id associated with this record.
  iree_hal_replay_object_id_t related_object_id;
  // Size of this record header in bytes.
  uint32_t header_length;
  // Type of this replay file record.
  iree_hal_replay_file_record_type_t record_type;
  // Flags specifying optional record behavior.
  iree_hal_replay_file_record_flags_t record_flags;
  // Producer-defined payload schema identifier.
  uint32_t payload_type;
  // HAL object kind for object records, or zero otherwise.
  iree_hal_replay_object_type_t object_type;
  // HAL operation kind for operation records, or zero otherwise.
  iree_hal_replay_operation_code_t operation_code;
  // IREE status code returned by the captured operation, or zero for OK/none.
  uint32_t status_code;
  // Reserved for future record fields; must be zero.
  uint32_t reserved0;
  // Reserved for future record fields; must be zero.
  uint64_t reserved1;
} iree_hal_replay_file_record_header_t;

// Returns true if |record_type| is known by this version of the format.
static inline bool iree_hal_replay_file_record_type_is_known(
    iree_hal_replay_file_record_type_t record_type) {
  switch (record_type) {
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION:
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT:
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION:
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_BLOB:
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_UNSUPPORTED:
      return true;
    default:
      return false;
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_FORMAT_H_
