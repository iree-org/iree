// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_PROFILE_SINK_H_
#define IREE_HAL_PROFILE_SINK_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_profile_sink_t
//===----------------------------------------------------------------------===//

// Retained interface receiving profiling chunks from HAL implementations.
//
// Profile sinks are the programmatic transport for HAL-native profiling data.
// They may write chunks to files, forward them to live tools, copy them into
// application-owned telemetry buffers, or discard selected streams. Drivers
// should never call sinks from queue submission hot paths or while holding
// queue locks; sinks are allowed to block and allocate unless a specific sink
// implementation documents a stricter contract.
typedef struct iree_hal_profile_sink_t iree_hal_profile_sink_t;

// Bitfield specifying properties of a profiling chunk.
typedef uint64_t iree_hal_profile_chunk_flags_t;
enum iree_hal_profile_chunk_flag_bits_t {
  IREE_HAL_PROFILE_CHUNK_FLAG_NONE = 0u,

  // Chunk contents are partial because the producer dropped or truncated data.
  // The chunk is still structurally valid, but consumers must not treat it as
  // a complete representation of the selected range. Producers that know the
  // number of omitted typed records should report it in
  // iree_hal_profile_chunk_metadata_t::dropped_record_count.
  //
  // Producers define pressure behavior by data family. Aggregate observability
  // streams such as queue and memory events may use this flag to keep hot paths
  // bounded. Precise execution timelines such as dispatch and device queue
  // events should fail the profiled operation/session instead of silently
  // omitting records needed for timing attribution.
  IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED = 1ull << 0,
};

// Metadata describing one profiling chunk.
typedef struct iree_hal_profile_chunk_metadata_t {
  // MIME-like content type of the chunk payload.
  iree_string_view_t content_type;
  // Human-readable stream or artifact name.
  iree_string_view_t name;
  // Process-local profiling session identifier.
  uint64_t session_id;
  // Producer-defined stream identifier within |session_id|.
  uint64_t stream_id;
  // Producer-defined event identifier associated with this chunk, or 0.
  uint64_t event_id;
  // Session-local executable identifier associated with this chunk, or 0.
  uint64_t executable_id;
  // Session-local command-buffer identifier associated with this chunk, or 0.
  uint64_t command_buffer_id;
  // Physical device ordinal associated with this chunk, or UINT32_MAX.
  uint32_t physical_device_ordinal;
  // Queue ordinal associated with this chunk, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Flags describing chunk completeness and producer behavior.
  iree_hal_profile_chunk_flags_t flags;
  // Number of typed records omitted from this chunk stream, or 0 when unknown
  // or not truncated. Producers should set
  // IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED when this is nonzero. A truncated
  // chunk may have zero payload records; in that case the metadata still
  // reports that records were lost from the producer-side stream.
  uint64_t dropped_record_count;
} iree_hal_profile_chunk_metadata_t;

// Returns default profiling chunk metadata.
static inline iree_hal_profile_chunk_metadata_t
iree_hal_profile_chunk_metadata_default(void) {
  iree_hal_profile_chunk_metadata_t metadata;
  memset(&metadata, 0, sizeof(metadata));
  metadata.physical_device_ordinal = UINT32_MAX;
  metadata.queue_ordinal = UINT32_MAX;
  return metadata;
}

// Retains the given |sink| for the caller.
IREE_API_EXPORT void iree_hal_profile_sink_retain(
    iree_hal_profile_sink_t* sink);

// Releases the given |sink| from the caller.
IREE_API_EXPORT void iree_hal_profile_sink_release(
    iree_hal_profile_sink_t* sink);

// Begins a profiling session on |sink|.
//
// The |metadata| describes the session-level stream. Implementations may use
// this to allocate per-session state, write headers, or validate that the sink
// can accept the requested content type. All strings and spans passed to this
// function are borrowed and need only remain valid for the duration of the
// call.
IREE_API_EXPORT iree_status_t iree_hal_profile_sink_begin_session(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata);

// Writes one profiling chunk to |sink|.
//
// The sink must consume or copy the provided iovecs before returning. The
// producer may reuse or release the storage immediately after this call
// returns. |iovec_count| may be zero for metadata-only chunks, including
// TRUNCATED chunks that report producer-side dropped records even when no typed
// records survived.
IREE_API_EXPORT iree_status_t iree_hal_profile_sink_write(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs);

// Ends a profiling session on |sink|.
//
// |session_status_code| describes the producer-side session outcome without
// transferring ownership of an iree_status_t. Sinks that need richer error
// details should receive them as a chunk before end_session.
IREE_API_EXPORT iree_status_t iree_hal_profile_sink_end_session(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code);

//===----------------------------------------------------------------------===//
// iree_hal_profile_sink_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_profile_sink_vtable_t {
  // Destroys the sink when its final reference is released.
  void(IREE_API_PTR* destroy)(iree_hal_profile_sink_t* sink);

  // Notifies the sink that a new profiling session is beginning.
  iree_status_t(IREE_API_PTR* begin_session)(
      iree_hal_profile_sink_t* sink,
      const iree_hal_profile_chunk_metadata_t* metadata);

  // Writes one complete profiling chunk to the sink.
  iree_status_t(IREE_API_PTR* write)(
      iree_hal_profile_sink_t* sink,
      const iree_hal_profile_chunk_metadata_t* metadata,
      iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs);

  // Notifies the sink that the profiling session has ended.
  iree_status_t(IREE_API_PTR* end_session)(
      iree_hal_profile_sink_t* sink,
      const iree_hal_profile_chunk_metadata_t* metadata,
      iree_status_code_t session_status_code);
} iree_hal_profile_sink_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_profile_sink_vtable_t);

IREE_API_EXPORT void iree_hal_profile_sink_destroy(
    iree_hal_profile_sink_t* sink);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_PROFILE_SINK_H_
