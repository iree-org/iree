// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_PROFILE_SINK_H_
#define IREE_HAL_PROFILE_SINK_H_

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
  // a complete representation of the selected range.
  IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED = 1ull << 0,
};

// Content type for a profiling session boundary metadata chunk.
#define IREE_HAL_PROFILE_CONTENT_TYPE_SESSION \
  IREE_SV("application/vnd.iree.hal.profile.session")

// Content type for an array of iree_hal_profile_device_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES \
  IREE_SV("application/vnd.iree.hal.profile.devices")

// Content type for an array of iree_hal_profile_queue_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES \
  IREE_SV("application/vnd.iree.hal.profile.queues")

// Content type for an array of iree_hal_profile_dispatch_event_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS \
  IREE_SV("application/vnd.iree.hal.profile.dispatch-events")

// Bitfield specifying which optional device record fields are populated.
typedef uint32_t iree_hal_profile_device_flags_t;
enum iree_hal_profile_device_flag_bits_t {
  IREE_HAL_PROFILE_DEVICE_FLAG_NONE = 0u,

  // |physical_device_uuid| contains a stable physical device identifier.
  IREE_HAL_PROFILE_DEVICE_FLAG_PHYSICAL_DEVICE_UUID = 1u << 0,
};

// Session-level physical device description.
//
// Producers should emit device records before event records that reference the
// corresponding |physical_device_ordinal|. The ordinal is a compact
// session-local key; consumers that need hermetic identity should use
// |physical_device_uuid| when present.
typedef struct iree_hal_profile_device_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags specifying which optional fields are populated.
  iree_hal_profile_device_flags_t flags;
  // Session-local physical device ordinal used by compact event records.
  uint32_t physical_device_ordinal;
  // Number of queues described for this physical device.
  uint32_t queue_count;
  // Stable physical device UUID when
  // IREE_HAL_PROFILE_DEVICE_FLAG_PHYSICAL_DEVICE_UUID is set.
  uint8_t physical_device_uuid[16];
} iree_hal_profile_device_record_t;

// Returns a default physical device record.
static inline iree_hal_profile_device_record_t
iree_hal_profile_device_record_default(void) {
  iree_hal_profile_device_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  return record;
}

// Session-level queue description.
//
// Producers should emit queue records before event records that reference the
// corresponding |stream_id| or |queue_ordinal|. Queue ordinals are only scoped
// to their physical device within a profiling session.
typedef struct iree_hal_profile_queue_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Session-local physical device ordinal owning this queue.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal within |physical_device_ordinal|.
  uint32_t queue_ordinal;
  // Reserved for future queue record flags; must be zero.
  uint32_t reserved0;
  // Producer-defined stream identifier used in chunk/event metadata.
  uint64_t stream_id;
} iree_hal_profile_queue_record_t;

// Returns a default queue record.
static inline iree_hal_profile_queue_record_t
iree_hal_profile_queue_record_default(void) {
  iree_hal_profile_queue_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  record.queue_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying properties of one dispatch event record.
typedef uint32_t iree_hal_profile_dispatch_event_flags_t;
enum iree_hal_profile_dispatch_event_flag_bits_t {
  IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_NONE = 0u,

  // Dispatch was enqueued through a reusable command buffer.
  IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER = 1u << 0,

  // Workgroup counts were loaded from device memory before dispatch.
  IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS = 1u << 1,
};

// Device-timestamped dispatch execution event.
//
// Producers emit dispatch events after the device and queue metadata chunks
// that define the chunk's physical_device_ordinal, queue_ordinal, and
// stream_id. Times are raw device ticks in the producer's clock domain.
typedef struct iree_hal_profile_dispatch_event_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags describing how the dispatch was produced.
  iree_hal_profile_dispatch_event_flags_t flags;
  // Producer-defined event identifier unique within the chunk stream.
  uint64_t event_id;
  // Queue submission epoch containing this dispatch.
  uint64_t submission_id;
  // Process-local command-buffer identifier, or 0 for direct queue dispatch.
  uint64_t command_buffer_id;
  // Process-local executable identifier, or 0 when unavailable.
  uint64_t executable_id;
  // Command ordinal within a command buffer, or UINT32_MAX for direct dispatch.
  uint32_t command_index;
  // Executable export ordinal dispatched.
  uint32_t export_ordinal;
  // Workgroup counts submitted for each dimension.
  uint32_t workgroup_count[3];
  // Workgroup sizes submitted for each dimension.
  uint32_t workgroup_size[3];
  // Device timestamp captured when dispatch execution started.
  uint64_t start_tick;
  // Device timestamp captured when dispatch execution completed.
  uint64_t end_tick;
} iree_hal_profile_dispatch_event_t;

// Returns a default dispatch event record.
static inline iree_hal_profile_dispatch_event_t
iree_hal_profile_dispatch_event_default(void) {
  iree_hal_profile_dispatch_event_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.command_index = UINT32_MAX;
  record.export_ordinal = UINT32_MAX;
  return record;
}

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
  // Process-local executable identifier associated with this chunk, or 0.
  uint64_t executable_id;
  // Process-local command-buffer identifier associated with this chunk, or 0.
  uint64_t command_buffer_id;
  // Physical device ordinal associated with this chunk, or UINT32_MAX.
  uint32_t physical_device_ordinal;
  // Queue ordinal associated with this chunk, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Flags describing chunk completeness and producer behavior.
  iree_hal_profile_chunk_flags_t flags;
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
// returns.
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
