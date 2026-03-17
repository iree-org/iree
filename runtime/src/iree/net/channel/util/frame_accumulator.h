// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stream reassembly utility for TCP/QUIC receive paths.
//
// The frame accumulator handles the common pattern of reassembling
// length-prefixed frames from a byte stream, optimizing for the zero-copy
// case where complete frames arrive in a single buffer.
//
// ## Zero-copy optimization
//
// When a complete frame fits in a single received buffer, the frame callback
// receives the lease directly so it can retain ownership if needed (e.g., for
// deferred processing). This avoids copying data into the internal buffer.
//
// When frames span multiple buffers, data is copied into the internal buffer
// and the callback receives a NULL lease (data must be consumed synchronously).
//
// ## Scope
//
// This utility is for CPU-accessible streams only (control channel, queue
// channel). Bulk channels with devmem TCP use device memory payloads that
// cannot be read by the CPU and require a different mechanism.
//
// ## Protocol requirements
//
// The frame_length_fn callback MUST be able to determine the frame size from
// a compact header that fits within max_frame_size bytes. Protocols with
// arbitrarily-large headers before the length is known (e.g., HTTP/1.1 with
// unbounded headers) are not suitable for this accumulator.
//
// Suitable protocols include:
//   - Fixed-size length prefix (e.g., 4-byte little-endian length)
//   - Variable-length integer prefix (e.g., protobuf varint, QUIC varint)
//   - Fixed-size binary frame headers (e.g., HTTP/2's 9-byte header)
//
// If frame_length_fn cannot determine the frame size after the internal buffer
// fills, push_lease returns IREE_STATUS_RESOURCE_EXHAUSTED. This prevents
// denial-of-service attacks where malicious input stalls the receive pump.
//
// ## Usage
//
//   // Allocate storage for accumulator + internal buffer.
//   iree_host_size_t storage_size =
//       iree_net_frame_accumulator_storage_size(max_frame_size);
//   void* storage = iree_allocator_malloc(allocator, storage_size, &storage);
//   iree_net_frame_accumulator_t* accumulator =
//       (iree_net_frame_accumulator_t*)storage;
//
//   // Initialize.
//   IREE_RETURN_IF_ERROR(iree_net_frame_accumulator_initialize(
//       accumulator, max_frame_size, frame_length, on_frame_complete));
//
//   // Process received data (from io_uring completion, etc).
//   IREE_RETURN_IF_ERROR(iree_net_frame_accumulator_push_lease(
//       accumulator, &lease, bytes_received));
//
//   // Cleanup.
//   iree_net_frame_accumulator_deinitialize(accumulator);
//   iree_allocator_free(allocator, storage);

#ifndef IREE_NET_CHANNEL_UTIL_FRAME_ACCUMULATOR_H_
#define IREE_NET_CHANNEL_UTIL_FRAME_ACCUMULATOR_H_

#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Callback types
//===----------------------------------------------------------------------===//

// Returns the total frame size (header + payload) if determinable from the
// available data, or 0 if more bytes are needed to determine the frame size.
//
// The callback examines the available bytes (which may be partial) and returns:
//   - The complete frame size if the header is fully available.
//   - 0 if more bytes are needed to read the length field.
//
// The returned frame size must include any header bytes that encode the length.
// For example, a 4-byte length-prefixed protocol returning a frame with 100
// bytes of payload should return 104 (4 header + 100 payload).
//
// IMPORTANT: This callback must be able to determine the frame size from a
// compact header. If it continues returning 0 after the accumulator's internal
// buffer fills (max_frame_size bytes), push_lease will fail with
// IREE_STATUS_RESOURCE_EXHAUSTED rather than loop indefinitely.
typedef iree_host_size_t (*iree_net_frame_length_fn_t)(
    void* user_data, iree_const_byte_span_t available);

typedef struct iree_net_frame_length_callback_t {
  iree_net_frame_length_fn_t fn;
  void* user_data;
} iree_net_frame_length_callback_t;

// Called when a complete frame is available for processing.
//
// |frame| contains the complete frame data (including any header bytes).
//
// |lease| is non-NULL if the frame data comes directly from a received buffer
// (zero-copy path). If |lease| is NULL, the frame data is from the internal
// buffer and must be consumed synchronously before returning.
//
// ## Lease ownership
//
// The accumulator retains ownership of the lease and will release it after
// push_lease returns (regardless of callback success or failure). If the
// callback needs to defer processing of a zero-copy frame, it MUST call
// iree_async_buffer_lease_retain() to acquire its own reference before
// returning. The retained reference must later be released when processing
// completes.
//
// Note: A single lease may span multiple frames. Retaining the lease keeps
// the entire buffer alive, not just the current frame's bytes.
//
// ## Error handling
//
// If the callback returns an error, frame processing stops and the error
// propagates to the push_lease caller. The accumulator state remains valid
// and processing can continue after the error is handled.
typedef iree_status_t (*iree_net_frame_complete_fn_t)(
    void* user_data, iree_const_byte_span_t frame,
    iree_async_buffer_lease_t* lease);

typedef struct iree_net_frame_complete_callback_t {
  iree_net_frame_complete_fn_t fn;
  void* user_data;
} iree_net_frame_complete_callback_t;

//===----------------------------------------------------------------------===//
// Frame accumulator
//===----------------------------------------------------------------------===//

// Stream reassembly buffer with flexible array member for internal storage.
// Use iree_net_frame_accumulator_storage_size() to determine allocation size.
typedef struct iree_net_frame_accumulator_t {
  // Callback to determine frame length from partial data.
  iree_net_frame_length_callback_t frame_length;
  // Callback invoked when a complete frame is available.
  iree_net_frame_complete_callback_t on_frame_complete;
  // Capacity of the internal buffer (set from max_frame_size at init).
  iree_host_size_t buffer_capacity;
  // Bytes currently buffered (partial frame in progress).
  iree_host_size_t buffer_used;
  // Internal buffer for reassembling fragmented frames.
  // Sized by caller via storage_size().
  uint8_t buffer[];
} iree_net_frame_accumulator_t;

// Returns the total bytes needed for an accumulator with the given max frame
// size. Caller allocates this much memory, then calls initialize.
static inline iree_host_size_t iree_net_frame_accumulator_storage_size(
    iree_host_size_t max_frame_size) {
  return sizeof(iree_net_frame_accumulator_t) + max_frame_size;
}

// Initializes an accumulator in pre-allocated memory.
//
// |accumulator| must point to at least storage_size(max_frame_size) bytes.
// |max_frame_size| is the maximum frame size this accumulator can handle;
//   frames larger than this will return IREE_STATUS_RESOURCE_EXHAUSTED.
// |frame_length| determines frame boundaries from partial data.
// |on_frame_complete| is called for each complete frame.
iree_status_t iree_net_frame_accumulator_initialize(
    iree_net_frame_accumulator_t* accumulator, iree_host_size_t max_frame_size,
    iree_net_frame_length_callback_t frame_length,
    iree_net_frame_complete_callback_t on_frame_complete);

// Deinitializes an accumulator. No deallocation occurs (caller owns memory).
void iree_net_frame_accumulator_deinitialize(
    iree_net_frame_accumulator_t* accumulator);

// Discards any partial frame in progress and resets to initial state.
void iree_net_frame_accumulator_reset(
    iree_net_frame_accumulator_t* accumulator);

// Processes received data from a buffer lease.
//
// Parses frames from the buffer and invokes on_frame_complete for each
// complete frame. When possible, passes the lease directly to the callback
// for zero-copy handling. Otherwise, copies data to the internal buffer.
//
// |lease| is the buffer lease from the receive operation. The accumulator
//   takes ownership and will release it (idempotent) after processing.
// |valid_bytes| is the number of valid bytes in the lease (may be less than
//   the lease's span length).
//
// Returns OK if all frames were processed successfully.
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if a frame exceeds max_frame_size.
// Returns any error returned by on_frame_complete (processing stops on error).
iree_status_t iree_net_frame_accumulator_push_lease(
    iree_net_frame_accumulator_t* accumulator, iree_async_buffer_lease_t* lease,
    iree_host_size_t valid_bytes);

// Returns the number of bytes currently buffered (partial frame in progress).
static inline iree_host_size_t iree_net_frame_accumulator_buffered_bytes(
    const iree_net_frame_accumulator_t* accumulator) {
  return accumulator->buffer_used;
}

// Returns true if a partial frame is currently being accumulated.
static inline bool iree_net_frame_accumulator_has_partial_frame(
    const iree_net_frame_accumulator_t* accumulator) {
  return accumulator->buffer_used > 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_UTIL_FRAME_ACCUMULATOR_H_
