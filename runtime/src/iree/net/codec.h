// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree_net_codec_t for documentation.

#ifndef IREE_NET_CODEC_H_
#define IREE_NET_CODEC_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_net_codec_flags_t
//===----------------------------------------------------------------------===//

// Flags describing codec behavior and requirements.
// Set by codec implementations at creation time.
typedef enum iree_net_codec_flag_bits_e {
  IREE_NET_CODEC_FLAG_NONE = 0u,

  // Codec requires CPU-accessible memory for encode/decode.
  // Most codecs set this because they need to read/write payload bytes.
  //
  // When NOT set, the codec doesn't inspect data and can be used with
  // device-only memory paths (devmem TCP, RDMA direct).
  //
  // Channels check this at creation time: if codec requires CPU access but
  // the carrier only provides device memory, creation fails.
  IREE_NET_CODEC_FLAG_REQUIRES_CPU_ACCESS = 1u << 0,
} iree_net_codec_flag_bits_t;
typedef uint32_t iree_net_codec_flags_t;

// Fixed overhead added by a codec around the payload.
// Total frame size = prefix + payload_length + suffix.
typedef struct iree_net_codec_overhead_t {
  iree_host_size_t prefix;  // Bytes before payload.
  iree_host_size_t suffix;  // Bytes after payload.
} iree_net_codec_overhead_t;

// Returns overhead with both prefix and suffix set to zero.
static inline iree_net_codec_overhead_t iree_net_codec_overhead_zero(void) {
  iree_net_codec_overhead_t overhead = {0, 0};
  return overhead;
}

// Returns total overhead (prefix + suffix).
static inline iree_host_size_t iree_net_codec_overhead_total(
    iree_net_codec_overhead_t overhead) {
  return overhead.prefix + overhead.suffix;
}

// Computes frame size for a given payload length.
static inline iree_host_size_t iree_net_codec_frame_size(
    iree_net_codec_overhead_t overhead, iree_host_size_t payload_length) {
  return overhead.prefix + payload_length + overhead.suffix;
}

//===----------------------------------------------------------------------===//
// iree_net_codec_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_codec_t iree_net_codec_t;
typedef struct iree_net_codec_vtable_t iree_net_codec_vtable_t;

struct iree_net_codec_vtable_t {
  void (*destroy)(iree_net_codec_t* codec);
  iree_net_codec_overhead_t (*query_overhead)(iree_net_codec_t* codec);
  iree_status_t (*encode)(iree_net_codec_t* codec, uint8_t* payload_ptr,
                          iree_host_size_t payload_length);
  iree_status_t (*decode)(iree_net_codec_t* codec, uint8_t* frame_ptr,
                          iree_host_size_t frame_length,
                          iree_byte_span_t* out_payload);
};

// Codec for in-place frame transforms.
//
// Codecs transform frame payloads with fixed overhead, enabling zero-copy
// operation. Pipes query overhead once at setup and pre-allocate frames with
// space for prefix and suffix:
//
//   Frame layout: [prefix][payload][suffix]
//
// Codecs do not allocate buffers. The caller provides the frame buffer and
// the codec operates in-place. For encode, the caller writes payload data
// starting at buffer + prefix. For decode, the codec returns a span pointing
// into the same buffer (past the prefix).
//
// A NULL codec pointer means no codec. The inline API functions short-circuit
// without vtable calls: query_overhead returns {0,0}, encode is a no-op,
// decode returns the entire frame as payload.
//
// Concrete implementations embed this structure at offset 0.
struct iree_net_codec_t {
  iree_atomic_ref_count_t ref_count;
  const iree_net_codec_vtable_t* vtable;
  iree_net_codec_flags_t flags;
  iree_allocator_t host_allocator;
};

// Initializes base codec fields. Called by codec implementations.
static inline void iree_net_codec_initialize(
    const iree_net_codec_vtable_t* vtable, iree_net_codec_flags_t flags,
    iree_allocator_t host_allocator, iree_net_codec_t* out_codec) {
  iree_atomic_ref_count_init(&out_codec->ref_count);
  out_codec->vtable = vtable;
  out_codec->flags = flags;
  out_codec->host_allocator = host_allocator;
}

// Returns the codec's behavior flags.
// Returns NONE for NULL codec.
static inline iree_net_codec_flags_t iree_net_codec_flags(
    iree_net_codec_t* codec) {
  return codec ? codec->flags : IREE_NET_CODEC_FLAG_NONE;
}

// Returns true if the codec requires CPU-accessible memory.
// Returns false for NULL codec (no codec = no requirements).
static inline bool iree_net_codec_requires_cpu_access(iree_net_codec_t* codec) {
  return codec && iree_any_bit_set(codec->flags,
                                   IREE_NET_CODEC_FLAG_REQUIRES_CPU_ACCESS);
}

static inline void iree_net_codec_retain(iree_net_codec_t* codec) {
  if (codec) {
    iree_atomic_ref_count_inc(&codec->ref_count);
  }
}

static inline void iree_net_codec_release(iree_net_codec_t* codec) {
  if (codec && iree_atomic_ref_count_dec(&codec->ref_count) == 1) {
    codec->vtable->destroy(codec);
  }
}

// Returns the codec's fixed overhead. Immutable after codec creation.
// Returns {0, 0} for NULL codec.
static inline iree_net_codec_overhead_t iree_net_codec_query_overhead(
    iree_net_codec_t* codec) {
  if (!codec) return iree_net_codec_overhead_zero();
  return codec->vtable->query_overhead(codec);
}

// Encodes payload in-place.
//
// Buffer layout: [prefix][payload][suffix]
//                        ^^^^^^^^
//                        |payload_ptr| points here
//
// The codec writes to the prefix and suffix regions and transforms the
// payload in-place. The payload length does not change.
//
// |payload_ptr| points to the start of the payload within the buffer.
// |payload_length| is the payload size (unchanged after encoding).
//
// For NULL codec, returns success immediately (no-op).
static inline iree_status_t iree_net_codec_encode(
    iree_net_codec_t* codec, uint8_t* payload_ptr,
    iree_host_size_t payload_length) {
  if (!codec) return iree_ok_status();
  return codec->vtable->encode(codec, payload_ptr, payload_length);
}

// Decodes frame in-place.
//
// Buffer layout: [prefix][encoded payload][suffix]
//                ^^^^^^^^
//                |frame_ptr| points here
//
// The codec reads from the prefix and suffix regions and transforms the
// payload in-place. Returns a span pointing to the decoded payload.
//
// |frame_ptr| points to the start of the entire frame (including prefix).
// |frame_length| is the total frame size (prefix + payload + suffix).
// |out_payload| receives a span pointing to the decoded payload within
// the buffer (at frame_ptr + prefix_overhead).
//
// For NULL codec, returns the entire frame as the payload.
//
// On error, the contents of the frame buffer are undefined. Callers must not
// read from the buffer after decode fails.
//
// Returns IREE_STATUS_DATA_LOSS if validation fails.
// Returns IREE_STATUS_INVALID_ARGUMENT if frame_length < total overhead.
static inline iree_status_t iree_net_codec_decode(
    iree_net_codec_t* codec, uint8_t* frame_ptr, iree_host_size_t frame_length,
    iree_byte_span_t* out_payload) {
  if (!codec) {
    *out_payload = iree_make_byte_span(frame_ptr, frame_length);
    return iree_ok_status();
  }
  return codec->vtable->decode(codec, frame_ptr, frame_length, out_payload);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CODEC_H_
