// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Queue channel wire format.
//
// Queue frames carry frontier-ordered commands for HAL operations. The format
// extends a base frame structure with frontier metadata for causal ordering.
//
// ## Frame header layout
//
//   ┌────────────────────────────────────────────────────────────────────────┐
//   │ Bytes 0-3:   magic (0x51455249 "IREQ" - IREE Queue)                    │
//   │ Byte 4:      version (currently 1)                                     │
//   │ Byte 5:      type (see iree_net_queue_frame_type_t)                    │
//   │ Byte 6:      flags (see iree_net_queue_frame_flags_t)                  │
//   │ Byte 7:      reserved (must be 0)                                      │
//   │ Bytes 8-11:  payload_length (little-endian uint32)                     │
//   │ Bytes 12-15: stream_id (little-endian uint32, for multiplexing)        │
//   └────────────────────────────────────────────────────────────────────────┘
//
// ## Frontier encoding
//
// When frontier flags are set, frontiers are prepended to the payload:
//   [wait_frontier (if HAS_WAIT_FRONTIER)]
//   [signal_frontier (if HAS_SIGNAL_FRONTIER)]
//   [command data]
//
// Frontiers are encoded as their wire representation (see
// iree/async/frontier.h).

#ifndef IREE_NET_CHANNEL_QUEUE_FRAME_H_
#define IREE_NET_CHANNEL_QUEUE_FRAME_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Queue frame magic bytes ("IREQ" - IREE Queue, little-endian).
#define IREE_NET_QUEUE_FRAME_MAGIC 0x51455249u

// Current queue frame format version.
#define IREE_NET_QUEUE_FRAME_VERSION 1

// Size of the queue frame header in bytes.
#define IREE_NET_QUEUE_FRAME_HEADER_SIZE 16

// Default maximum frame size (header + payload) for the framing accumulator.
// Individual queue frames are limited to this size on the wire. Command
// payloads larger than this (common — typical range is 64KB-512KB) are
// fragmented into DATA/DATA_END frames by stream_id and reassembled by
// the channel before delivery.
#define IREE_NET_QUEUE_FRAME_DEFAULT_MAX_SIZE (64 * 1024)

//===----------------------------------------------------------------------===//
// Frame types
//===----------------------------------------------------------------------===//

// Queue frame type identifiers.
typedef enum iree_net_queue_frame_type_e {
  // Command frame carrying a HAL command with optional frontiers.
  IREE_NET_QUEUE_FRAME_TYPE_COMMAND = 0x01,

  // Advance frame: server→client notification that operations up to the given
  // signal frontier have completed. May carry resolution payload (e.g.,
  // provisional→resolved buffer ID mappings for alloca).
  IREE_NET_QUEUE_FRAME_TYPE_ADVANCE = 0x02,

  // Partial command data (more fragments follow).
  IREE_NET_QUEUE_FRAME_TYPE_DATA = 0x80,

  // Final command data fragment.
  IREE_NET_QUEUE_FRAME_TYPE_DATA_END = 0x81,
} iree_net_queue_frame_type_t;

//===----------------------------------------------------------------------===//
// Frame flags
//===----------------------------------------------------------------------===//

// Queue frame flag bits.
typedef enum iree_net_queue_frame_flag_bits_e {
  IREE_NET_QUEUE_FRAME_FLAG_NONE = 0u,

  // Payload includes a wait frontier (prepended to command data).
  IREE_NET_QUEUE_FRAME_FLAG_HAS_WAIT_FRONTIER = 1u << 0,

  // Payload includes a signal frontier (prepended to command data).
  IREE_NET_QUEUE_FRAME_FLAG_HAS_SIGNAL_FRONTIER = 1u << 1,

  // Payload is compressed (codec must decompress).
  IREE_NET_QUEUE_FRAME_FLAG_COMPRESSED = 1u << 2,

  // Payload is encrypted (codec must decrypt).
  IREE_NET_QUEUE_FRAME_FLAG_ENCRYPTED = 1u << 3,
} iree_net_queue_frame_flag_bits_t;
typedef uint8_t iree_net_queue_frame_flags_t;

//===----------------------------------------------------------------------===//
// Frame header
//===----------------------------------------------------------------------===//

// On-wire queue frame header. All fields are little-endian.
typedef struct iree_net_queue_frame_header_t {
  uint32_t magic;           // IREE_NET_QUEUE_FRAME_MAGIC
  uint8_t version;          // IREE_NET_QUEUE_FRAME_VERSION
  uint8_t type;             // iree_net_queue_frame_type_t
  uint8_t flags;            // iree_net_queue_frame_flags_t
  uint8_t reserved;         // Must be 0.
  uint32_t payload_length;  // Payload size in bytes.
  uint32_t stream_id;       // Stream identifier for multiplexing.
} iree_net_queue_frame_header_t;

static_assert(sizeof(iree_net_queue_frame_header_t) ==
                  IREE_NET_QUEUE_FRAME_HEADER_SIZE,
              "Queue frame header must be exactly 16 bytes");
static_assert(offsetof(iree_net_queue_frame_header_t, magic) == 0, "");
static_assert(offsetof(iree_net_queue_frame_header_t, version) == 4, "");
static_assert(offsetof(iree_net_queue_frame_header_t, type) == 5, "");
static_assert(offsetof(iree_net_queue_frame_header_t, flags) == 6, "");
static_assert(offsetof(iree_net_queue_frame_header_t, reserved) == 7, "");
static_assert(offsetof(iree_net_queue_frame_header_t, payload_length) == 8, "");
static_assert(offsetof(iree_net_queue_frame_header_t, stream_id) == 12, "");

//===----------------------------------------------------------------------===//
// Frame header initialization
//===----------------------------------------------------------------------===//

// Initializes a queue frame header with the given parameters.
static inline void iree_net_queue_frame_header_initialize(
    iree_net_queue_frame_type_t type, iree_net_queue_frame_flags_t flags,
    uint32_t payload_length, uint32_t stream_id,
    iree_net_queue_frame_header_t* out_header) {
  out_header->magic = IREE_NET_QUEUE_FRAME_MAGIC;
  out_header->version = IREE_NET_QUEUE_FRAME_VERSION;
  out_header->type = (uint8_t)type;
  out_header->flags = flags;
  out_header->reserved = 0;
  out_header->payload_length = payload_length;
  out_header->stream_id = stream_id;
}

//===----------------------------------------------------------------------===//
// Frame header accessors
//===----------------------------------------------------------------------===//

// Validates a queue frame header's magic, version, and reserved fields.
static inline iree_status_t iree_net_queue_frame_header_validate(
    iree_net_queue_frame_header_t header) {
  if (header.magic != IREE_NET_QUEUE_FRAME_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid queue frame magic: 0x%08X", header.magic);
  }
  if (header.version != IREE_NET_QUEUE_FRAME_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported queue frame version: %u",
                            header.version);
  }
  if (header.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue frame reserved field must be 0");
  }
  return iree_ok_status();
}

// Returns the frame type.
static inline iree_net_queue_frame_type_t iree_net_queue_frame_header_type(
    iree_net_queue_frame_header_t header) {
  return (iree_net_queue_frame_type_t)header.type;
}

// Returns the frame flags.
static inline iree_net_queue_frame_flags_t iree_net_queue_frame_header_flags(
    iree_net_queue_frame_header_t header) {
  return header.flags;
}

// Returns the payload length.
static inline uint32_t iree_net_queue_frame_header_payload_length(
    iree_net_queue_frame_header_t header) {
  return header.payload_length;
}

// Returns the stream ID.
static inline uint32_t iree_net_queue_frame_header_stream_id(
    iree_net_queue_frame_header_t header) {
  return header.stream_id;
}

// Returns true if the specified flag is set.
static inline bool iree_net_queue_frame_header_has_flag(
    iree_net_queue_frame_header_t header,
    iree_net_queue_frame_flag_bits_t flag) {
  return iree_any_bit_set(header.flags, flag);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_QUEUE_FRAME_H_
