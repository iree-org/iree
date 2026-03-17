// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bulk channel wire format.
//
// Bulk frames carry large data transfers with chunking and progress tracking.
// The format is optimized for transfers exceeding 4GB with minimal overhead.
//
// ## Frame header layout
//
//   ┌────────────────────────────────────────────────────────────────────────┐
//   │ Bytes 0-3:   magic (0x42455249 "IREB" - IREE Bulk)                     │
//   │ Byte 4:      version (currently 1)                                     │
//   │ Byte 5:      type (see iree_net_bulk_frame_type_t)                     │
//   │ Byte 6:      flags                                                     │
//   │ Byte 7:      reserved (must be 0)                                      │
//   │ Bytes 8-15:  transfer_id (little-endian uint64)                        │
//   │ Bytes 16-23: total_size (little-endian uint64, in START frame)         │
//   │ Bytes 24-31: chunk_offset (little-endian uint64)                       │
//   │ Bytes 32-35: chunk_length (little-endian uint32)                       │
//   │ Bytes 36-39: sequence (little-endian uint32, for datagram ordering)    │
//   └────────────────────────────────────────────────────────────────────────┘
//
// [FUTURE - not yet implemented]

#ifndef IREE_NET_CHANNEL_BULK_FRAME_H_
#define IREE_NET_CHANNEL_BULK_FRAME_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Bulk frame magic bytes ("IREB" - IREE Bulk, little-endian).
#define IREE_NET_BULK_FRAME_MAGIC 0x42455249u

// Current bulk frame format version.
#define IREE_NET_BULK_FRAME_VERSION 1

// Size of the bulk frame header in bytes.
#define IREE_NET_BULK_FRAME_HEADER_SIZE 40

//===----------------------------------------------------------------------===//
// Frame types
//===----------------------------------------------------------------------===//

// Bulk frame type identifiers.
typedef enum iree_net_bulk_frame_type_e {
  // Transfer start: announces a new transfer with total_size.
  IREE_NET_BULK_FRAME_TYPE_START = 0x01,

  // Data chunk: carries chunk_length bytes at chunk_offset.
  IREE_NET_BULK_FRAME_TYPE_DATA = 0x02,

  // Transfer complete: signals successful completion.
  IREE_NET_BULK_FRAME_TYPE_COMPLETE = 0x03,

  // Transfer abort: signals transfer cancellation.
  IREE_NET_BULK_FRAME_TYPE_ABORT = 0x04,
} iree_net_bulk_frame_type_t;

//===----------------------------------------------------------------------===//
// Frame flags
//===----------------------------------------------------------------------===//

// Bulk frame flag bits.
typedef enum iree_net_bulk_frame_flag_bits_e {
  IREE_NET_BULK_FRAME_FLAG_NONE = 0u,

  // Payload is compressed.
  IREE_NET_BULK_FRAME_FLAG_COMPRESSED = 1u << 0,

  // Payload is encrypted.
  IREE_NET_BULK_FRAME_FLAG_ENCRYPTED = 1u << 1,

  // This is the final chunk (can deliver before all chunks received).
  IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK = 1u << 2,
} iree_net_bulk_frame_flag_bits_t;
typedef uint8_t iree_net_bulk_frame_flags_t;

//===----------------------------------------------------------------------===//
// Frame header
//===----------------------------------------------------------------------===//

// On-wire bulk frame header. All fields are little-endian.
typedef struct iree_net_bulk_frame_header_t {
  uint32_t magic;         // IREE_NET_BULK_FRAME_MAGIC
  uint8_t version;        // IREE_NET_BULK_FRAME_VERSION
  uint8_t type;           // iree_net_bulk_frame_type_t
  uint8_t flags;          // iree_net_bulk_frame_flags_t
  uint8_t reserved;       // Must be 0.
  uint64_t transfer_id;   // Unique transfer identifier.
  uint64_t total_size;    // Total transfer size (in START frame).
  uint64_t chunk_offset;  // Offset of this chunk within transfer.
  uint32_t chunk_length;  // Length of this chunk's payload.
  uint32_t sequence;      // Sequence number for datagram ordering.
} iree_net_bulk_frame_header_t;

static_assert(sizeof(iree_net_bulk_frame_header_t) ==
                  IREE_NET_BULK_FRAME_HEADER_SIZE,
              "Bulk frame header must be exactly 40 bytes");

// TODO(benvanik): header initialization and accessor functions.

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_BULK_FRAME_H_
