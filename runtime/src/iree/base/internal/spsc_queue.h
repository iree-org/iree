// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lock-free single-producer single-consumer (SPSC) queue for variable-length
// messages operating on caller-provided memory.
//
// Designed for cross-process shared memory: the producer and consumer may be
// in different processes mapping the same shared memory region. The queue uses
// monotonically increasing 64-bit positions with acquire-release ordering and
// cache-line isolation between producer and consumer fields.
//
// Entry format: each message is framed as a 4-byte length prefix followed by
// the payload, padded to entry_alignment:
//
//   ┌──────────┬───────────────────────────────────┬─────────┐
//   │ uint32_t │ payload (entry_size bytes)        │ padding │
//   │entry_size│                                   │ to align│
//   └──────────┴───────────────────────────────────┴─────────┘
//              ╰─ entry_size bytes ─╯
//   ╰────────── padded to entry_alignment ─────────────────╯
//
// A skip marker (entry_size = UINT32_MAX) at the end of the data region
// signals the consumer to wrap to offset 0.
//
// ABI versioning: the header contains a magic number and version. The opener
// validates both and fails loud on mismatch — no forward compatibility.
//
// Usage:
//   Producer (creator):
//     void* memory = <shared memory base>;
//     iree_spsc_queue_t queue;
//     iree_spsc_queue_initialize(memory, memory_size, capacity, &queue);
//     iree_spsc_queue_write(&queue, data, length);
//
//   Consumer (opener):
//     iree_spsc_queue_t queue;
//     iree_spsc_queue_open(memory, memory_size, &queue);
//     iree_host_size_t length;
//     uint8_t buffer[4096];
//     while (iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length)) {
//       // process buffer[0..length)
//     }
//
// Thread safety: exactly one producer and one consumer. The producer API is
// not thread-safe with other producers; the consumer API is not thread-safe
// with other consumers. Producer and consumer may run concurrently.

#ifndef IREE_BASE_INTERNAL_SPSC_QUEUE_H_
#define IREE_BASE_INTERNAL_SPSC_QUEUE_H_

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Magic number identifying an SPSC queue header ("IRPQ" in little-endian).
#define IREE_SPSC_QUEUE_MAGIC ((uint32_t)0x51525049)

// Current ABI version. Exact match required on open — no forward compatibility.
#define IREE_SPSC_QUEUE_VERSION ((uint32_t)1)

// Total bytes consumed by the header region (4 cache lines).
#define IREE_SPSC_QUEUE_HEADER_SIZE ((iree_host_size_t)256)

// Minimum data capacity in bytes. Must be a power of two.
#define IREE_SPSC_QUEUE_MIN_CAPACITY ((uint32_t)64)

// Default byte alignment for entries (length prefix + payload + padding).
#define IREE_SPSC_QUEUE_DEFAULT_ENTRY_ALIGNMENT ((uint32_t)8)

// Sentinel value in the entry length field indicating a skip-to-wrap marker.
// The consumer silently advances past these.
#define IREE_SPSC_QUEUE_SKIP_MARKER UINT32_MAX

//===----------------------------------------------------------------------===//
// Memory layout structures
//===----------------------------------------------------------------------===//

// Immutable header at offset 0 of the queue memory region.
// Occupies one cache line. Written once during initialize, read-only after.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_spsc_queue_header_t {
  uint32_t magic;
  uint32_t version;
  // Data region capacity in bytes. Must be a power of two.
  uint32_t capacity;
  // Byte alignment for entries. Must be a power of two, >= 4.
  uint32_t entry_alignment;
  // Reserved for future use. Must be zero.
  uint8_t reserved[48];
} iree_spsc_queue_header_t;

// Producer write position at offset 64 (cache line 1).
// Written by the producer, read by the consumer.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_spsc_queue_write_position_t {
  iree_atomic_int64_t value;
} iree_spsc_queue_write_position_t;

// Consumer read position at offset 128 (cache line 2).
// Written by the consumer, read by the producer.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_spsc_queue_read_position_t {
  iree_atomic_int64_t value;
} iree_spsc_queue_read_position_t;

// Producer-private cached read position at offset 192 (cache line 3).
// Only the producer reads and writes this — no atomic needed.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_spsc_queue_cached_position_t {
  int64_t value;
} iree_spsc_queue_cached_position_t;

//===----------------------------------------------------------------------===//
// Queue handle
//===----------------------------------------------------------------------===//

// View into an initialized SPSC queue.
//
// Holds derived pointers into caller-provided memory — does not own the memory.
// Initialize with iree_spsc_queue_initialize() (creator) or
// iree_spsc_queue_open() (opener).
typedef struct iree_spsc_queue_t {
  iree_spsc_queue_header_t* header;
  iree_spsc_queue_write_position_t* write_position;
  iree_spsc_queue_read_position_t* read_position;
  iree_spsc_queue_cached_position_t* cached_read_position;
  uint8_t* data;
  // Cached from header for fast access.
  uint32_t capacity;
  // capacity - 1: for position-to-offset masking.
  uint32_t mask;
  uint32_t entry_alignment;
  // Producer-private pending write state (between begin_write and
  // commit_write). Tracks where the skip marker needs to be written and where
  // write_pos should advance to, so that commit_write can publish everything
  // with a single release store.
  int64_t pending_write_pos;
  uint32_t pending_skip_offset;
  uint32_t pending_skip_bytes;
  uint32_t pending_entry_offset;
} iree_spsc_queue_t;

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

// Returns the total memory required for a queue with the given data capacity.
// |capacity| must be a power of two and >= IREE_SPSC_QUEUE_MIN_CAPACITY.
static inline iree_host_size_t iree_spsc_queue_required_size(
    uint32_t capacity) {
  return IREE_SPSC_QUEUE_HEADER_SIZE + (iree_host_size_t)capacity;
}

// Initializes a new queue in |memory| of |memory_size| bytes.
//
// Writes the header (magic, version, capacity, entry_alignment) and zeroes
// all positions. |capacity| must be a power of two and >=
// IREE_SPSC_QUEUE_MIN_CAPACITY. |memory_size| must be >=
// iree_spsc_queue_required_size(capacity).
iree_status_t iree_spsc_queue_initialize(void* memory,
                                         iree_host_size_t memory_size,
                                         uint32_t capacity,
                                         iree_spsc_queue_t* out_queue);

// Opens an existing queue from |memory| of |memory_size| bytes.
//
// Validates magic and version. Does NOT reset positions — the producer and
// consumer resume from wherever positions are. Returns an error if the header
// is invalid (bad magic, version mismatch, non-power-of-two capacity, etc.).
iree_status_t iree_spsc_queue_open(void* memory, iree_host_size_t memory_size,
                                   iree_spsc_queue_t* out_queue);

//===----------------------------------------------------------------------===//
// Producer API (single-threaded — not safe with concurrent producers)
//===----------------------------------------------------------------------===//

// Writes |data| of |length| bytes into the queue as a single entry.
//
// Returns true if the entry was written, false if there is insufficient space.
// This is the simple one-shot API; for zero-copy writes use
// iree_spsc_queue_begin_write/commit_write.
bool iree_spsc_queue_write(iree_spsc_queue_t* queue, const void* data,
                           iree_host_size_t length);

// Reserves space for an entry of |length| payload bytes.
//
// On success, returns a pointer to the payload region where the caller writes
// directly (zero-copy). On failure (insufficient space), returns NULL.
//
// The caller MUST call iree_spsc_queue_commit_write() after writing the
// payload. Between begin_write and commit_write, no other producer operations
// may be called.
void* iree_spsc_queue_begin_write(iree_spsc_queue_t* queue,
                                  iree_host_size_t length);

// Commits a previously reserved write, making it visible to the consumer.
// |length| must match the length passed to the preceding begin_write.
void iree_spsc_queue_commit_write(iree_spsc_queue_t* queue,
                                  iree_host_size_t length);

//===----------------------------------------------------------------------===//
// Consumer API (single-threaded — not safe with concurrent consumers)
//===----------------------------------------------------------------------===//

// Reads the next entry from the queue into |out_data|.
//
// Copies up to |out_capacity| bytes of payload into |out_data| and sets
// |*out_length| to the actual payload length. Advances read_position.
//
// Returns true if an entry was read, false if the queue is empty.
// If the payload exceeds |out_capacity|, the entry is consumed but the data
// is truncated. Callers should size |out_data| to the maximum expected message.
bool iree_spsc_queue_read(iree_spsc_queue_t* queue, void* out_data,
                          iree_host_size_t out_capacity,
                          iree_host_size_t* out_length);

// Returns a pointer to the next entry's payload without consuming it.
//
// Sets |*out_length| to the payload length. Returns NULL if the queue is empty.
// The returned pointer is valid until the next call to
// iree_spsc_queue_consume() or iree_spsc_queue_read().
const void* iree_spsc_queue_peek(iree_spsc_queue_t* queue,
                                 iree_host_size_t* out_length);

// Consumes the entry most recently returned by iree_spsc_queue_peek().
// Must be called exactly once after each successful peek.
void iree_spsc_queue_consume(iree_spsc_queue_t* queue);

//===----------------------------------------------------------------------===//
// Query API (safe from either producer or consumer side)
//===----------------------------------------------------------------------===//

// Returns true if there are entries available to read.
static inline bool iree_spsc_queue_can_read(const iree_spsc_queue_t* queue) {
  int64_t write_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->write_position->value,
                       iree_memory_order_acquire);
  int64_t read_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->read_position->value,
                       iree_memory_order_acquire);
  return write_pos > read_pos;
}

// Returns the approximate number of committed bytes available for reading.
// The producer may commit more entries between this call and the caller's use,
// so the actual amount may be larger.
static inline iree_host_size_t iree_spsc_queue_read_available(
    const iree_spsc_queue_t* queue) {
  int64_t write_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->write_position->value,
                       iree_memory_order_acquire);
  int64_t read_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->read_position->value,
                       iree_memory_order_acquire);
  return (iree_host_size_t)(write_pos - read_pos);
}

// Returns the approximate number of bytes available for writing.
// The consumer may advance read_position between this call and the caller's
// use, so the actual amount may be larger.
static inline iree_host_size_t iree_spsc_queue_write_available(
    const iree_spsc_queue_t* queue) {
  int64_t write_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->write_position->value,
                       iree_memory_order_acquire);
  int64_t read_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->read_position->value,
                       iree_memory_order_acquire);
  return (iree_host_size_t)(queue->capacity - (write_pos - read_pos));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_SPSC_QUEUE_H_
