// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lock-free multiple-producer single-consumer (MPSC) queue for variable-length
// messages operating on caller-provided memory.
//
// Designed for cross-process shared memory: producers and consumer may be in
// different processes mapping the same shared memory region. The queue uses
// monotonically increasing 64-bit positions with CAS-based reservation for
// producers and per-entry acquire-release ordering for the consumer.
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
// The length prefix doubles as the entry state indicator:
//   0x00000000            — reserved but uncommitted (consumer waits)
//   0x00000001-0x7FFFFFFF — committed (consumer delivers this many bytes)
//   0x80000001-0xFFFFFFFE — canceled (consumer skips,
//                           length = value & 0x7FFFFFFF)
//   0xFFFFFFFF            — skip marker (consumer wraps to offset 0)
//
// A skip marker at the end of the data region signals the consumer to wrap
// to offset 0, same as the SPSC queue.
//
// Ordering: entries are delivered in reservation order. If producer A reserves
// before producer B, A's entry appears first even if B commits first. The
// consumer waits (head-of-line blocking) for uncommitted entries — this is
// correct because the transport layer delivers messages in order.
//
// ABI versioning: the header contains a magic number and version. The opener
// validates both and fails loud on mismatch — no forward compatibility.
//
// Usage:
//   Producer (any thread):
//     iree_mpsc_queue_reservation_t reservation;
//     void* ptr = iree_mpsc_queue_begin_write(&queue, length, &reservation);
//     if (ptr) {
//       memcpy(ptr, data, length);
//       iree_mpsc_queue_commit_write(&queue, reservation);
//     }
//
//   Consumer (single thread):
//     iree_host_size_t length;
//     const void* payload = iree_mpsc_queue_peek(&queue, &length);
//     if (payload) {
//       process(payload, length);
//       iree_mpsc_queue_consume(&queue);
//     }
//
// Thread safety: any number of producers may call begin_write/commit_write/
// cancel_write concurrently. The consumer API (peek/consume/read) must be
// called from a single thread.

#ifndef IREE_BASE_INTERNAL_MPSC_QUEUE_H_
#define IREE_BASE_INTERNAL_MPSC_QUEUE_H_

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

// Magic number identifying an MPSC queue header ("IRMQ" in little-endian).
#define IREE_MPSC_QUEUE_MAGIC ((uint32_t)0x514D5249)

// Current ABI version. Exact match required on open — no forward compatibility.
#define IREE_MPSC_QUEUE_VERSION ((uint32_t)1)

// Total bytes consumed by the header region (4 cache lines).
#define IREE_MPSC_QUEUE_HEADER_SIZE ((iree_host_size_t)256)

// Minimum data capacity in bytes. Must be a power of two.
#define IREE_MPSC_QUEUE_MIN_CAPACITY ((uint32_t)64)

// Default byte alignment for entries (length prefix + payload + padding).
#define IREE_MPSC_QUEUE_DEFAULT_ENTRY_ALIGNMENT ((uint32_t)8)

// Sentinel value in the entry length field indicating a skip-to-wrap marker.
// The consumer silently advances past these.
#define IREE_MPSC_QUEUE_SKIP_MARKER UINT32_MAX

// High bit in the entry length field indicating a canceled entry.
// The consumer silently advances past these. Payload length is the remaining
// 31 bits.
#define IREE_MPSC_QUEUE_CANCEL_BIT ((uint32_t)0x80000000u)

// Maximum payload length (2GB - 1). Upper bit is reserved for cancel flag.
#define IREE_MPSC_QUEUE_MAX_PAYLOAD_LENGTH ((uint32_t)0x7FFFFFFFu)

//===----------------------------------------------------------------------===//
// Memory layout structures
//===----------------------------------------------------------------------===//

// Immutable header at offset 0 of the queue memory region.
// Occupies one cache line. Written once during initialize, read-only after.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_mpsc_queue_header_t {
  uint32_t magic;
  uint32_t version;
  // Data region capacity in bytes. Must be a power of two.
  uint32_t capacity;
  // Byte alignment for entries. Must be a power of two, >= 4.
  uint32_t entry_alignment;
  // Reserved for future use. Must be zero.
  uint8_t reserved[48];
} iree_mpsc_queue_header_t;

// Reserve position at offset 64 (cache line 1).
// CAS-advanced by producers to atomically claim space. Read by the consumer
// to determine whether entries exist beyond the read position.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_mpsc_queue_reserve_position_t {
  iree_atomic_int64_t value;
} iree_mpsc_queue_reserve_position_t;

// Consumer read position at offset 128 (cache line 2).
// Written by the consumer, read by producers to determine free space.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_mpsc_queue_read_position_t {
  iree_atomic_int64_t value;
} iree_mpsc_queue_read_position_t;

//===----------------------------------------------------------------------===//
// Reservation handle
//===----------------------------------------------------------------------===//

// Opaque handle returned by begin_write, passed to commit_write or
// cancel_write. Contains the information needed to publish or discard the
// reserved entry. 8 bytes — fits in a register pair.
typedef struct iree_mpsc_queue_reservation_t {
  // Data ring offset where the length prefix lives.
  uint32_t entry_offset;
  // Payload length (written as the length prefix on commit, or with the cancel
  // bit set on cancel).
  uint32_t payload_length;
} iree_mpsc_queue_reservation_t;

//===----------------------------------------------------------------------===//
// Queue handle
//===----------------------------------------------------------------------===//

// View into an initialized MPSC queue.
//
// Holds derived pointers into caller-provided memory — does not own the memory.
// Initialize with iree_mpsc_queue_initialize() (creator) or
// iree_mpsc_queue_open() (opener).
typedef struct iree_mpsc_queue_t {
  iree_mpsc_queue_header_t* header;
  iree_mpsc_queue_reserve_position_t* reserve_position;
  iree_mpsc_queue_read_position_t* read_position;
  uint8_t* data;
  // Cached from header for fast access.
  uint32_t capacity;
  // capacity - 1: for position-to-offset masking.
  uint32_t mask;
  uint32_t entry_alignment;
} iree_mpsc_queue_t;

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

// Returns the total memory required for a queue with the given data capacity.
// |capacity| must be a power of two and >= IREE_MPSC_QUEUE_MIN_CAPACITY.
static inline iree_host_size_t iree_mpsc_queue_required_size(
    uint32_t capacity) {
  return IREE_MPSC_QUEUE_HEADER_SIZE + (iree_host_size_t)capacity;
}

// Initializes a new queue in |memory| of |memory_size| bytes.
//
// Writes the header (magic, version, capacity, entry_alignment) and zeroes
// all positions. |capacity| must be a power of two and >=
// IREE_MPSC_QUEUE_MIN_CAPACITY. |memory_size| must be >=
// iree_mpsc_queue_required_size(capacity).
iree_status_t iree_mpsc_queue_initialize(void* memory,
                                         iree_host_size_t memory_size,
                                         uint32_t capacity,
                                         iree_mpsc_queue_t* out_queue);

// Opens an existing queue from |memory| of |memory_size| bytes.
//
// Validates magic and version. Does NOT reset positions — producers and
// consumer resume from wherever positions are. Returns an error if the header
// is invalid (bad magic, version mismatch, non-power-of-two capacity, etc.).
iree_status_t iree_mpsc_queue_open(void* memory, iree_host_size_t memory_size,
                                   iree_mpsc_queue_t* out_queue);

//===----------------------------------------------------------------------===//
// Producer API (thread-safe — any number of concurrent producers)
//===----------------------------------------------------------------------===//

// Writes |data| of |length| bytes into the queue as a single entry.
//
// Returns true if the entry was written, false if there is insufficient space.
// This is the simple one-shot API; for zero-copy writes use
// iree_mpsc_queue_begin_write/commit_write.
//
// Thread-safe: multiple producers may call this concurrently.
bool iree_mpsc_queue_write(iree_mpsc_queue_t* queue, const void* data,
                           iree_host_size_t length);

// Atomically reserves space for an entry of |length| payload bytes.
//
// On success, returns a pointer to the payload region where the caller writes
// directly (zero-copy) and fills |*out_reservation| with a handle for
// commit_write or cancel_write. On failure (insufficient space), returns NULL.
//
// The caller MUST call either iree_mpsc_queue_commit_write() or
// iree_mpsc_queue_cancel_write() after a successful begin_write.
//
// |length| must be > 0 and <= IREE_MPSC_QUEUE_MAX_PAYLOAD_LENGTH. Returns
// NULL for invalid lengths. Also returns NULL if the aligned entry size
// exceeds the ring capacity (the message can never fit regardless of free
// space — callers should check capacity before entering retry loops).
//
// Thread-safe: multiple producers may call this concurrently.
void* iree_mpsc_queue_begin_write(
    iree_mpsc_queue_t* queue, iree_host_size_t length,
    iree_mpsc_queue_reservation_t* out_reservation);

// Commits a previously reserved write, making it visible to the consumer.
//
// The reservation handle is consumed and must not be reused. After this call,
// the producer holds no queue resources.
//
// Thread-safe: may be called concurrently with other producer operations.
void iree_mpsc_queue_commit_write(iree_mpsc_queue_t* queue,
                                  iree_mpsc_queue_reservation_t reservation);

// Cancels a previously reserved write without publishing any data.
//
// The consumer will skip past this entry. The reserved space is not reclaimed
// until the consumer advances past it.
//
// Thread-safe: may be called concurrently with other producer operations.
void iree_mpsc_queue_cancel_write(iree_mpsc_queue_t* queue,
                                  iree_mpsc_queue_reservation_t reservation);

//===----------------------------------------------------------------------===//
// Consumer API (single-threaded — not safe with concurrent consumers)
//===----------------------------------------------------------------------===//

// Reads the next entry from the queue into |out_data|.
//
// Copies up to |out_capacity| bytes of payload into |out_data| and sets
// |*out_length| to the actual payload length. Advances read_position.
//
// Returns true if an entry was read, false if the queue is empty or the next
// entry is not yet committed (head-of-line blocking).
// If the payload exceeds |out_capacity|, the entry is consumed but the data
// is truncated. Callers should size |out_data| to the maximum expected message.
bool iree_mpsc_queue_read(iree_mpsc_queue_t* queue, void* out_data,
                          iree_host_size_t out_capacity,
                          iree_host_size_t* out_length);

// Returns a pointer to the next committed entry's payload without consuming it.
//
// Sets |*out_length| to the payload length. Returns NULL if the queue is empty
// or the next entry is not yet committed (head-of-line blocking). Skip markers
// and canceled entries are silently advanced past.
//
// The returned pointer is valid until the next call to
// iree_mpsc_queue_consume() or iree_mpsc_queue_read().
const void* iree_mpsc_queue_peek(iree_mpsc_queue_t* queue,
                                 iree_host_size_t* out_length);

// Consumes the entry most recently returned by iree_mpsc_queue_peek().
// Must be called exactly once after each successful peek.
void iree_mpsc_queue_consume(iree_mpsc_queue_t* queue);

//===----------------------------------------------------------------------===//
// Query API (safe from either producer or consumer side)
//===----------------------------------------------------------------------===//

// Returns true if there may be entries available to read.
//
// Note: unlike the SPSC queue, this can return true even when no committed
// entries are available (entries may be reserved but not yet committed).
static inline bool iree_mpsc_queue_can_read(const iree_mpsc_queue_t* queue) {
  int64_t reserve_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->reserve_position->value,
                       iree_memory_order_acquire);
  int64_t read_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->read_position->value,
                       iree_memory_order_acquire);
  return reserve_pos > read_pos;
}

// Returns the approximate number of reserved bytes beyond the read position.
// This includes committed, uncommitted, and canceled entries.
static inline iree_host_size_t iree_mpsc_queue_read_available(
    const iree_mpsc_queue_t* queue) {
  int64_t reserve_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->reserve_position->value,
                       iree_memory_order_acquire);
  int64_t read_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->read_position->value,
                       iree_memory_order_acquire);
  return (iree_host_size_t)(reserve_pos - read_pos);
}

// Returns the approximate number of bytes available for writing.
// The consumer may advance read_position between this call and the caller's
// use, so the actual amount may be larger.
static inline iree_host_size_t iree_mpsc_queue_write_available(
    const iree_mpsc_queue_t* queue) {
  int64_t reserve_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->reserve_position->value,
                       iree_memory_order_acquire);
  int64_t read_pos =
      iree_atomic_load((iree_atomic_int64_t*)&queue->read_position->value,
                       iree_memory_order_acquire);
  return (iree_host_size_t)(queue->capacity - (reserve_pos - read_pos));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_MPSC_QUEUE_H_
