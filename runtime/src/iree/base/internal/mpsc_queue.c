// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/mpsc_queue.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns the total bytes consumed by an entry with |payload_length| bytes,
// including the 4-byte length prefix and alignment padding.
static inline iree_host_size_t iree_mpsc_queue_entry_size(
    uint32_t entry_alignment, iree_host_size_t payload_length) {
  return iree_host_align(sizeof(uint32_t) + payload_length,
                         (iree_host_size_t)entry_alignment);
}

// Populates the queue handle fields from the memory region.
// Assumes the header at |memory| is valid.
static void iree_mpsc_queue_bind(void* memory, iree_mpsc_queue_t* out_queue) {
  uint8_t* base = (uint8_t*)memory;
  out_queue->header = (iree_mpsc_queue_header_t*)base;
  out_queue->reserve_position =
      (iree_mpsc_queue_reserve_position_t*)(base + 1 * 64);
  out_queue->read_position = (iree_mpsc_queue_read_position_t*)(base + 2 * 64);
  out_queue->data = base + IREE_MPSC_QUEUE_HEADER_SIZE;
  out_queue->capacity = out_queue->header->capacity;
  out_queue->mask = out_queue->capacity - 1;
  out_queue->entry_alignment = out_queue->header->entry_alignment;
}

// Validates the header fields of an existing queue.
static iree_status_t iree_mpsc_queue_validate_header(
    const iree_mpsc_queue_header_t* header, iree_host_size_t memory_size) {
  if (IREE_UNLIKELY(header->magic != IREE_MPSC_QUEUE_MAGIC)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "MPSC queue magic mismatch: expected 0x%08X, got 0x%08X",
        IREE_MPSC_QUEUE_MAGIC, header->magic);
  }
  if (IREE_UNLIKELY(header->version != IREE_MPSC_QUEUE_VERSION)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MPSC queue version mismatch: expected %" PRIu32
                            ", got %" PRIu32,
                            IREE_MPSC_QUEUE_VERSION, header->version);
  }
  if (IREE_UNLIKELY(header->capacity < IREE_MPSC_QUEUE_MIN_CAPACITY)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MPSC queue capacity %" PRIu32
                            " is below minimum %" PRIu32,
                            header->capacity, IREE_MPSC_QUEUE_MIN_CAPACITY);
  }
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(
          (iree_host_size_t)header->capacity))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MPSC queue capacity %" PRIu32
                            " is not a power of two",
                            header->capacity);
  }
  if (IREE_UNLIKELY(header->entry_alignment < sizeof(uint32_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MPSC queue entry_alignment %" PRIu32
                            " is less than minimum %" PRIhsz
                            " (sizeof uint32_t)",
                            header->entry_alignment, sizeof(uint32_t));
  }
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(
          (iree_host_size_t)header->entry_alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MPSC queue entry_alignment %" PRIu32
                            " is not a power of two",
                            header->entry_alignment);
  }
  iree_host_size_t required = iree_mpsc_queue_required_size(header->capacity);
  if (IREE_UNLIKELY(memory_size < required)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MPSC queue memory_size %" PRIhsz
                            " is less than required %" PRIhsz
                            " for capacity %" PRIu32,
                            memory_size, required, header->capacity);
  }
  return iree_ok_status();
}

// Returns the available free space given position values.
static inline iree_host_size_t iree_mpsc_queue_free_space(uint32_t capacity,
                                                          int64_t reserve_pos,
                                                          int64_t read_pos) {
  return (iree_host_size_t)(capacity - (reserve_pos - read_pos));
}

// Acquire-loads the length prefix (entry state indicator) at the given data
// ring offset. The acquire ordering pairs with the release-store in
// commit_write/cancel_write/skip_marker.
static inline uint32_t iree_mpsc_queue_load_entry_state(
    const iree_mpsc_queue_t* queue, uint32_t offset) {
  return (uint32_t)iree_atomic_load((iree_atomic_int32_t*)&queue->data[offset],
                                    iree_memory_order_acquire);
}

// Release-stores a value to the length prefix at the given data ring offset.
static inline void iree_mpsc_queue_store_entry_state(
    const iree_mpsc_queue_t* queue, uint32_t offset, uint32_t value) {
  iree_atomic_store((iree_atomic_int32_t*)&queue->data[offset], (int32_t)value,
                    iree_memory_order_release);
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

iree_status_t iree_mpsc_queue_initialize(void* memory,
                                         iree_host_size_t memory_size,
                                         uint32_t capacity,
                                         iree_mpsc_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_queue, 0, sizeof(*out_queue));

  if (IREE_UNLIKELY(!memory)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory must not be NULL");
  }
  if (IREE_UNLIKELY(capacity < IREE_MPSC_QUEUE_MIN_CAPACITY)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capacity %" PRIu32 " is below minimum %" PRIu32,
                            capacity, IREE_MPSC_QUEUE_MIN_CAPACITY);
  }
  if (IREE_UNLIKELY(
          !iree_host_size_is_power_of_two((iree_host_size_t)capacity))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capacity %" PRIu32 " is not a power of two",
                            capacity);
  }
  iree_host_size_t required = iree_mpsc_queue_required_size(capacity);
  if (IREE_UNLIKELY(memory_size < required)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory_size %" PRIhsz
                            " is less than required %" PRIhsz,
                            memory_size, required);
  }

  // Zero the entire region (header + positions + data).
  // Zeroing the data region is critical: the consumer relies on all bytes
  // being 0 at any offset that hasn't been written to yet. After the initial
  // pass the consumer maintains this invariant by zeroing entire entry regions
  // on consume, but the first iteration relies on this memset.
  memset(memory, 0, required);

  // Write the immutable header.
  iree_mpsc_queue_header_t* header = (iree_mpsc_queue_header_t*)memory;
  header->magic = IREE_MPSC_QUEUE_MAGIC;
  header->version = IREE_MPSC_QUEUE_VERSION;
  header->capacity = capacity;
  header->entry_alignment = IREE_MPSC_QUEUE_DEFAULT_ENTRY_ALIGNMENT;

  // Bind the queue handle to the memory region.
  iree_mpsc_queue_bind(memory, out_queue);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_mpsc_queue_open(void* memory, iree_host_size_t memory_size,
                                   iree_mpsc_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_queue, 0, sizeof(*out_queue));

  if (IREE_UNLIKELY(!memory)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory must not be NULL");
  }
  if (IREE_UNLIKELY(memory_size < IREE_MPSC_QUEUE_HEADER_SIZE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory_size %" PRIhsz
                            " is less than header size %" PRIhsz,
                            memory_size, IREE_MPSC_QUEUE_HEADER_SIZE);
  }

  const iree_mpsc_queue_header_t* header =
      (const iree_mpsc_queue_header_t*)memory;
  iree_status_t status = iree_mpsc_queue_validate_header(header, memory_size);

  if (iree_status_is_ok(status)) {
    iree_mpsc_queue_bind(memory, out_queue);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Producer API
//===----------------------------------------------------------------------===//

void* iree_mpsc_queue_begin_write(
    iree_mpsc_queue_t* queue, iree_host_size_t length,
    iree_mpsc_queue_reservation_t* out_reservation) {
  // Validate payload length. Zero-length messages are not supported (the
  // length prefix doubles as the entry state and 0 means "uncommitted").
  // Lengths above MAX_PAYLOAD_LENGTH would collide with the cancel bit or
  // skip marker in the 32-bit length prefix.
  if (IREE_UNLIKELY(length == 0 ||
                    length > IREE_MPSC_QUEUE_MAX_PAYLOAD_LENGTH)) {
    return NULL;
  }

  iree_host_size_t total_entry_bytes =
      iree_mpsc_queue_entry_size(queue->entry_alignment, length);

  // Messages that can never fit regardless of how much space the consumer
  // frees. Returning NULL here is the same as "queue full" — callers must
  // check message size against ring capacity before entering retry loops.
  if (IREE_UNLIKELY(total_entry_bytes > queue->capacity)) {
    return NULL;
  }

  // CAS loop: atomically reserve space in the ring.
  for (;;) {
    int64_t reserve_pos = iree_atomic_load(&queue->reserve_position->value,
                                           iree_memory_order_relaxed);
    int64_t read_pos = iree_atomic_load(&queue->read_position->value,
                                        iree_memory_order_acquire);

    // Check whether the entry fits without wrapping.
    uint32_t physical_offset = (uint32_t)(reserve_pos & queue->mask);
    iree_host_size_t contiguous_at_tail =
        (iree_host_size_t)(queue->capacity - physical_offset);

    int64_t new_reserve_pos;
    uint32_t entry_offset;
    bool needs_skip;
    iree_host_size_t skip_bytes;

    if (total_entry_bytes <= contiguous_at_tail) {
      // Entry fits at the current position without wrapping.
      iree_host_size_t free_space =
          iree_mpsc_queue_free_space(queue->capacity, reserve_pos, read_pos);
      if (free_space < total_entry_bytes) {
        return NULL;
      }
      new_reserve_pos = reserve_pos + (int64_t)total_entry_bytes;
      entry_offset = physical_offset;
      needs_skip = false;
      skip_bytes = 0;
    } else {
      // Entry doesn't fit at the tail — need a skip marker to wrap.
      skip_bytes = contiguous_at_tail;
      iree_host_size_t total_needed = skip_bytes + total_entry_bytes;
      iree_host_size_t free_space =
          iree_mpsc_queue_free_space(queue->capacity, reserve_pos, read_pos);
      if (free_space < total_needed) {
        return NULL;
      }
      new_reserve_pos = reserve_pos + (int64_t)total_needed;
      entry_offset = 0;
      needs_skip = true;
    }

    // Attempt to claim the space.
    if (!iree_atomic_compare_exchange_strong(
            &queue->reserve_position->value, &reserve_pos, new_reserve_pos,
            iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
      // Another producer won — retry with updated reserve_position.
      continue;
    }

    // CAS succeeded — we own the reserved range. Write skip marker if needed.
    if (needs_skip) {
      iree_mpsc_queue_store_entry_state(queue, physical_offset,
                                        IREE_MPSC_QUEUE_SKIP_MARKER);
    }

    // Fill reservation handle.
    out_reservation->entry_offset = entry_offset;
    out_reservation->payload_length = (uint32_t)length;

    // Return pointer to the payload region (after the length prefix).
    return &queue->data[entry_offset + sizeof(uint32_t)];
  }
}

void iree_mpsc_queue_commit_write(iree_mpsc_queue_t* queue,
                                  iree_mpsc_queue_reservation_t reservation) {
  // Release-store the payload length to the length prefix, making the entry
  // visible to the consumer. All payload writes (by the caller between
  // begin_write and commit_write) are ordered before this release-store.
  iree_mpsc_queue_store_entry_state(queue, reservation.entry_offset,
                                    reservation.payload_length);
}

void iree_mpsc_queue_cancel_write(iree_mpsc_queue_t* queue,
                                  iree_mpsc_queue_reservation_t reservation) {
  // Release-store the payload length with the cancel bit set. The consumer
  // will skip past this entry without delivering it.
  iree_mpsc_queue_store_entry_state(
      queue, reservation.entry_offset,
      reservation.payload_length | IREE_MPSC_QUEUE_CANCEL_BIT);
}

bool iree_mpsc_queue_write(iree_mpsc_queue_t* queue, const void* data,
                           iree_host_size_t length) {
  iree_mpsc_queue_reservation_t reservation;
  void* payload = iree_mpsc_queue_begin_write(queue, length, &reservation);
  if (!payload) return false;
  memcpy(payload, data, length);
  iree_mpsc_queue_commit_write(queue, reservation);
  return true;
}

//===----------------------------------------------------------------------===//
// Consumer API
//===----------------------------------------------------------------------===//

const void* iree_mpsc_queue_peek(iree_mpsc_queue_t* queue,
                                 iree_host_size_t* out_length) {
  *out_length = 0;

  int64_t read_pos =
      iree_atomic_load(&queue->read_position->value, iree_memory_order_relaxed);
  int64_t reserve_pos = iree_atomic_load(&queue->reserve_position->value,
                                         iree_memory_order_acquire);

  while (read_pos < reserve_pos) {
    uint32_t physical_offset = (uint32_t)(read_pos & queue->mask);

    // Acquire-load the length prefix to determine entry state.
    uint32_t entry_state =
        iree_mpsc_queue_load_entry_state(queue, physical_offset);

    if (entry_state == 0) {
      // Reserved but uncommitted — head-of-line blocking. The producer has
      // claimed this space but hasn't committed or canceled yet.
      return NULL;
    }

    if (entry_state == IREE_MPSC_QUEUE_SKIP_MARKER) {
      // Skip marker: advance past the remaining tail bytes and zero the
      // entire skip region so the next ring iteration sees clean memory.
      iree_host_size_t skip_bytes =
          (iree_host_size_t)(queue->capacity - physical_offset);
      memset(&queue->data[physical_offset], 0, skip_bytes);
      read_pos += (int64_t)skip_bytes;
      iree_atomic_store(&queue->read_position->value, read_pos,
                        iree_memory_order_release);
      continue;
    }

    if (entry_state & IREE_MPSC_QUEUE_CANCEL_BIT) {
      // Canceled entry: skip past it. Compute the total entry size from the
      // payload length (lower 31 bits) and zero the entire entry region.
      uint32_t payload_length =
          entry_state & IREE_MPSC_QUEUE_MAX_PAYLOAD_LENGTH;
      iree_host_size_t total_entry_bytes = iree_mpsc_queue_entry_size(
          queue->entry_alignment, (iree_host_size_t)payload_length);
      memset(&queue->data[physical_offset], 0, total_entry_bytes);
      read_pos += (int64_t)total_entry_bytes;
      iree_atomic_store(&queue->read_position->value, read_pos,
                        iree_memory_order_release);
      continue;
    }

    // Committed entry. The entry_state value is the payload length.
    *out_length = (iree_host_size_t)entry_state;
    return &queue->data[physical_offset + sizeof(uint32_t)];
  }

  return NULL;  // Queue is empty.
}

void iree_mpsc_queue_consume(iree_mpsc_queue_t* queue) {
  int64_t read_pos =
      iree_atomic_load(&queue->read_position->value, iree_memory_order_relaxed);
  uint32_t physical_offset = (uint32_t)(read_pos & queue->mask);

  // Read the entry length to compute how far to advance.
  uint32_t entry_state =
      iree_mpsc_queue_load_entry_state(queue, physical_offset);

  iree_host_size_t total_entry_bytes = iree_mpsc_queue_entry_size(
      queue->entry_alignment, (iree_host_size_t)entry_state);

  // Zero the entire entry region (prefix + payload + padding) so that the next
  // ring iteration sees clean memory at all byte positions. This is the
  // critical invariant that makes the MPSC queue correct: when a new entry's
  // prefix lands at an offset that was previously inside a larger entry's
  // payload, stale payload data could be misinterpreted as an entry state
  // (committed length, skip marker, or cancel).
  //
  // This zeroing MUST happen on the consumer side rather than the producer side
  // because the producer's CAS advances reserve_position before the producer
  // can zero the prefix, creating a race window: a preempted producer leaves
  // stale data visible to the consumer. The consumer is single-threaded, so
  // its own prior memsets are trivially visible on subsequent ring iterations.
  // Producers see the zeroed memory through the read_position release/acquire
  // chain (memset → release-store read_position → producer acquire-loads
  // read_position in free space check).
  memset(&queue->data[physical_offset], 0, total_entry_bytes);

  read_pos += (int64_t)total_entry_bytes;
  iree_atomic_store(&queue->read_position->value, read_pos,
                    iree_memory_order_release);
}

bool iree_mpsc_queue_read(iree_mpsc_queue_t* queue, void* out_data,
                          iree_host_size_t out_capacity,
                          iree_host_size_t* out_length) {
  iree_host_size_t payload_length = 0;
  const void* payload = iree_mpsc_queue_peek(queue, &payload_length);
  if (!payload) {
    *out_length = 0;
    return false;
  }

  *out_length = payload_length;
  iree_host_size_t copy_length = iree_min(payload_length, out_capacity);
  memcpy(out_data, payload, copy_length);
  iree_mpsc_queue_consume(queue);
  return true;
}
