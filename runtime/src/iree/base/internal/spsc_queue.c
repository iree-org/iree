// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/spsc_queue.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns the total bytes consumed by an entry with |payload_length| bytes,
// including the 4-byte length prefix and alignment padding.
static inline iree_host_size_t iree_spsc_queue_entry_size(
    uint32_t entry_alignment, iree_host_size_t payload_length) {
  return iree_host_align(sizeof(uint32_t) + payload_length,
                         (iree_host_size_t)entry_alignment);
}

// Populates the queue handle fields from the memory region.
// Assumes the header at |memory| is valid.
static void iree_spsc_queue_bind(void* memory, iree_spsc_queue_t* out_queue) {
  uint8_t* base = (uint8_t*)memory;
  out_queue->header = (iree_spsc_queue_header_t*)base;
  out_queue->write_position =
      (iree_spsc_queue_write_position_t*)(base + 1 * 64);
  out_queue->read_position = (iree_spsc_queue_read_position_t*)(base + 2 * 64);
  out_queue->cached_read_position =
      (iree_spsc_queue_cached_position_t*)(base + 3 * 64);
  out_queue->data = base + IREE_SPSC_QUEUE_HEADER_SIZE;
  out_queue->capacity = out_queue->header->capacity;
  out_queue->mask = out_queue->capacity - 1;
  out_queue->entry_alignment = out_queue->header->entry_alignment;
}

// Validates the header fields of an existing queue.
static iree_status_t iree_spsc_queue_validate_header(
    const iree_spsc_queue_header_t* header, iree_host_size_t memory_size) {
  if (IREE_UNLIKELY(header->magic != IREE_SPSC_QUEUE_MAGIC)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPSC queue magic mismatch: expected 0x%08X, got 0x%08X",
        IREE_SPSC_QUEUE_MAGIC, header->magic);
  }
  if (IREE_UNLIKELY(header->version != IREE_SPSC_QUEUE_VERSION)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPSC queue version mismatch: expected %" PRIu32
                            ", got %" PRIu32,
                            IREE_SPSC_QUEUE_VERSION, header->version);
  }
  if (IREE_UNLIKELY(header->capacity < IREE_SPSC_QUEUE_MIN_CAPACITY)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPSC queue capacity %" PRIu32
                            " is below minimum %" PRIu32,
                            header->capacity, IREE_SPSC_QUEUE_MIN_CAPACITY);
  }
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(
          (iree_host_size_t)header->capacity))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPSC queue capacity %" PRIu32
                            " is not a power of two",
                            header->capacity);
  }
  if (IREE_UNLIKELY(header->entry_alignment < sizeof(uint32_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPSC queue entry_alignment %" PRIu32
                            " is less than minimum %" PRIhsz
                            " (sizeof uint32_t)",
                            header->entry_alignment, sizeof(uint32_t));
  }
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(
          (iree_host_size_t)header->entry_alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPSC queue entry_alignment %" PRIu32
                            " is not a power of two",
                            header->entry_alignment);
  }
  iree_host_size_t required = iree_spsc_queue_required_size(header->capacity);
  if (IREE_UNLIKELY(memory_size < required)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPSC queue memory_size %" PRIhsz
                            " is less than required %" PRIhsz
                            " for capacity %" PRIu32,
                            memory_size, required, header->capacity);
  }
  return iree_ok_status();
}

// Returns the available free space for the producer given the current write
// position and a (possibly cached) read position.
static inline iree_host_size_t iree_spsc_queue_free_space(uint32_t capacity,
                                                          int64_t write_pos,
                                                          int64_t read_pos) {
  return (iree_host_size_t)(capacity - (write_pos - read_pos));
}

// Checks whether the entry fits at the current write position and computes
// the layout. If the entry would straddle the wrap boundary, records a pending
// skip marker. Does NOT write to the data region or update write_pos — all
// writes happen in commit_write with a single release store.
//
// On success, populates the pending write state in |queue| and returns the
// physical offset for the entry. On failure (insufficient space), returns
// UINT32_MAX.
static uint32_t iree_spsc_queue_plan_write(iree_spsc_queue_t* queue,
                                           iree_host_size_t total_entry_bytes,
                                           int64_t write_pos) {
  uint32_t physical_offset = (uint32_t)(write_pos & queue->mask);
  iree_host_size_t contiguous_at_tail =
      (iree_host_size_t)(queue->capacity - physical_offset);
  if (total_entry_bytes <= contiguous_at_tail) {
    // Entry fits without wrapping. No skip marker needed.
    queue->pending_skip_offset = 0;
    queue->pending_skip_bytes = 0;
    queue->pending_entry_offset = physical_offset;
    queue->pending_write_pos = write_pos;
    return physical_offset;
  }

  // Entry doesn't fit at the tail. A skip marker will be needed.
  uint32_t skip_bytes = (uint32_t)contiguous_at_tail;

  // Check that we have enough total free space for the skip marker plus the
  // entry at offset 0.
  int64_t cached_read = queue->cached_read_position->value;
  iree_host_size_t free_space =
      iree_spsc_queue_free_space(queue->capacity, write_pos, cached_read);
  if (free_space < (iree_host_size_t)skip_bytes + total_entry_bytes) {
    // Refresh the cached read position.
    cached_read = iree_atomic_load(&queue->read_position->value,
                                   iree_memory_order_acquire);
    queue->cached_read_position->value = cached_read;
    free_space =
        iree_spsc_queue_free_space(queue->capacity, write_pos, cached_read);
    if (free_space < (iree_host_size_t)skip_bytes + total_entry_bytes) {
      return UINT32_MAX;  // Not enough space.
    }
  }

  // Record the skip marker for commit_write to apply.
  queue->pending_skip_offset = physical_offset;
  queue->pending_skip_bytes = skip_bytes;
  queue->pending_entry_offset = 0;
  queue->pending_write_pos = write_pos + (int64_t)skip_bytes;
  return 0;  // Entry will go at physical offset 0.
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

iree_status_t iree_spsc_queue_initialize(void* memory,
                                         iree_host_size_t memory_size,
                                         uint32_t capacity,
                                         iree_spsc_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_queue, 0, sizeof(*out_queue));

  if (IREE_UNLIKELY(!memory)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory must not be NULL");
  }
  if (IREE_UNLIKELY(capacity < IREE_SPSC_QUEUE_MIN_CAPACITY)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capacity %" PRIu32 " is below minimum %" PRIu32,
                            capacity, IREE_SPSC_QUEUE_MIN_CAPACITY);
  }
  if (IREE_UNLIKELY(
          !iree_host_size_is_power_of_two((iree_host_size_t)capacity))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capacity %" PRIu32 " is not a power of two",
                            capacity);
  }
  iree_host_size_t required = iree_spsc_queue_required_size(capacity);
  if (IREE_UNLIKELY(memory_size < required)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory_size %" PRIhsz
                            " is less than required %" PRIhsz,
                            memory_size, required);
  }

  // Zero the entire region (header + positions + data).
  memset(memory, 0, required);

  // Write the immutable header.
  iree_spsc_queue_header_t* header = (iree_spsc_queue_header_t*)memory;
  header->magic = IREE_SPSC_QUEUE_MAGIC;
  header->version = IREE_SPSC_QUEUE_VERSION;
  header->capacity = capacity;
  header->entry_alignment = IREE_SPSC_QUEUE_DEFAULT_ENTRY_ALIGNMENT;

  // Bind the queue handle to the memory region.
  iree_spsc_queue_bind(memory, out_queue);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_spsc_queue_open(void* memory, iree_host_size_t memory_size,
                                   iree_spsc_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_queue, 0, sizeof(*out_queue));

  if (IREE_UNLIKELY(!memory)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory must not be NULL");
  }
  if (IREE_UNLIKELY(memory_size < IREE_SPSC_QUEUE_HEADER_SIZE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory_size %" PRIhsz
                            " is less than header size %" PRIhsz,
                            memory_size, IREE_SPSC_QUEUE_HEADER_SIZE);
  }

  const iree_spsc_queue_header_t* header =
      (const iree_spsc_queue_header_t*)memory;
  iree_status_t status = iree_spsc_queue_validate_header(header, memory_size);

  if (iree_status_is_ok(status)) {
    iree_spsc_queue_bind(memory, out_queue);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Producer API
//===----------------------------------------------------------------------===//

void* iree_spsc_queue_begin_write(iree_spsc_queue_t* queue,
                                  iree_host_size_t length) {
  iree_host_size_t total_entry_bytes =
      iree_spsc_queue_entry_size(queue->entry_alignment, length);

  // Fast path: check against cached read position.
  int64_t write_pos = iree_atomic_load(&queue->write_position->value,
                                       iree_memory_order_relaxed);
  int64_t cached_read = queue->cached_read_position->value;
  iree_host_size_t free_space =
      iree_spsc_queue_free_space(queue->capacity, write_pos, cached_read);
  if (free_space < total_entry_bytes) {
    // Slow path: refresh the cached read position from the consumer.
    cached_read = iree_atomic_load(&queue->read_position->value,
                                   iree_memory_order_acquire);
    queue->cached_read_position->value = cached_read;
    free_space =
        iree_spsc_queue_free_space(queue->capacity, write_pos, cached_read);
    if (free_space < total_entry_bytes) {
      return NULL;  // Not enough space.
    }
  }

  // Plan the write layout (skip marker + entry offset) without touching the
  // data region or write_pos. All data writes happen in commit_write.
  uint32_t entry_offset =
      iree_spsc_queue_plan_write(queue, total_entry_bytes, write_pos);
  if (entry_offset == UINT32_MAX) {
    return NULL;  // Skip marker + entry don't fit.
  }

  // Return pointer to the payload region (after the length prefix slot).
  // The caller writes payload here; commit_write fills in the length prefix
  // and publishes everything with a release store.
  return &queue->data[entry_offset + sizeof(uint32_t)];
}

void iree_spsc_queue_commit_write(iree_spsc_queue_t* queue,
                                  iree_host_size_t length) {
  iree_host_size_t total_entry_bytes =
      iree_spsc_queue_entry_size(queue->entry_alignment, length);

  // Write the skip marker if wrapping was needed.
  if (queue->pending_skip_bytes > 0) {
    uint32_t skip_length = IREE_SPSC_QUEUE_SKIP_MARKER;
    memcpy(&queue->data[queue->pending_skip_offset], &skip_length,
           sizeof(uint32_t));
  }

  // Write the length prefix at the entry offset.
  uint32_t payload_length = (uint32_t)length;
  memcpy(&queue->data[queue->pending_entry_offset], &payload_length,
         sizeof(uint32_t));

  // Publish everything with a single release store. The release ordering
  // ensures the skip marker, length prefix, and payload (written by the
  // caller between begin_write and commit_write) are all visible to the
  // consumer before it sees the advanced write_pos.
  int64_t new_write_pos = queue->pending_write_pos + (int64_t)total_entry_bytes;
  iree_atomic_store(&queue->write_position->value, new_write_pos,
                    iree_memory_order_release);
}

bool iree_spsc_queue_write(iree_spsc_queue_t* queue, const void* data,
                           iree_host_size_t length) {
  void* payload = iree_spsc_queue_begin_write(queue, length);
  if (!payload) return false;
  memcpy(payload, data, length);
  iree_spsc_queue_commit_write(queue, length);
  return true;
}

//===----------------------------------------------------------------------===//
// Consumer API
//===----------------------------------------------------------------------===//

const void* iree_spsc_queue_peek(iree_spsc_queue_t* queue,
                                 iree_host_size_t* out_length) {
  *out_length = 0;

  int64_t read_pos =
      iree_atomic_load(&queue->read_position->value, iree_memory_order_relaxed);
  int64_t write_pos = iree_atomic_load(&queue->write_position->value,
                                       iree_memory_order_acquire);

  while (read_pos < write_pos) {
    uint32_t physical_offset = (uint32_t)(read_pos & queue->mask);

    // Read the entry length.
    uint32_t entry_length = 0;
    memcpy(&entry_length, &queue->data[physical_offset], sizeof(uint32_t));

    if (entry_length == IREE_SPSC_QUEUE_SKIP_MARKER) {
      // Skip marker: advance past the remaining tail bytes and continue.
      iree_host_size_t skip_bytes =
          (iree_host_size_t)(queue->capacity - physical_offset);
      read_pos += (int64_t)skip_bytes;
      // Publish the advanced read position so the producer can reclaim space.
      iree_atomic_store(&queue->read_position->value, read_pos,
                        iree_memory_order_release);
      continue;
    }

    // Valid entry found.
    *out_length = (iree_host_size_t)entry_length;
    return &queue->data[physical_offset + sizeof(uint32_t)];
  }

  return NULL;  // Queue is empty.
}

void iree_spsc_queue_consume(iree_spsc_queue_t* queue) {
  int64_t read_pos =
      iree_atomic_load(&queue->read_position->value, iree_memory_order_relaxed);
  uint32_t physical_offset = (uint32_t)(read_pos & queue->mask);

  // Read the entry length to compute how far to advance.
  uint32_t entry_length = 0;
  memcpy(&entry_length, &queue->data[physical_offset], sizeof(uint32_t));

  iree_host_size_t total_entry_bytes = iree_spsc_queue_entry_size(
      queue->entry_alignment, (iree_host_size_t)entry_length);
  read_pos += (int64_t)total_entry_bytes;
  iree_atomic_store(&queue->read_position->value, read_pos,
                    iree_memory_order_release);
}

bool iree_spsc_queue_read(iree_spsc_queue_t* queue, void* out_data,
                          iree_host_size_t out_capacity,
                          iree_host_size_t* out_length) {
  iree_host_size_t payload_length = 0;
  const void* payload = iree_spsc_queue_peek(queue, &payload_length);
  if (!payload) {
    *out_length = 0;
    return false;
  }

  *out_length = payload_length;
  iree_host_size_t copy_length = iree_min(payload_length, out_capacity);
  memcpy(out_data, payload, copy_length);
  iree_spsc_queue_consume(queue);
  return true;
}
