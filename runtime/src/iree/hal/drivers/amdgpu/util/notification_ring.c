// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/notification_ring.h"

#include <string.h>

#include "iree/hal/utils/resource_set.h"

// All frontier snapshot sizes are multiples of 16 bytes (header=16,
// entry=16 each), so positions within the byte ring stay aligned as long
// as the base is aligned. Verify at compile time.
static_assert(sizeof(iree_hal_amdgpu_frontier_snapshot_t) % 8 == 0,
              "frontier snapshot header must be 8-byte aligned size");
static_assert(sizeof(iree_async_frontier_entry_t) % 8 == 0,
              "frontier entry must be 8-byte aligned size");

// read_frontier_snapshot casts &snapshot->entry_count to
// iree_async_frontier_t*. The region from entry_count to end-of-header must be
// exactly sizeof(frontier_t) so that the FAM entries[] of the cast frontier
// align with the snapshot's trailing entries.
static_assert(sizeof(iree_hal_amdgpu_frontier_snapshot_t) -
                      offsetof(iree_hal_amdgpu_frontier_snapshot_t,
                               entry_count) ==
                  sizeof(iree_async_frontier_t),
              "snapshot entry_count region must match frontier_t layout");

static inline iree_host_size_t
iree_hal_amdgpu_notification_ring_frontier_offset(
    const iree_hal_amdgpu_notification_ring_t* ring,
    iree_host_size_t position) {
  return position & (ring->frontier_ring.capacity - 1);
}

static inline uint64_t iree_hal_amdgpu_notification_ring_load_position(
    const iree_atomic_int64_t* position, iree_memory_order_t memory_order) {
  return (uint64_t)iree_atomic_load(position, memory_order);
}

static inline void iree_hal_amdgpu_notification_ring_store_position(
    iree_atomic_int64_t* position, uint64_t value,
    iree_memory_order_t memory_order) {
  iree_atomic_store(position, (int64_t)value, memory_order);
}

static inline iree_host_size_t
iree_hal_amdgpu_notification_ring_frontier_snapshot_size(
    const iree_hal_amdgpu_frontier_snapshot_t* snapshot) {
  return sizeof(*snapshot) +
         snapshot->entry_count * sizeof(iree_async_frontier_entry_t);
}

iree_status_t iree_hal_amdgpu_reclaim_entry_prepare(
    iree_hal_amdgpu_reclaim_entry_t* entry, iree_arena_block_pool_t* block_pool,
    uint16_t count, iree_hal_resource_t*** out_resources) {
  IREE_ASSERT_ARGUMENT(entry);
  IREE_ASSERT_ARGUMENT(out_resources);
  entry->pre_signal_action.fn = NULL;
  entry->pre_signal_action.user_data = NULL;
  entry->profile_event_first_position = 0;
  entry->profile_event_count = 0;
  entry->queue_device_event_first_position = 0;
  entry->queue_device_event_count = 0;
  entry->resource_set = NULL;
  entry->kernarg_write_position = 0;
  entry->count = 0;
  if (count <= IREE_HAL_AMDGPU_RECLAIM_INLINE_CAPACITY) {
    entry->resources = entry->inline_resources;
  } else {
    iree_host_size_t required_size = 0;
    IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
        0, &required_size,
        IREE_STRUCT_FIELD(count, iree_hal_resource_t*, NULL)));
    if (required_size > block_pool->usable_block_size) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "reclaim overflow (%" PRIhsz
                              " bytes) exceeds block pool block size (%" PRIhsz
                              " bytes)",
                              required_size, block_pool->usable_block_size);
    }
    iree_arena_block_t* block = NULL;
    void* block_ptr = NULL;
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_block_pool_acquire(block_pool, &block, &block_ptr));
    entry->resources = (iree_hal_resource_t**)block_ptr;
    IREE_TRACE_ZONE_END(z0);
  }
  *out_resources = entry->resources;
  return iree_ok_status();
}

void iree_hal_amdgpu_reclaim_entry_release(
    iree_hal_amdgpu_reclaim_entry_t* entry,
    iree_arena_block_pool_t* block_pool) {
  for (uint16_t i = 0; i < entry->count; ++i) {
    iree_hal_resource_release(entry->resources[i]);
  }
  iree_hal_resource_set_free(entry->resource_set);
  if (entry->resources != entry->inline_resources && entry->resources != NULL) {
    IREE_TRACE_ZONE_BEGIN(z0);
    iree_arena_block_t* block =
        iree_arena_block_trailer(block_pool, entry->resources);
    iree_arena_block_pool_release(block_pool, block, block);
    IREE_TRACE_ZONE_END(z0);
  }
  entry->resources = NULL;
  entry->pre_signal_action.fn = NULL;
  entry->pre_signal_action.user_data = NULL;
  entry->profile_event_first_position = 0;
  entry->profile_event_count = 0;
  entry->queue_device_event_first_position = 0;
  entry->queue_device_event_count = 0;
  entry->resource_set = NULL;
  entry->kernarg_write_position = 0;
  entry->count = 0;
}

static inline void iree_hal_amdgpu_reclaim_entry_execute_pre_signal_action(
    iree_hal_amdgpu_reclaim_entry_t* entry, iree_status_t status) {
  if (!entry->pre_signal_action.fn) return;
  iree_hal_amdgpu_reclaim_action_fn_t fn = entry->pre_signal_action.fn;
  void* user_data = entry->pre_signal_action.user_data;
  entry->pre_signal_action.fn = NULL;
  entry->pre_signal_action.user_data = NULL;
  fn(entry, user_data, status);
}

iree_status_t iree_hal_amdgpu_notification_ring_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_arena_block_pool_t* block_pool,
    uint32_t capacity, iree_allocator_t host_allocator,
    iree_hal_amdgpu_notification_ring_t* out_ring) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_ring);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!iree_host_size_is_power_of_two(capacity)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "notification ring capacity must be a power of two");
  }

  memset(out_ring, 0, sizeof(*out_ring));
  out_ring->libhsa = libhsa;
  out_ring->block_pool = block_pool;
  out_ring->host_allocator = host_allocator;

  // Allocate hot entries + frontier byte ring + reclaim entries in one block.
  // Reserve one extra max-size snapshot so a wrap sentinel's tail padding can
  // coexist with a full hot ring's worth of transition snapshots.
  iree_host_size_t min_frontier_ring_capacity = 0;
  if (!iree_host_size_checked_mul_add(
          IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_SIZE,
          (iree_host_size_t)capacity,
          IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_SIZE,
          &min_frontier_ring_capacity)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "notification ring frontier snapshot capacity overflow");
  }
  iree_host_size_t frontier_ring_capacity =
      iree_host_size_next_power_of_two(min_frontier_ring_capacity);
  if (!iree_host_size_is_power_of_two(frontier_ring_capacity)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "notification ring frontier snapshot capacity overflow");
  }

  iree_host_size_t entries_offset = 0;
  iree_host_size_t frontier_ring_offset = 0;
  iree_host_size_t reclaim_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &total_size,
              IREE_STRUCT_FIELD(capacity, iree_hal_amdgpu_notification_entry_t,
                                &entries_offset),
              IREE_STRUCT_ARRAY_FIELD_ALIGNED(
                  frontier_ring_capacity, 1, uint8_t,
                  iree_alignof(iree_hal_amdgpu_frontier_snapshot_t),
                  &frontier_ring_offset),
              IREE_STRUCT_FIELD(capacity, iree_hal_amdgpu_reclaim_entry_t,
                                &reclaim_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, &out_ring->storage));
  memset(out_ring->storage, 0, total_size);
  uint8_t* base = (uint8_t*)out_ring->storage;
  out_ring->entries =
      (iree_hal_amdgpu_notification_entry_t*)(base + entries_offset);
  iree_hal_amdgpu_notification_ring_store_position(&out_ring->write, 0,
                                                   iree_memory_order_release);
  iree_hal_amdgpu_notification_ring_store_position(&out_ring->read, 0,
                                                   iree_memory_order_release);
  out_ring->frontier_ring.data = base + frontier_ring_offset;
  out_ring->frontier_ring.capacity = frontier_ring_capacity;
  iree_hal_amdgpu_notification_ring_store_position(
      &out_ring->frontier_ring.write, 0, iree_memory_order_release);
  iree_hal_amdgpu_notification_ring_store_position(
      &out_ring->frontier_ring.read, 0, iree_memory_order_release);
  out_ring->reclaim_entries =
      (iree_hal_amdgpu_reclaim_entry_t*)(base + reclaim_offset);
  out_ring->capacity = capacity;
  iree_atomic_store(&out_ring->epoch.last_drained, 0,
                    iree_memory_order_release);

  // Create the epoch signal.
  iree_status_t status = iree_hsa_amd_signal_create(
      IREE_LIBHSA(libhsa), IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE,
      /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0,
      &out_ring->epoch.signal);

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_notification_ring_deinitialize(out_ring);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_notification_ring_deinitialize(
    iree_hal_amdgpu_notification_ring_t* ring) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release any outstanding reclaim entries (should have been drained, but
  // handle partial teardown gracefully).
  if (ring->reclaim_entries && ring->block_pool) {
    uint64_t last_drained = (uint64_t)iree_atomic_load(
        &ring->epoch.last_drained, iree_memory_order_acquire);
    for (uint64_t epoch = last_drained; epoch < ring->epoch.next_submission;
         ++epoch) {
      uint32_t index = (uint32_t)(epoch & (ring->capacity - 1));
      iree_hal_amdgpu_reclaim_entry_release(&ring->reclaim_entries[index],
                                            ring->block_pool);
    }
  }

  if (ring->epoch.signal.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_signal_destroy_raw(ring->libhsa, ring->epoch.signal));
    ring->epoch.signal.handle = 0;
  }
  iree_allocator_free(ring->host_allocator, ring->storage);
  ring->storage = NULL;
  ring->entries = NULL;
  ring->reclaim_entries = NULL;
  ring->frontier_ring.data = NULL;

  IREE_TRACE_ZONE_END(z0);
}

hsa_signal_t iree_hal_amdgpu_notification_ring_epoch_signal(
    const iree_hal_amdgpu_notification_ring_t* ring) {
  return ring->epoch.signal;
}

uint64_t iree_hal_amdgpu_notification_ring_advance_epoch(
    iree_hal_amdgpu_notification_ring_t* ring) {
  return ++ring->epoch.next_submission;
}

iree_status_t iree_hal_amdgpu_notification_ring_reserve(
    const iree_hal_amdgpu_notification_ring_t* ring,
    iree_host_size_t entry_count, iree_host_size_t frontier_snapshot_count) {
  IREE_ASSERT_ARGUMENT(ring);

  uint64_t last_drained = (uint64_t)iree_atomic_load(&ring->epoch.last_drained,
                                                     iree_memory_order_acquire);
  if (ring->epoch.next_submission - last_drained >= ring->capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "notification ring reclaim capacity exhausted (pending_epochs=%" PRIu64
        ", capacity=%u)",
        ring->epoch.next_submission - last_drained, ring->capacity);
  }

  uint64_t write = iree_hal_amdgpu_notification_ring_load_position(
      &ring->write, iree_memory_order_relaxed);
  uint64_t read = iree_hal_amdgpu_notification_ring_load_position(
      &ring->read, iree_memory_order_acquire);
  if (entry_count > ring->capacity ||
      write - read + entry_count > ring->capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "notification ring capacity exhausted (available=%" PRIu64
        ", required=%" PRIhsz ")",
        ring->capacity - (write - read), entry_count);
  }

  if (frontier_snapshot_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t reserved_snapshot_bytes = 0;
  if (!iree_host_size_checked_mul_add(
          IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_SIZE, frontier_snapshot_count,
          IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_SIZE,
          &reserved_snapshot_bytes)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "notification ring frontier snapshot reservation overflow");
  }

  iree_host_size_t frontier_write =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.write, iree_memory_order_relaxed);
  iree_host_size_t frontier_read =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.read, iree_memory_order_acquire);
  iree_host_size_t frontier_occupied = frontier_write - frontier_read;
  if (frontier_occupied + reserved_snapshot_bytes >
      ring->frontier_ring.capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "notification ring frontier snapshot capacity exhausted "
        "(available=%" PRIhsz ", required=%" PRIhsz ")",
        ring->frontier_ring.capacity - frontier_occupied,
        reserved_snapshot_bytes);
  }

  return iree_ok_status();
}

bool iree_hal_amdgpu_notification_ring_can_reserve(
    const iree_hal_amdgpu_notification_ring_t* ring,
    iree_host_size_t entry_count, iree_host_size_t frontier_snapshot_count) {
  IREE_ASSERT_ARGUMENT(ring);

  const uint64_t last_drained = (uint64_t)iree_atomic_load(
      &ring->epoch.last_drained, iree_memory_order_acquire);
  if (ring->epoch.next_submission - last_drained >= ring->capacity) {
    return false;
  }

  const uint64_t write = iree_hal_amdgpu_notification_ring_load_position(
      &ring->write, iree_memory_order_relaxed);
  const uint64_t read = iree_hal_amdgpu_notification_ring_load_position(
      &ring->read, iree_memory_order_acquire);
  if (entry_count > ring->capacity ||
      write - read + entry_count > ring->capacity) {
    return false;
  }

  if (frontier_snapshot_count == 0) {
    return true;
  }

  iree_host_size_t reserved_snapshot_bytes = 0;
  if (!iree_host_size_checked_mul_add(
          IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_SIZE, frontier_snapshot_count,
          IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_SIZE,
          &reserved_snapshot_bytes)) {
    return false;
  }

  const iree_host_size_t frontier_write =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.write, iree_memory_order_relaxed);
  const iree_host_size_t frontier_read =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.read, iree_memory_order_acquire);
  return frontier_write - frontier_read + reserved_snapshot_bytes <=
         ring->frontier_ring.capacity;
}

void iree_hal_amdgpu_notification_ring_push(
    iree_hal_amdgpu_notification_ring_t* ring, uint64_t submission_epoch,
    iree_async_semaphore_t* semaphore, uint64_t timeline_value,
    iree_hal_amdgpu_notification_entry_flags_t flags) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_ASSERT_ARGUMENT(semaphore);

  uint64_t write = iree_hal_amdgpu_notification_ring_load_position(
      &ring->write, iree_memory_order_relaxed);
  uint64_t read = iree_hal_amdgpu_notification_ring_load_position(
      &ring->read, iree_memory_order_acquire);
  IREE_ASSERT(write - read < ring->capacity, "notification ring overflow");

  uint32_t index = (uint32_t)(write & (ring->capacity - 1));
  iree_hal_amdgpu_notification_entry_t* entry = &ring->entries[index];

  entry->semaphore = semaphore;
  entry->timeline_value = timeline_value;
  entry->submission_epoch = submission_epoch;
  entry->flags = flags;
  entry->reserved0 = 0;

  iree_hal_amdgpu_notification_ring_store_position(&ring->write, write + 1,
                                                   iree_memory_order_release);
}

void iree_hal_amdgpu_notification_ring_push_frontier_snapshot(
    iree_hal_amdgpu_notification_ring_t* ring, uint64_t epoch,
    const iree_async_frontier_t* frontier) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_ASSERT_ARGUMENT(frontier);

  uint8_t entry_count = frontier->entry_count;
  IREE_ASSERT(entry_count <= IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT,
              "frontier snapshot exceeds notification ring storage capacity");
  iree_host_size_t snapshot_size =
      sizeof(iree_hal_amdgpu_frontier_snapshot_t) +
      entry_count * sizeof(iree_async_frontier_entry_t);

  iree_host_size_t write =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.write, iree_memory_order_relaxed);
  iree_host_size_t read =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.read, iree_memory_order_acquire);
  IREE_ASSERT(write - read + snapshot_size <= ring->frontier_ring.capacity,
              "notification ring frontier snapshot overflow");

  iree_host_size_t write_offset =
      iree_hal_amdgpu_notification_ring_frontier_offset(ring, write);
  iree_host_size_t remaining = ring->frontier_ring.capacity - write_offset;

  // If the snapshot doesn't fit in the remaining space, write a sentinel
  // header and wrap to byte 0. All snapshot sizes are multiples of
  // sizeof(frontier_snapshot_t), so remaining is also a multiple — there
  // is always room for a sentinel header if remaining > 0.
  if (remaining < snapshot_size) {
    IREE_ASSERT(write - read + remaining + snapshot_size <=
                    ring->frontier_ring.capacity,
                "notification ring frontier snapshot overflow");
    if (remaining >= sizeof(iree_hal_amdgpu_frontier_snapshot_t)) {
      iree_hal_amdgpu_frontier_snapshot_t* sentinel =
          (iree_hal_amdgpu_frontier_snapshot_t*)(ring->frontier_ring.data +
                                                 write_offset);
      sentinel->entry_count = IREE_HAL_AMDGPU_FRONTIER_SNAPSHOT_SENTINEL;
    }
    write += remaining;
    write_offset = 0;
  }

  // Write the snapshot at the current position.
  iree_hal_amdgpu_frontier_snapshot_t* snapshot =
      (iree_hal_amdgpu_frontier_snapshot_t*)(ring->frontier_ring.data +
                                             write_offset);
  snapshot->epoch = epoch;
  snapshot->entry_count = entry_count;
  memset(snapshot->reserved, 0, sizeof(snapshot->reserved));
  if (entry_count > 0) {
    memcpy(snapshot + 1, frontier->entries,
           entry_count * sizeof(iree_async_frontier_entry_t));
  }

  iree_hal_amdgpu_notification_ring_store_position(&ring->frontier_ring.write,
                                                   write + snapshot_size,
                                                   iree_memory_order_release);
}

static const iree_hal_amdgpu_frontier_snapshot_t*
iree_hal_amdgpu_notification_ring_frontier_snapshot_at(
    iree_hal_amdgpu_notification_ring_t* ring, iree_host_size_t* inout_read) {
  iree_host_size_t read_offset =
      iree_hal_amdgpu_notification_ring_frontier_offset(ring, *inout_read);
  const iree_hal_amdgpu_frontier_snapshot_t* snapshot =
      (const iree_hal_amdgpu_frontier_snapshot_t*)(ring->frontier_ring.data +
                                                   read_offset);
  if (snapshot->entry_count == IREE_HAL_AMDGPU_FRONTIER_SNAPSHOT_SENTINEL) {
    *inout_read += ring->frontier_ring.capacity - read_offset;
    snapshot =
        (const iree_hal_amdgpu_frontier_snapshot_t*)ring->frontier_ring.data;
  }
  return snapshot;
}

static void iree_hal_amdgpu_notification_ring_discard_stale_frontier_snapshots(
    iree_hal_amdgpu_notification_ring_t* ring, uint64_t last_drained_epoch) {
  iree_host_size_t read =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.read, iree_memory_order_relaxed);
  iree_host_size_t write =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.write, iree_memory_order_acquire);
  const iree_host_size_t original_read = read;
  while (read < write) {
    iree_host_size_t snapshot_read = read;
    const iree_hal_amdgpu_frontier_snapshot_t* snapshot =
        iree_hal_amdgpu_notification_ring_frontier_snapshot_at(ring,
                                                               &snapshot_read);
    if (snapshot->epoch > last_drained_epoch) break;
    read = snapshot_read +
           iree_hal_amdgpu_notification_ring_frontier_snapshot_size(snapshot);
  }
  if (read != original_read) {
    iree_hal_amdgpu_notification_ring_store_position(
        &ring->frontier_ring.read, read, iree_memory_order_release);
  }
}

// Reads the next frontier snapshot from the frontier byte ring. Returns a
// pointer to an iree_async_frontier_t that can be passed to semaphore_signal.
// The returned frontier is only valid until the next read (it points into the
// ring buffer or into a stack-local single_frontier).
//
// If the ring is empty, returns |fallback|.
static const iree_async_frontier_t*
iree_hal_amdgpu_notification_ring_read_frontier_snapshot(
    iree_hal_amdgpu_notification_ring_t* ring,
    const iree_async_frontier_t* fallback) {
  iree_host_size_t read =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.read, iree_memory_order_relaxed);
  iree_host_size_t write =
      (iree_host_size_t)iree_hal_amdgpu_notification_ring_load_position(
          &ring->frontier_ring.write, iree_memory_order_acquire);
  if (read == write) {
    return fallback;
  }

  const iree_hal_amdgpu_frontier_snapshot_t* snapshot =
      iree_hal_amdgpu_notification_ring_frontier_snapshot_at(ring, &read);

  iree_host_size_t snapshot_size =
      iree_hal_amdgpu_notification_ring_frontier_snapshot_size(snapshot);
  iree_hal_amdgpu_notification_ring_store_position(&ring->frontier_ring.read,
                                                   read + snapshot_size,
                                                   iree_memory_order_release);

  // The snapshot's {entry_count, reserved[7], entries[]} region starting at
  // &snapshot->entry_count is layout-compatible with iree_async_frontier_t.
  // Return a pointer to it — valid until the next read advances past it.
  return (const iree_async_frontier_t*)&snapshot->entry_count;
}

static const iree_async_frontier_t*
iree_hal_amdgpu_notification_ring_read_span_frontier(
    iree_hal_amdgpu_notification_ring_t* ring,
    iree_hal_amdgpu_notification_entry_flags_t flags,
    bool has_transition_snapshot, const iree_async_frontier_t* fallback) {
  if (!has_transition_snapshot ||
      iree_any_bit_set(
          flags,
          IREE_HAL_AMDGPU_NOTIFICATION_ENTRY_FLAG_OMIT_FRONTIER_SNAPSHOT)) {
    return fallback;
  }
  return iree_hal_amdgpu_notification_ring_read_frontier_snapshot(ring,
                                                                  fallback);
}

iree_host_size_t iree_hal_amdgpu_notification_ring_drain(
    iree_hal_amdgpu_notification_ring_t* ring,
    const iree_async_frontier_t* fallback_frontier,
    iree_hal_amdgpu_reclaim_retire_fn_t retire_fn, void* retire_user_data,
    uint64_t* out_kernarg_reclaim_position) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_ASSERT_ARGUMENT(out_kernarg_reclaim_position);

  *out_kernarg_reclaim_position = 0;

  // Early out if the ring was never initialized or already deinitialized.
  if (!ring->epoch.signal.handle) return 0;

  hsa_signal_value_t signal_value = iree_hsa_signal_load_scacquire(
      IREE_LIBHSA(ring->libhsa), ring->epoch.signal);
  uint64_t current_epoch =
      (uint64_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE - signal_value);

  uint64_t previous_drained = (uint64_t)iree_atomic_load(
      &ring->epoch.last_drained, iree_memory_order_relaxed);
  if (current_epoch <= previous_drained) return 0;
  iree_hal_amdgpu_notification_ring_discard_stale_frontier_snapshots(
      ring, previous_drained);

  // Execute all pre-signal completion actions first so wrapper-visible state
  // transitions happen-before any semaphore publication for the completed
  // epochs. This is intentionally a separate pass from post-signal resource
  // release to keep queue retire ordering explicit.
  for (uint64_t epoch = previous_drained; epoch < current_epoch; ++epoch) {
    uint32_t reclaim_index = (uint32_t)(epoch & (ring->capacity - 1));
    iree_hal_amdgpu_reclaim_entry_execute_pre_signal_action(
        &ring->reclaim_entries[reclaim_index], iree_ok_status());
  }
  if (retire_fn) {
    for (uint64_t epoch = previous_drained; epoch < current_epoch; ++epoch) {
      uint32_t reclaim_index = (uint32_t)(epoch & (ring->capacity - 1));
      retire_fn(&ring->reclaim_entries[reclaim_index], epoch + 1,
                retire_user_data);
    }
  }

  // Single-slot coalescing: accumulate consecutive same-semaphore entries
  // and signal once per unique semaphore span. This turns N signals into 1
  // for the common case of N dispatches on the same stream semaphore.
  iree_async_semaphore_t* pending_semaphore = NULL;
  uint64_t pending_value = 0;
  iree_hal_amdgpu_notification_entry_flags_t pending_flags =
      IREE_HAL_AMDGPU_NOTIFICATION_ENTRY_FLAG_NONE;
  iree_host_size_t drained_count = 0;
  uint64_t read = iree_hal_amdgpu_notification_ring_load_position(
      &ring->read, iree_memory_order_relaxed);
  uint64_t write = iree_hal_amdgpu_notification_ring_load_position(
      &ring->write, iree_memory_order_acquire);

  while (read < write) {
    uint32_t index = (uint32_t)(read & (ring->capacity - 1));
    iree_hal_amdgpu_notification_entry_t* entry = &ring->entries[index];
    if (entry->submission_epoch > current_epoch) break;

    if (entry->semaphore != pending_semaphore) {
      // Semaphore changed — flush the previous span.
      if (pending_semaphore != NULL) {
        const iree_async_frontier_t* frontier =
            iree_hal_amdgpu_notification_ring_read_span_frontier(
                ring, pending_flags, /*has_transition_snapshot=*/true,
                fallback_frontier);
        iree_status_t signal_status = iree_async_semaphore_signal_untainted(
            pending_semaphore, pending_value, frontier);
        if (IREE_UNLIKELY(!iree_status_is_ok(signal_status))) {
          iree_async_semaphore_fail(pending_semaphore, signal_status);
        }
      }
      pending_semaphore = entry->semaphore;
      pending_value = entry->timeline_value;
      pending_flags = entry->flags;
    } else {
      // Same semaphore, later epoch — take the later value (monotonic).
      pending_value = entry->timeline_value;
      pending_flags |= entry->flags;
    }

    ++read;
    ++drained_count;
  }

  // Flush the final span. If the next unread entry is a different semaphore
  // then a transition snapshot for this completed span has already been
  // written, even though that next entry has not completed yet.
  if (pending_semaphore != NULL) {
    bool has_transition_snapshot = false;
    if (read < write) {
      uint32_t next_index = (uint32_t)(read & (ring->capacity - 1));
      const iree_hal_amdgpu_notification_entry_t* next_entry =
          &ring->entries[next_index];
      has_transition_snapshot = next_entry->semaphore != pending_semaphore;
    }
    const iree_async_frontier_t* frontier =
        iree_hal_amdgpu_notification_ring_read_span_frontier(
            ring, pending_flags, has_transition_snapshot, fallback_frontier);
    iree_status_t signal_status = iree_async_semaphore_signal_untainted(
        pending_semaphore, pending_value, frontier);
    if (IREE_UNLIKELY(!iree_status_is_ok(signal_status))) {
      iree_async_semaphore_fail(pending_semaphore, signal_status);
    }
  }

  // Release retained resources for all completed epochs.
  uint64_t highest_kernarg_position = 0;
  for (uint64_t epoch = previous_drained; epoch < current_epoch; ++epoch) {
    uint32_t reclaim_index = (uint32_t)(epoch & (ring->capacity - 1));
    uint64_t kernarg_write_position =
        ring->reclaim_entries[reclaim_index].kernarg_write_position;
    if (kernarg_write_position > highest_kernarg_position) {
      highest_kernarg_position = kernarg_write_position;
    }
    iree_hal_amdgpu_reclaim_entry_release(&ring->reclaim_entries[reclaim_index],
                                          ring->block_pool);
  }
  iree_atomic_store(&ring->epoch.last_drained, (int64_t)current_epoch,
                    iree_memory_order_release);

  iree_hal_amdgpu_notification_ring_store_position(&ring->read, read,
                                                   iree_memory_order_release);

  *out_kernarg_reclaim_position = highest_kernarg_position;
  return drained_count;
}

iree_host_size_t iree_hal_amdgpu_notification_ring_fail_all(
    iree_hal_amdgpu_notification_ring_t* ring, iree_status_t error_status,
    uint64_t* out_kernarg_reclaim_position) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_ASSERT_ARGUMENT(out_kernarg_reclaim_position);

  *out_kernarg_reclaim_position = 0;

  iree_host_size_t failed_count = 0;
  uint64_t read = iree_hal_amdgpu_notification_ring_load_position(
      &ring->read, iree_memory_order_relaxed);
  uint64_t write = iree_hal_amdgpu_notification_ring_load_position(
      &ring->write, iree_memory_order_acquire);
  while (read < write) {
    uint32_t index = (uint32_t)(read & (ring->capacity - 1));
    iree_hal_amdgpu_notification_entry_t* entry = &ring->entries[index];

    // Check-before-clone: only clone and fail if this semaphore hasn't been
    // failed yet. Avoids cloning status objects (which contain stack traces)
    // for every entry when many entries share a semaphore. The TOCTOU between
    // the load and the CAS inside semaphore_fail is harmless — fail_all runs
    // single-threaded on the proactor, so the only way failure_status is
    // non-zero is from an earlier entry in this same loop.
    if (iree_atomic_load(&entry->semaphore->failure_status,
                         iree_memory_order_acquire) == 0) {
      iree_async_semaphore_fail(entry->semaphore,
                                iree_status_clone(error_status));
    }

    ++read;
    ++failed_count;
  }

  // Release retained resources for all epochs.
  uint64_t highest_kernarg_position = 0;
  uint64_t last_drained = (uint64_t)iree_atomic_load(&ring->epoch.last_drained,
                                                     iree_memory_order_relaxed);
  for (uint64_t epoch = last_drained; epoch < ring->epoch.next_submission;
       ++epoch) {
    uint32_t reclaim_index = (uint32_t)(epoch & (ring->capacity - 1));
    uint64_t kernarg_write_position =
        ring->reclaim_entries[reclaim_index].kernarg_write_position;
    if (kernarg_write_position > highest_kernarg_position) {
      highest_kernarg_position = kernarg_write_position;
    }
    iree_hal_amdgpu_reclaim_entry_execute_pre_signal_action(
        &ring->reclaim_entries[reclaim_index], error_status);
    iree_hal_amdgpu_reclaim_entry_release(&ring->reclaim_entries[reclaim_index],
                                          ring->block_pool);
  }
  iree_atomic_store(&ring->epoch.last_drained,
                    (int64_t)ring->epoch.next_submission,
                    iree_memory_order_release);

  iree_hal_amdgpu_notification_ring_store_position(&ring->read, read,
                                                   iree_memory_order_release);

  *out_kernarg_reclaim_position = highest_kernarg_position;
  return failed_count;
}
