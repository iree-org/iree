// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/profile_event_ring.h"

#include <string.h>

void iree_hal_profile_event_ring_initialize(
    void* records, iree_host_size_t record_size, iree_host_size_t capacity,
    iree_hal_profile_event_ring_t* out_ring) {
  IREE_ASSERT_ARGUMENT(out_ring);
  IREE_ASSERT(capacity == 0 || iree_host_size_is_power_of_two(capacity));
  IREE_ASSERT(capacity == 0 || records != NULL);
  memset(out_ring, 0, sizeof(*out_ring));
  out_ring->records = records;
  out_ring->record_size = record_size;
  out_ring->capacity = capacity;
  out_ring->mask = capacity ? capacity - 1 : 0;
  out_ring->next_event_id = 1;
}

void iree_hal_profile_event_ring_clear(iree_hal_profile_event_ring_t* ring) {
  IREE_ASSERT_ARGUMENT(ring);
  ring->read_position = 0;
  ring->write_position = 0;
  ring->dropped_record_count = 0;
  ring->next_event_id = 1;
  if (ring->records && ring->capacity != 0) {
    memset(ring->records, 0, ring->capacity * ring->record_size);
  }
}

void* iree_hal_profile_event_ring_record_at(
    const iree_hal_profile_event_ring_t* ring, uint64_t position) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_ASSERT(ring->records != NULL);
  IREE_ASSERT(ring->capacity != 0);
  return (uint8_t*)ring->records + (position & ring->mask) * ring->record_size;
}

bool iree_hal_profile_event_ring_try_append(iree_hal_profile_event_ring_t* ring,
                                            uint64_t* out_position,
                                            uint64_t* out_event_id) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_ASSERT_ARGUMENT(out_position);
  IREE_ASSERT_ARGUMENT(out_event_id);
  *out_position = 0;
  *out_event_id = 0;
  if (!ring->records || ring->capacity == 0) return false;

  const uint64_t read_position = ring->read_position;
  const uint64_t write_position = ring->write_position;
  const uint64_t occupied_count = write_position - read_position;
  if (occupied_count >= ring->capacity) {
    ++ring->dropped_record_count;
    return false;
  }

  *out_position = write_position;
  *out_event_id = ring->next_event_id++;
  ring->write_position = write_position + 1;
  return true;
}

iree_status_t iree_hal_profile_event_ring_snapshot(
    const iree_hal_profile_event_ring_t* ring,
    iree_hal_profile_event_ring_snapshot_t* out_snapshot) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_ASSERT_ARGUMENT(out_snapshot);
  memset(out_snapshot, 0, sizeof(*out_snapshot));
  if (!ring->records || ring->capacity == 0) return iree_ok_status();

  out_snapshot->read_position = ring->read_position;
  out_snapshot->record_count =
      (iree_host_size_t)(ring->write_position - ring->read_position);
  out_snapshot->dropped_record_count = ring->dropped_record_count;
  IREE_ASSERT_LE(out_snapshot->record_count, ring->capacity);
  if (out_snapshot->record_count == 0) return iree_ok_status();

  const iree_host_size_t first_record_index =
      (iree_host_size_t)(ring->read_position & ring->mask);
  const iree_host_size_t first_record_count =
      iree_min(out_snapshot->record_count, ring->capacity - first_record_index);
  iree_host_size_t first_byte_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          first_record_count, ring->record_size, &first_byte_length))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile event ring snapshot size overflows");
  }
  out_snapshot->record_spans[out_snapshot->record_span_count++] =
      iree_make_const_byte_span((const uint8_t*)ring->records +
                                    first_record_index * ring->record_size,
                                first_byte_length);

  const iree_host_size_t second_record_count =
      out_snapshot->record_count - first_record_count;
  if (second_record_count != 0) {
    iree_host_size_t second_byte_length = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            second_record_count, ring->record_size, &second_byte_length))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile event ring snapshot size overflows");
    }
    out_snapshot->record_spans[out_snapshot->record_span_count++] =
        iree_make_const_byte_span(ring->records, second_byte_length);
  }
  return iree_ok_status();
}

void iree_hal_profile_event_ring_commit_snapshot(
    iree_hal_profile_event_ring_t* ring,
    const iree_hal_profile_event_ring_snapshot_t* snapshot) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_ASSERT_ARGUMENT(snapshot);
  ring->read_position = snapshot->read_position + snapshot->record_count;
  if (ring->dropped_record_count >= snapshot->dropped_record_count) {
    ring->dropped_record_count -= snapshot->dropped_record_count;
  } else {
    ring->dropped_record_count = 0;
  }
}
