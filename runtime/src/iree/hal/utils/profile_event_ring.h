// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_PROFILE_EVENT_RING_H_
#define IREE_HAL_UTILS_PROFILE_EVENT_RING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_profile_event_ring_t
//===----------------------------------------------------------------------===//

// Lossy fixed-capacity profiling event ring.
//
// The ring owns no synchronization and no storage lifetime. Callers decide
// which mutex or higher-level exclusion contract protects positions, event ids,
// drop counts, and record contents.
typedef struct iree_hal_profile_event_ring_t {
  // Storage for fixed-size event records, or NULL when disabled.
  void* records;

  // Byte size of each event record in |records|.
  iree_host_size_t record_size;

  // Power-of-two number of event records in |records|.
  iree_host_size_t capacity;

  // Capacity minus one, for mapping logical positions to physical slots.
  iree_host_size_t mask;

  // Absolute position of the first retained event record.
  uint64_t read_position;

  // Absolute position one past the last retained event record.
  uint64_t write_position;

  // Records dropped since the last successful flush accounted them.
  uint64_t dropped_record_count;

  // Next nonzero event id assigned by this ring.
  uint64_t next_event_id;
} iree_hal_profile_event_ring_t;

// Immutable view of a ring flush attempt.
typedef struct iree_hal_profile_event_ring_snapshot_t {
  // Absolute read position captured for this flush attempt.
  uint64_t read_position;

  // Number of event records captured for this flush attempt.
  iree_host_size_t record_count;

  // Dropped records captured for this flush attempt.
  uint64_t dropped_record_count;

  // Contiguous byte spans covering captured records in ring order.
  iree_const_byte_span_t record_spans[2];

  // Number of initialized entries in |record_spans|.
  iree_host_size_t record_span_count;
} iree_hal_profile_event_ring_snapshot_t;

// Initializes |out_ring| over caller-owned record storage.
//
// |capacity| must be zero or a nonzero power of two. |records| may be NULL
// only when |capacity| is zero, which creates a disabled ring.
void iree_hal_profile_event_ring_initialize(
    void* records, iree_host_size_t record_size, iree_host_size_t capacity,
    iree_hal_profile_event_ring_t* out_ring);

// Clears positions, drop counts, event ids, and retained record bytes.
void iree_hal_profile_event_ring_clear(iree_hal_profile_event_ring_t* ring);

// Returns the fixed-size record slot for |position|.
void* iree_hal_profile_event_ring_record_at(
    const iree_hal_profile_event_ring_t* ring, uint64_t position);

// Attempts to reserve one event record.
//
// Returns false when the ring is disabled. Returns false and accounts one
// dropped record when the ring is full. On success, returns the reserved
// logical position and assigned event id.
bool iree_hal_profile_event_ring_try_append(iree_hal_profile_event_ring_t* ring,
                                            uint64_t* out_position,
                                            uint64_t* out_event_id);

// Captures retained records and dropped-record count for a flush attempt.
iree_status_t iree_hal_profile_event_ring_snapshot(
    const iree_hal_profile_event_ring_t* ring,
    iree_hal_profile_event_ring_snapshot_t* out_snapshot);

// Advances |ring| after a successful sink write of |snapshot|.
void iree_hal_profile_event_ring_commit_snapshot(
    iree_hal_profile_event_ring_t* ring,
    const iree_hal_profile_event_ring_snapshot_t* snapshot);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_PROFILE_EVENT_RING_H_
