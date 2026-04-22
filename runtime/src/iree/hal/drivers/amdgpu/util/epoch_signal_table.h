// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_EPOCH_SIGNAL_TABLE_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_EPOCH_SIGNAL_TABLE_H_

#include <string.h>

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_epoch_signal_table_t
//===----------------------------------------------------------------------===//

// Flat lookup table mapping (device_index, queue_index) to the hsa_signal_t
// epoch signal for that queue. Shared read-only across all queues on the same
// machine during normal operation; mutated only during queue init/deinit.
//
// The epoch signal is the single hsa_signal_t that the CP decrements on each
// AQL packet completion. It is the mechanism by which tier 2 (device-side)
// cross-queue waits work: a queue waiting on a peer emits an AQL barrier-value
// packet referencing the peer's epoch signal with a condition that fires when
// the peer's epoch reaches the required value.
//
// For producer-frontier-exact cross-queue waits, the submission path reads the
// semaphore's last_signal cache to identify the producer axis/epoch directly,
// then does one lookup here to map that producer axis to an hsa_signal_t for a
// single barrier-value packet. For multi-dependency cases, TP collective joins
// can still require N lookups for N undominated peer axes discovered from the
// semaphore frontier.
//
// The table is allocated once at device group init, sized from the topology
// (device_count * queue_stride). Each queue registers its epoch signal during
// init and deregisters during deinit. Lookup verifies the axis's session epoch
// and machine index match this table's — axes from other sessions or machines
// fail the lookup (tier 3 fallback).
typedef struct iree_hal_amdgpu_epoch_signal_table_t {
  // Session epoch from the axis encoding. Used to verify that a lookup axis
  // belongs to the same session as this table. Prevents cross-session aliasing
  // if axes from different sessions happen to share device/queue indices.
  uint8_t session_epoch;
  // Machine index from the axis encoding. Used to verify that a lookup axis
  // belongs to the same machine. Cross-machine waits use tier 3 (host deferral)
  // since there is no shared HSA signal.
  uint8_t machine_index;
  // Maximum queues per device (uniform across all devices in the topology).
  // Columns in the 2D array: signals[device * queue_stride + queue].
  uint8_t queue_stride;
  // Number of devices in the table. Rows in the 2D array.
  uint8_t device_count;
  uint8_t reserved[4];
  // Flat 2D array of epoch signals indexed by [device_index * queue_stride +
  // queue_index]. Unregistered slots have handle == 0 (null signal). Registered
  // slots contain the epoch signal from the queue's notification ring.
  hsa_signal_t signals[];
} iree_hal_amdgpu_epoch_signal_table_t;

// Returns the total allocation size in bytes for an epoch signal table with
// the given dimensions.
static inline iree_host_size_t iree_hal_amdgpu_epoch_signal_table_size(
    uint8_t device_count, uint8_t queue_stride) {
  // uint8_t * uint8_t cannot overflow iree_host_size_t.
  return sizeof(iree_hal_amdgpu_epoch_signal_table_t) +
         (iree_host_size_t)device_count * queue_stride * sizeof(hsa_signal_t);
}

// Initializes an epoch signal table in caller-provided memory. The caller
// must have allocated at least iree_hal_amdgpu_epoch_signal_table_size()
// bytes. All signal slots are zeroed (unregistered).
static inline void iree_hal_amdgpu_epoch_signal_table_initialize(
    iree_hal_amdgpu_epoch_signal_table_t* table, uint8_t session_epoch,
    uint8_t machine_index, uint8_t device_count, uint8_t queue_stride) {
  table->session_epoch = session_epoch;
  table->machine_index = machine_index;
  table->queue_stride = queue_stride;
  table->device_count = device_count;
  memset(table->reserved, 0, sizeof(table->reserved));
  memset(table->signals, 0,
         (iree_host_size_t)device_count * queue_stride * sizeof(hsa_signal_t));
}

// Registers a queue's epoch signal in the table. Called during queue init
// after the notification ring (which owns the epoch signal) is created.
//
// The slot must not already be registered (programming error if it is).
static inline void iree_hal_amdgpu_epoch_signal_table_register(
    iree_hal_amdgpu_epoch_signal_table_t* table, uint8_t device_index,
    uint8_t queue_index, hsa_signal_t epoch_signal) {
  IREE_ASSERT(device_index < table->device_count, "device_index out of range");
  IREE_ASSERT(queue_index < table->queue_stride, "queue_index out of range");
  iree_host_size_t slot =
      (iree_host_size_t)device_index * table->queue_stride + queue_index;
  IREE_ASSERT(table->signals[slot].handle == 0,
              "epoch signal slot already registered");
  IREE_ASSERT(epoch_signal.handle != 0, "cannot register null epoch signal");
  table->signals[slot] = epoch_signal;
}

// Deregisters a queue's epoch signal from the table. Called during queue
// deinit before the notification ring (which owns the epoch signal) is
// destroyed. The slot must currently be registered.
static inline void iree_hal_amdgpu_epoch_signal_table_deregister(
    iree_hal_amdgpu_epoch_signal_table_t* table, uint8_t device_index,
    uint8_t queue_index) {
  IREE_ASSERT(device_index < table->device_count, "device_index out of range");
  IREE_ASSERT(queue_index < table->queue_stride, "queue_index out of range");
  iree_host_size_t slot =
      (iree_host_size_t)device_index * table->queue_stride + queue_index;
  IREE_ASSERT(table->signals[slot].handle != 0,
              "epoch signal slot not registered");
  table->signals[slot].handle = 0;
}

// Looks up the epoch signal for the queue identified by |axis|. Returns true
// and writes the signal to |out_signal| if the axis matches this table's
// session/machine, is a QUEUE-domain axis, is within bounds, and the slot
// is registered. Returns false otherwise (caller should fall back to tier 3).
//
// This is the hot-path lookup for tier 2 barrier emission. Two byte
// comparisons (session + machine), one domain check, two bounds checks,
// one array index. ~15 instructions.
static inline bool iree_hal_amdgpu_epoch_signal_table_lookup(
    const iree_hal_amdgpu_epoch_signal_table_t* table, iree_async_axis_t axis,
    hsa_signal_t* out_signal) {
  // Verify this axis is from our session and machine.
  if (iree_async_axis_session(axis) != table->session_epoch ||
      iree_async_axis_machine(axis) != table->machine_index) {
    return false;
  }
  // Must be a QUEUE-domain axis (not collective, host, etc.).
  if (iree_async_axis_domain(axis) != IREE_ASYNC_CAUSAL_DOMAIN_QUEUE) {
    return false;
  }
  uint8_t device_index = iree_async_axis_device_index(axis);
  uint8_t queue_index = iree_async_axis_queue_index(axis);
  if (device_index >= table->device_count ||
      queue_index >= table->queue_stride) {
    return false;
  }
  hsa_signal_t signal =
      table->signals[(iree_host_size_t)device_index * table->queue_stride +
                     queue_index];
  if (signal.handle == 0) return false;  // Slot not registered.
  *out_signal = signal;
  return true;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_EPOCH_SIGNAL_TABLE_H_
