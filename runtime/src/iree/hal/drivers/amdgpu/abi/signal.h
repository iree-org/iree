// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// HSA/AMDGPU signal struct layout as defined by the HSA spec and the AMD
// hardware implementation. These are the types that appear in device-visible
// memory and must have identical layout on both host and device.
//
// This header contains only type definitions, enums, and pure-logic helpers.
// Device-side signal manipulation functions (atomic stores, mailbox pokes,
// etc.) live in device/support/signal.h.
//
// Sources:
// https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
// https://github.com/ROCm/ROCR-Runtime

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_SIGNAL_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_SIGNAL_H_

#include "iree/hal/drivers/amdgpu/abi/common.h"

typedef struct iree_amd_queue_t iree_amd_queue_t;

//===----------------------------------------------------------------------===//
// HSA/AMDGPU Signal
//===----------------------------------------------------------------------===//

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// "Opaque" reference to an iree_amd_signal_t*.
// A value of 0 indicates a no-op signal (waits will succeed immediately and
// completions will no-op).
typedef struct iree_hsa_signal_t {
  uint64_t handle;
} iree_hsa_signal_t;

#else

typedef hsa_signal_t iree_hsa_signal_t;

#endif  // IREE_AMDGPU_TARGET_DEVICE

// No-op signal that will immediately succeed when waited on and be ignored when
// signaling.
#define iree_hsa_signal_null() (iree_hsa_signal_t){0}

// Returns true if the given signal is null.
#define iree_hsa_signal_is_null(signal) ((signal).handle == 0)

// Value of a signal.
// The interpretation of this is dependent on the operation consuming it.
// With barrier value packets it's user-defined and can be any value.
// With barrier-and/barrier-or and dispatch packets it acts as a semaphore where
// a 0 value indicates set and a non-zero value indicates unset. For example,
// if 3 operations are required to complete before another can proceed it should
// be set to 3, included as the completion_signal for the 3 operations, and
// used as the dependent signal in a barrier. As each operation completes it
// will decrement the value and when it reaches 0 the barrier will succeed and
// allow the dependent operation to execute.
typedef int64_t iree_hsa_signal_value_t;

// AMD signal kind.
enum iree_amd_signal_kind_t {
  // Unassigned (not seen).
  IREE_AMD_SIGNAL_KIND_INVALID = 0,
  // User-defined signal that supports all signal operations.
  IREE_AMD_SIGNAL_KIND_USER = 1,
  // Agent-defined doorbell (usually the queue's doorbell_signal field).
  // Only writes are permitted from any agent other than the origin and for our
  // purposes that means no writes ever. Soft queues created by the user must
  // use IREE_AMD_SIGNAL_KIND_USER as this is reserved for hardware.
  IREE_AMD_SIGNAL_KIND_DOORBELL = -1,
};
// Storage-width alias for iree_amd_signal_kind_t.
typedef int64_t iree_amd_signal_kind64_t;

// AMDGPU signal implementation.
// This is an implementation detail from the perspective of the HSA spec but a
// stable interface to the current generations of hardware implementing HSA.
// Signals are just locations in memory and have no special behavior other than
// how they are initialized. For our purposes there are two types: USER and
// DOORBELL.
//
// Signal values depend on the producer/consumer operations. See
// `iree_hsa_signal_value_t` for more information.
//
// Doorbell signals are firmware/hardware-specific and must only be written to
// by the host and other agents (that means no waiting either, as that's a
// read). Only the hardware queues as allocated by the HSA implementation should
// set these.
//
// User signals as presented to the hardware via `iree_amd_signal_t` are like
// futices: allocating memory accessible to a set of agents and populating it
// is enough to create and use the signal and (so long as it's not used
// afterward) deleting it is just freeing the memory. Special behavior only
// comes with host interaction: using any host HSA API (`hsa_signal_store_*`,
// `hsa_signal_wait_*`, etc) is only possible with signals allocated via either
// `hsa_signal_create` or `hsa_amd_signal_create` as those functions cast to an
// internal ROCR `Signal` interface. If the signal will only ever be used by our
// device code, the hardware queues, or our own host code not using the HSA APIs
// then we don't need to use signals created by HSA. When we do need to interact
// with the APIs the signals are implemented by two types: busy-wait and
// interrupt (as implemented in ROCR by `BusyWaitSignal` and `InterruptSignal`).
// Busy-wait are like a futex and _mostly_ exist entirely in user-mode.
// Interrupt are the same but with an additional platform event handle so that
// `hsaKmtWaitOnEvent` and other kernel-level waits can be performed. For such
// signals the platform event as returned by `hsaKmtCreateEvent` is stored in
// the `event_mailbox_ptr` and the value to post is `event_id`. I suspect in
// modern implementations that could be removed as they could be implemented
// with a futex when in-process and then the full platform handles would be
// reserved for IPC.
//
// Timestamps on the signal are set by the agent processing the operation.
// `start_ts` is set when the packet enters the active phase and `end_ts` is set
// when it completes. These timestamps are in agent-specific ticks and need to
// be translated into system-scope by scaling by relative frequencies of the
// system and the particular agent by
// `hsa_amd_profiling_convert_tick_to_system_domain` that handles the scaling.
// At its core that method occasionally queries the base timestamps and
// frequencies of the agents (as they may change over time) and the
// resynchronization accounts for drift. In order to resolve timestamps fully
// on-device we do the same thing by polling `AMDKFD_IOC_GET_CLOCK_COUNTERS`
// and providing it to the device runtime. Every time the clocks are resynced
// there's the potential for a discontinuity/backwards rolling timestamps so
// we try to only do it per-submission to at least keep all of the times within
// relatively aligned even if the entire submission may have drifted from the
// system by the end. Note that because work can happen out-of-order the
// timestamps on a set of signals may be out-of-order with respect to the system
// time once resolved and anything using the timestamps needs to handle that or
// unset the CONCURRENT execution flag on the queue.
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_amd_signal_t {
  iree_amd_signal_kind64_t kind;
  union {
    volatile iree_hsa_signal_value_t value;
    volatile uint64_t* hardware_doorbell_ptr;
  };
  uint64_t event_mailbox_ptr;
  uint32_t event_id;
  uint32_t reserved1;
  iree_amdgpu_device_tick_t start_ts;
  iree_amdgpu_device_tick_t end_ts;
  iree_amd_queue_t* queue_ptr;
  uint32_t reserved3[2];
} iree_amd_signal_t;

// Wait condition operation.
typedef uint32_t iree_hsa_signal_condition32_t;
typedef enum {
  // The two operands are equal.
  IREE_HSA_SIGNAL_CONDITION_EQ = 0,
  // The two operands are not equal.
  IREE_HSA_SIGNAL_CONDITION_NE = 1,
  // The first operand is less than the second operand.
  IREE_HSA_SIGNAL_CONDITION_LT = 2,
  // The first operand is greater than or equal to the second operand.
  IREE_HSA_SIGNAL_CONDITION_GTE = 3
} iree_hsa_signal_condition_t;

// Returns true if the given |current_signal| value matches the expected
// |desired_value| as defined by |condition|.
static IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE inline bool
iree_hsa_evaluate_signal_condition(iree_hsa_signal_condition32_t condition,
                                   iree_hsa_signal_value_t current_value,
                                   iree_hsa_signal_value_t desired_value) {
  switch (condition) {
    default:
    case IREE_HSA_SIGNAL_CONDITION_EQ:
      return current_value == desired_value;
    case IREE_HSA_SIGNAL_CONDITION_NE:
      return current_value != desired_value;
    case IREE_HSA_SIGNAL_CONDITION_LT:
      return current_value < desired_value;
    case IREE_HSA_SIGNAL_CONDITION_GTE:
      return current_value >= desired_value;
  }
}

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_SIGNAL_H_
