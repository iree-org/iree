// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NOTE: these structs/enums are taken from the HSA spec, the hsa.h and
// hsa_ext_amd.h headers, and the LLVM AMDGPU device library headers.
// We define them locally as the HSA headers cannot be directly used in
// bare-metal C and the device libraries are only available in a fork of LLM.
//
// Sources:
// https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
// https://github.com/ROCm/ROCR-Runtime
// https://github.com/ROCm/rocMLIR/blob/develop/external/llvm-project/amd/device-libs/README.md

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_SIGNAL_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_SIGNAL_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"

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
#define iree_hsa_signal_null() \
  (iree_hsa_signal_t) { 0 }

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
// AMD signal kind.
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
  struct iree_amd_queue_s* queue_ptr;
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
// Wait condition operation.

//===----------------------------------------------------------------------===//
// HSA Signal Utilities
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Device Library Functions
//===----------------------------------------------------------------------===//
// These are cloned from llvm-project/amd/device-libs/ockl/src/hsaqs.cl so that
// we don't need to rely on ockl.bc (and don't have to try to find them every
// time we want to see what they do).

#if defined(IREE_AMDGPU_TARGET_DEVICE)

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void
iree_hsa_signal_update_mailbox(
    const iree_amd_signal_t* IREE_AMDGPU_RESTRICT signal) {
  iree_amdgpu_scoped_atomic_uint64_t* mailbox =
      (iree_amdgpu_scoped_atomic_uint64_t*)signal->event_mailbox_ptr;
  if (mailbox) {
    const uint32_t event_id = signal->event_id;
    iree_amdgpu_scoped_atomic_store(mailbox, event_id,
                                    iree_amdgpu_memory_order_release,
                                    iree_amdgpu_memory_scope_system);
    __builtin_amdgcn_s_sendmsg(1 | (0 << 4),
                               __builtin_amdgcn_readfirstlane(event_id) & 0xFF);
  }
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE int64_t iree_hsa_signal_load(
    const iree_hsa_signal_t signal, iree_amdgpu_memory_order_t memory_order) {
  const iree_amd_signal_t* s = (const iree_amd_signal_t*)signal.handle;
  return iree_amdgpu_scoped_atomic_load(
      (iree_amdgpu_scoped_atomic_int64_t*)&s->value, memory_order,
      iree_amdgpu_memory_scope_system);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_hsa_signal_add(
    iree_hsa_signal_t signal, int64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amd_signal_t* s = (iree_amd_signal_t*)signal.handle;
  iree_amdgpu_scoped_atomic_fetch_add(
      (iree_amdgpu_scoped_atomic_int64_t*)&s->value, value, memory_order,
      iree_amdgpu_memory_scope_system);
  iree_hsa_signal_update_mailbox(s);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_hsa_signal_and(
    iree_hsa_signal_t signal, int64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amd_signal_t* s = (iree_amd_signal_t*)signal.handle;
  iree_amdgpu_scoped_atomic_fetch_and(
      (iree_amdgpu_scoped_atomic_int64_t*)&s->value, value, memory_order,
      iree_amdgpu_memory_scope_system);
  iree_hsa_signal_update_mailbox(s);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_hsa_signal_or(
    iree_hsa_signal_t signal, int64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amd_signal_t* s = (iree_amd_signal_t*)signal.handle;
  iree_amdgpu_scoped_atomic_fetch_or(
      (iree_amdgpu_scoped_atomic_int64_t*)&s->value, value, memory_order,
      iree_amdgpu_memory_scope_system);
  iree_hsa_signal_update_mailbox(s);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_hsa_signal_xor(
    iree_hsa_signal_t signal, int64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amd_signal_t* s = (iree_amd_signal_t*)signal.handle;
  iree_amdgpu_scoped_atomic_fetch_xor(
      (iree_amdgpu_scoped_atomic_int64_t*)&s->value, value, memory_order,
      iree_amdgpu_memory_scope_system);
  iree_hsa_signal_update_mailbox(s);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE int64_t
iree_hsa_signal_exchange(iree_hsa_signal_t signal, int64_t value,
                         iree_amdgpu_memory_order_t memory_order) {
  iree_amd_signal_t* s = (iree_amd_signal_t*)signal.handle;
  int64_t existing = iree_amdgpu_scoped_atomic_exchange(
      (iree_amdgpu_scoped_atomic_int64_t*)&s->value, value, memory_order,
      iree_amdgpu_memory_scope_system);
  iree_hsa_signal_update_mailbox(s);
  return existing;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_hsa_signal_subtract(
    iree_hsa_signal_t signal, int64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amd_signal_t* s = (iree_amd_signal_t*)signal.handle;
  iree_amdgpu_scoped_atomic_fetch_sub(
      (iree_amdgpu_scoped_atomic_int64_t*)&s->value, value, memory_order,
      iree_amdgpu_memory_scope_system);
  iree_hsa_signal_update_mailbox(s);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE int64_t
iree_hsa_signal_cas(iree_hsa_signal_t signal, int64_t expected, int64_t value,
                    iree_amdgpu_memory_order_t memory_order) {
  iree_amd_signal_t* s = (iree_amd_signal_t*)signal.handle;
  int64_t existing = expected;
  if (iree_amdgpu_scoped_atomic_compare_exchange_strong(
          (iree_amdgpu_scoped_atomic_int64_t*)&s->value, &existing, value,
          memory_order, iree_amdgpu_memory_order_relaxed,
          iree_amdgpu_memory_scope_system)) {
    iree_hsa_signal_update_mailbox(s);
  }
  return existing;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void iree_hsa_signal_store(
    iree_hsa_signal_t signal, int64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amd_signal_t* s = (iree_amd_signal_t*)signal.handle;
  if (s->kind == IREE_AMD_SIGNAL_KIND_USER) {
    // User signal may need a mailbox poke.
    iree_amdgpu_scoped_atomic_store(
        (iree_amdgpu_scoped_atomic_int64_t*)&s->value, value, memory_order,
        iree_amdgpu_memory_scope_system);
    iree_hsa_signal_update_mailbox(s);
  } else {
    // Hardware doorbell supports AQL semantics.
    // NOTE: this requires __oclc_ISA_version >= 9000; older hardware doesn't
    // support the atomic store knocks and needs emulation.
    iree_amdgpu_scoped_atomic_store(
        (iree_amdgpu_scoped_atomic_uint64_t*)s->hardware_doorbell_ptr,
        (uint64_t)value, iree_amdgpu_memory_order_release,
        iree_amdgpu_memory_scope_system);
  }
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_SIGNAL_H_
