// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-side signal manipulation functions built on top of the ABI types.
// For the signal struct layout and type definitions see abi/signal.h (exported
// below).
//
// Sources:
// https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
// https://github.com/ROCm/ROCR-Runtime
// https://github.com/ROCm/rocMLIR/blob/develop/external/llvm-project/amd/device-libs/README.md

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_SIGNAL_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_SIGNAL_H_

#include "iree/hal/drivers/amdgpu/abi/signal.h"  // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/device/support/common.h"

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
