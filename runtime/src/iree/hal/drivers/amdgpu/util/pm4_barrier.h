// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// PM4 in-stream barrier emission helpers for command-buffer replay.
//
// PM4 command buffers do not get the AQL packet header's BARRIER bit or fence
// scopes for each dispatch. Replay must record equivalent ordering explicitly:
// CS_PARTIAL_FLUSH waits for prior compute work to finish, and ACQUIRE_MEM
// performs the cache-management operation selected by the HAL fence scopes.

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_BARRIER_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_BARRIER_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amdgpu_pm4_barrier_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_NONE = 0u,
  // Waits for prior compute dispatches before subsequent PM4 commands run.
  IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION = 1u << 0,
  // Makes shader-written PM4 IB dwords visible to a following command
  // processor IB fetch.
  IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB = 1u << 1,
} iree_hal_amdgpu_pm4_barrier_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_pm4_barrier_flags_t;

enum {
  // Maximum dwords emitted by one gfx10+ PM4 command-buffer barrier.
  IREE_HAL_AMDGPU_PM4_BARRIER_GFX10_MAX_DWORD_COUNT =
      IREE_HAL_AMDGPU_PM4_EVENT_WRITE_DWORD_COUNT +
      IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GFX10_DWORD_COUNT,
};

static inline bool iree_hal_amdgpu_pm4_fence_scope_is_valid(
    iree_hsa_fence_scope_t scope) {
  return scope == IREE_HSA_FENCE_SCOPE_NONE ||
         scope == IREE_HSA_FENCE_SCOPE_AGENT ||
         scope == IREE_HSA_FENCE_SCOPE_SYSTEM;
}

static inline iree_hsa_fence_scope_t iree_hal_amdgpu_pm4_max_fence_scope(
    iree_hsa_fence_scope_t lhs, iree_hsa_fence_scope_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

// Returns the conservative GCR_CNTL bits for one HAL fence-scope pair.
static inline uint32_t iree_hal_amdgpu_pm4_barrier_gcr_cntl_for_scopes_gfx10(
    iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope) {
  const iree_hsa_fence_scope_t scope =
      iree_hal_amdgpu_pm4_max_fence_scope(acquire_scope, release_scope);
  if (scope == IREE_HSA_FENCE_SCOPE_NONE) return 0;
  uint32_t gcr_cntl = IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GLI_INV_ALL |
                      IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GLK_INV |
                      IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GLV_INV |
                      IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GL1_INV;
  if (scope == IREE_HSA_FENCE_SCOPE_SYSTEM) {
    gcr_cntl |= IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GLM_WB |
                IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GLM_INV |
                IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GL2_INV |
                IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GL2_WB;
  }
  return gcr_cntl;
}

// Returns the exact dword count a gfx10+ barrier would emit, or zero when the
// arguments do not describe a valid PM4 command-buffer barrier.
static inline uint32_t iree_hal_amdgpu_pm4_barrier_dword_count_gfx10(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_hal_amdgpu_pm4_barrier_flags_t barrier_flags,
    iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope) {
  if (!iree_hal_amdgpu_pm4_fence_scope_is_valid(acquire_scope) ||
      !iree_hal_amdgpu_pm4_fence_scope_is_valid(release_scope)) {
    return 0;
  }
  const iree_hal_amdgpu_pm4_barrier_flags_t valid_barrier_flags =
      IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION |
      IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB;
  if ((barrier_flags & ~valid_barrier_flags) != 0) {
    return 0;
  }
  const bool has_execution_barrier = iree_any_bit_set(
      barrier_flags, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION |
                         IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB);
  uint32_t gcr_cntl = iree_hal_amdgpu_pm4_barrier_gcr_cntl_for_scopes_gfx10(
      acquire_scope, release_scope);
  if (iree_any_bit_set(barrier_flags,
                       IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB)) {
    gcr_cntl = IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_CNTL_CONSERVATIVE;
  }
  if (!has_execution_barrier) return 0;
  if (!iree_any_bit_set(
          capabilities,
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE)) {
    return 0;
  }
  uint32_t dword_count = IREE_HAL_AMDGPU_PM4_EVENT_WRITE_DWORD_COUNT;
  if (gcr_cntl != 0) {
    if (!iree_any_bit_set(
            capabilities,
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM)) {
      return 0;
    }
    dword_count += IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GFX10_DWORD_COUNT;
  }
  return dword_count;
}

// Emits a gfx10+ PM4 command-buffer barrier into |target_dwords|.
//
// The conservative fixup-to-IB contract is intentionally stronger than normal
// dispatch-to-dispatch visibility: a fixup dispatch writes PM4 dwords that the
// command processor, not a shader, fetches immediately afterward.
static inline bool iree_hal_amdgpu_pm4_barrier_emit_gfx10(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_hal_amdgpu_pm4_barrier_flags_t barrier_flags,
    iree_hsa_fence_scope_t acquire_scope, iree_hsa_fence_scope_t release_scope,
    uint32_t capacity, uint32_t* target_dwords, uint32_t* out_dword_count) {
  *out_dword_count = 0;
  const uint32_t dword_count = iree_hal_amdgpu_pm4_barrier_dword_count_gfx10(
      capabilities, barrier_flags, acquire_scope, release_scope);
  if (dword_count == 0 || capacity < dword_count) return false;

  uint32_t gcr_cntl = iree_hal_amdgpu_pm4_barrier_gcr_cntl_for_scopes_gfx10(
      acquire_scope, release_scope);
  if (iree_any_bit_set(barrier_flags,
                       IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB)) {
    gcr_cntl = IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_CNTL_CONSERVATIVE;
  }

  uint32_t dword_index = 0;
  target_dwords[dword_index++] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_EVENT_WRITE,
      IREE_HAL_AMDGPU_PM4_EVENT_WRITE_DWORD_COUNT);
  target_dwords[dword_index++] =
      IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_TYPE_CS_PARTIAL_FLUSH |
      IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_INDEX_CS_PARTIAL_FLUSH;

  if (gcr_cntl != 0) {
    target_dwords[dword_index++] = iree_hal_amdgpu_pm4_make_header(
        IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_ACQUIRE_MEM,
        IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GFX10_DWORD_COUNT);
    target_dwords[dword_index++] = iree_hal_amdgpu_pm4_acquire_mem_gfx10_engine(
        IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_ENGINE_DEFAULT);
    target_dwords[dword_index++] = IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_COHER_SIZE;
    target_dwords[dword_index++] =
        IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GFX10_COHER_SIZE_HI;
    target_dwords[dword_index++] = 0;
    target_dwords[dword_index++] = 0;
    target_dwords[dword_index++] =
        IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_POLL_INTERVAL;
    target_dwords[dword_index++] = gcr_cntl;
  }

  *out_dword_count = dword_count;
  return true;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_BARRIER_H_
