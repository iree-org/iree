// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_NCCL_CHANNEL_H_
#define IREE_HAL_DRIVERS_HIP_NCCL_CHANNEL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/rccl_dynamic_symbols.h"
#include "iree/hal/drivers/hip/tracing.h"
#include "iree/hal/utils/collective_batch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true if |id| is all zeros indicating an empty ID.
static inline bool iree_hal_hip_nccl_id_is_empty(
    const iree_hal_hip_nccl_id_t* id) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(id->data); ++i) {
    if (id->data[i] != 0) return false;
  }
  return true;
}

// Gets a unique ID to bootstrap a new communicator. It calls ncclGetUniqueId
// under the hood.
iree_status_t iree_hal_hip_nccl_get_unique_id(
    const iree_hal_hip_nccl_dynamic_symbols_t* symbols,
    iree_hal_hip_nccl_id_t* out_id);

// Creates a IREE HAL channel using the given NCCL |id|, |rank|, and |count|.
// It calls ncclCommInitRankConfig under the hood.
iree_status_t iree_hal_hip_nccl_channel_create(
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    const iree_hal_hip_nccl_id_t* id, int rank, int count,
    iree_allocator_t host_allocator, iree_hal_channel_t** out_channel);

// Performs a non-blocking submission of |batch| to |stream|.
// The backing storage of |batch| is dropped immediately but all resources
// referenced will be retained by the parent command buffer for its lifetime.
// Note that operations in the batch may apply to different channels.
iree_status_t iree_hal_hip_nccl_submit_batch(
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_hal_hip_tracing_context_t* tracing_context,
    const iree_hal_collective_batch_t* batch, hipStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_NCCL_CHANNEL_H_
