// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/builtins/ukernel/common.h"

[[clang::always_inline]] void iree_uk_amdgpu_argmax_f32i32(
    const float *inputBuffer, int64_t input_offset, float *outputBufferVal,
    int64_t output_val_offset, int32_t *outputBufferIdx,
    int64_t output_idx_offset, int64_t reductionSize, bool writeValue) {
  const int warpSize = __builtin_amdgcn_wavefrontsize();
  int32_t laneID = __builtin_amdgcn_workitem_id_x();
  // Set identity value to handle problem non divisible by subgroupSize.
  float laneMax =
      laneID >= reductionSize ? -FLT_MAX : inputBuffer[input_offset + laneID];
  int32_t laneResult = laneID;

  // NOTE: On F32 kernels with clang, reductionSize/blockDim.x has numerical
  // inaccuracy.
  int32_t numBatches = (reductionSize + warpSize - 1) / warpSize;
  for (int i = 1; i < numBatches; ++i) {
    int32_t idx = warpSize * i + laneID;
    float newIn =
        idx >= reductionSize ? -FLT_MAX : inputBuffer[input_offset + idx];
    if (newIn == laneMax)
      continue;
    laneMax = __builtin_fmaxf(newIn, laneMax);
    laneResult = newIn == laneMax ? idx : laneResult;
  }

  // Final reduction with one subgroup
  // NOTE: __ockl_wfred_max_f32 has correctness issue on gfx1100 documented on
  // https://github.com/iree-org/iree/issues/16112.
  float wgMax = laneMax;
  for (int i = 1; i < warpSize; i *= 2) {
    wgMax = __builtin_fmaxf(__shfl_xor_f(wgMax, i), wgMax);
  }
  // Check if there are multiple max value holders.
  uint64_t laneHasMaxValmask = __ballot(wgMax == laneMax);
  // if there is only one max value holder, write and exit.
  if (__builtin_popcountll(laneHasMaxValmask) == 1) {
    if (wgMax == laneMax) {
      if (writeValue) {
        outputBufferVal[output_val_offset] = wgMax;
      }
      outputBufferIdx[output_idx_offset] = laneResult;
    }
  } else {
    // if there are multiple max value holder, find smallest index (argmax
    // semantics).
    int32_t indexVal = wgMax == laneMax ? laneResult : __INT32_MAX__;
    laneResult = __ockl_wfred_min_i32(indexVal);
    if (laneID == 0) {
      if (writeValue) {
        outputBufferVal[output_val_offset] = wgMax;
      }
      outputBufferIdx[output_idx_offset] = laneResult;
    }
  }
  // TODO(bjacob): this fence should be on the caller side. Move to TileAndFuse?
  __threadfence_block();
}
