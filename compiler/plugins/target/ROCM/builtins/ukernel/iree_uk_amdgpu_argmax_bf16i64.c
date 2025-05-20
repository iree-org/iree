// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/builtins/ukernel/common.h"

[[clang::always_inline]] void iree_uk_amdgpu_argmax_bf16i64(
    const __bf16 *inputBuffer, int64_t input_offset, __bf16 *outputBufferVal,
    int64_t output_val_offset, int64_t *outputBufferIdx,
    int64_t output_idx_offset, int64_t reductionSize, bool writeValue) {
  // NOTE:
  // We convert bf16 inputs to f32 before computation because HIP/OCKL and
  // Clang/LLVM do not currently support native arithmetic or comparisons on
  // bf16. In practice, these operations are internally performed by first
  // converting bf16 to float.
  const int warpSize = __builtin_amdgcn_wavefrontsize();
  int32_t laneID = __builtin_amdgcn_workitem_id_x();
  // Set identity value to handle problem non divisible by subgroupSize.
  float laneMax = laneID >= reductionSize
                      ? -FLT_MAX
                      : (float)(inputBuffer[input_offset + laneID]);
  int64_t laneResult = laneID;

  // NOTE: On F32 kernels with clang, reductionSize/blockDim.x has numerical
  // inaccuracy.
  int32_t numBatches = (reductionSize + warpSize - 1) / warpSize;
  for (int i = 1; i < numBatches; ++i) {
    int32_t idx = warpSize * i + laneID;
    float newIn = idx >= reductionSize
                      ? -FLT_MAX
                      : (float)(inputBuffer[input_offset + idx]);
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
        outputBufferVal[output_val_offset] = (__bf16)wgMax;
      }
      outputBufferIdx[output_idx_offset] = laneResult;
    }
  } else {
    // if there are multiple max value holder, find smallest index (argmax
    // semantics).
    int64_t indexVal = wgMax == laneMax ? laneResult : INT64_MAX;
    laneResult = __ockl_wfred_min_i64(indexVal);
    if (laneID == 0) {
      if (writeValue) {
        outputBufferVal[output_val_offset] = (__bf16)wgMax;
      }
      outputBufferIdx[output_idx_offset] = laneResult;
    }
  }
  // TODO(bjacob): this fence should be on the caller side. Move to TileAndFuse?
  __threadfence_block();
}
