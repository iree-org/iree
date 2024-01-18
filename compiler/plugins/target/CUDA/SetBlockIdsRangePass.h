// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_TARGET_CUDA_SETBLOCKIDSRANGEPASS_H_
#define IREE_COMPILER_PLUGINS_TARGET_CUDA_SETBLOCKIDSRANGEPASS_H_

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

/// Pass to set range metadata attached to block id intrinsics.
struct SetBlockIdsRangePass : PassInfoMixin<SetBlockIdsRangePass> {
  SetBlockIdsRangePass(const std::array<int32_t, 3> &maxWorkgroupSize)
      : maxWorkgroupSize(maxWorkgroupSize) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  std::array<int32_t, 3> maxWorkgroupSize;
};

} // namespace llvm

#endif // IREE_COMPILER_PLUGINS_TARGET_CUDA_SETBLOCKIDSRANGEPASS_H_
