// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_CUDA_PASS_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_CUDA_PASS_H_

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

/// Pass to set range metadata attached to block id intrinsics.
struct SetBlockIdsRangePass : PassInfoMixin<SetBlockIdsRangePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

}  // namespace llvm

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_CUDA_PASS_H_
