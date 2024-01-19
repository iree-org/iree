// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPUPATTERNS_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPUPATTERNS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

/// Adds patterns for preparing vector transfer ops for converting to GPU
/// subgroup MMA load/store ops.
void populateVectorTransferToGPUMMAPreparationPatterns(
    RewritePatternSet &patterns);

/// Adds patterns to Merge transpose op into the transfer read op. Transposes
/// are not supported on MMA types but MMA load can transpose the matrix when
/// loading.
void populateCombineVectorTransferReadBroadcastPatterns(
    RewritePatternSet &patterns);

/// Adds patterns for promoting Linalg contract op's operands to use GPU shared
/// memory.
void populateContractPromotionPatterns(RewritePatternSet &patterns,
                                       ArrayRef<int64_t> operandsToPromote);

void populateDropSharedMemoryDeallocOpPatterns(RewritePatternSet &patterns);

void populateGPUDistributionPatterns(RewritePatternSet &patterns);

void populateGPUDistributionLayoutAttrPatterns(ArrayRef<Value> threadGrid,
                                               RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPUPATTERNS_H_
