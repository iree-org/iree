// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_LLVMGPUUTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_LLVMGPUUTILS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {

/// Helper to convert copy to shared memory to async copy. This creates groups
/// of consecutive copies and emit wait operation right after.
void createAsyncGroups(RewriterBase &rewriter, func::FuncOp funcOp,
                       bool useMMASync);

/// Function to do layout analysis and distribution.
void doLayoutAnalysisAndDistribution(IRRewriter &rewriter, func::FuncOp funcOp);

/// Function to reorder transposes and elementwise ops.
void reorderTranspose(IRRewriter &rewriter, func::FuncOp funcOp);

/// Function to create extract slice after reduction + broadcast + transpose
/// when the broadcast size is less than that of the reduction source.
void createExtractSliceAfterReductionBroadcastTranspose(IRRewriter &rewriter,
                                                        func::FuncOp funcOp);

/// Find broadcast and transpose ops that use the given reduction op.
SmallVector<std::tuple<vector::BroadcastOp, vector::TransposeOp>>
getReductionBroadcastPairs(vector::MultiDimReductionOp reductionOp);

/// Checks to see if the result of the reduction, broadcast and transpose op
/// has dimensions that are the same or smaller shape than the reduction source.
bool compatibleBroadcastReductionShapes(vector::MultiDimReductionOp reductionOp,
                                        vector::BroadcastOp broadcastOp,
                                        vector::TransposeOp transposeOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif
