// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

/// Function signature to control reduction splitting. This returns the split
/// reduction ratio used to split the reduction dimension. The ratio is applied
/// to the reduction dimension of TopK. If the ratio value is less or equal to 1
/// then nothing will be done. Input is the current depth of recursive split
/// reduction, starting from 0 (first level).
using TopkSplitReductionControlFn =
    std::function<int64_t(int64_t splitReductionDepth)>;

LogicalResult
splitReduction(RewriterBase &rewriter, LinalgExt::TopkOp topkOp,
               const TopkSplitReductionControlFn &splitReductionFn);

/// Patterns to convert linalg convolution ops into a gemm with an im2col
/// op and reshapes on the inputs.
/// TODO(Max191): Maybe move to transforms and use a funcOp walk instead of a
///               rewrite pattern for this.
void populateConv2DToIm2colOpPatterns(
    RewritePatternSet &patterns,
    std::optional<std::function<bool(Operation *)>> controlFn = std::nullopt);

void convertToOnlineAttention(IREE::LinalgExt::AttentionOp attnOp,
                              SmallVectorImpl<Operation *> &ops,
                              RewriterBase &rewriter);

//===---------------------------------------------------------------------===//
// Register LinalgExt Passes.
//===---------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc" // IWYU pragma: keep

void registerPasses();

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
