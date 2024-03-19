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

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLinalgExtToLoopsPass();

std::unique_ptr<OperationPass<>> createPadContractionToBlockSizePass();

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

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTopkSplitReductionPass();

/// Tile and decompose the winograd transform ops into a sequence
/// of linalg ops.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileAndDecomposeWinogradTransformPass();

// Creates a pass to convert linalg convolution ops into a sequence of
// linalg_ext.winograd.* ops and linalg.batch_matmul ops using the winograd
// tranformation.
std::unique_ptr<Pass> createConvertConv2DToWinogradPass();

// Transform dialect version of tile and decompose attention wrapper.
// The optional tile size specifies the step for the innermost for loop.
void tileAndDecomposeAttention(IREE::LinalgExt::AttentionOp attnOp,
                               SmallVectorImpl<Operation *> &ops,
                               RewriterBase &rewriter, bool onlyTile = false,
                               std::optional<uint64_t> tileSize = std::nullopt);

IREE::LinalgExt::AttentionOp
tileAttention(IREE::LinalgExt::AttentionOp attnOp,
              SmallVectorImpl<Operation *> &ops, RewriterBase &rewriter,
              std::optional<uint64_t> tileSize = std::nullopt);

void decomposeTiledAttention(IREE::LinalgExt::AttentionOp tiledAttnOp,
                             SmallVectorImpl<Operation *> &ops,
                             RewriterBase &rewriter,
                             std::optional<uint64_t> tileSize = std::nullopt);

// Creates a pass to convert the attention op into a sequence of
// linalg ops.
std::unique_ptr<Pass> createTileAndDecomposeAttentionPass();

//===---------------------------------------------------------------------===//
// Codegen Strategy passes that are moved into IREE.
//===---------------------------------------------------------------------===//

void registerPasses();

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
