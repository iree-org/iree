// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

std::unique_ptr<OperationPass<func::FuncOp>> createTilingInterfaceTilingPass();

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgExtToLoopsPass();

std::unique_ptr<OperationPass<>> createPadContractionToBlockSizePass();

/// Function signature to control reduction splitting. This returns the split
/// reduction ratio used to split the reduction dimension. The ratio is applied
/// to the reduction dimension of TopK. If the ratio value is less or equal to 1
/// then nothing will be done. Input is the current depth of recursive split
/// reduction, starting from 0 (first level).
using TopkSplitReductionControlFn =
    std::function<int64_t(int64_t splitReductionDepth)>;

/// Patterns to apply `topk split reduction` pass.
void populateTopkSplitReductionPattern(
    RewritePatternSet &patterns,
    const TopkSplitReductionControlFn &splitReductionFn,
    const linalg::LinalgTransformationFilter &f =
        linalg::LinalgTransformationFilter());

std::unique_ptr<OperationPass<func::FuncOp>> createTopkSplitReductionPass();

// Marker used as attribute the depth of the split reduction transformations.
const StringLiteral kSplitReductionDepthMarker = "__split_reduction_depth__";

//===---------------------------------------------------------------------===//
// Codegen Strategy passes that are moved into IREE.
//===---------------------------------------------------------------------===//
/// Create a LinalgStrategyTileAndFusePass.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgStrategyTileAndFusePass(
    StringRef opName = "", const linalg::LinalgTilingAndFusionOptions &opt = {},
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyTilePass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyTilePass(
    StringRef opName = "",
    const linalg::LinalgTilingOptions &opt = linalg::LinalgTilingOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyPadPass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyPadPass(
    StringRef opName = "",
    const linalg::LinalgPaddingOptions &opt = linalg::LinalgPaddingOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyDecomposePass.
// TODO: if/when we need finer control add an `opName` parameter.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyDecomposePass(
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyPeelPass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyPeelPass(
    StringRef opName = "",
    const linalg::LinalgPeelOptions &opt = linalg::LinalgPeelOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyVectorizePass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyVectorizePass(
    StringRef opName = "",
    linalg::LinalgVectorizationOptions opt =
        linalg::LinalgVectorizationOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter(),
    bool padVectorize = false);

/// Create a LinalgStrategyEnablePass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyEnablePass(
    linalg::LinalgEnablingOptions opt = linalg::LinalgEnablingOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyLowerVectorsPass.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgStrategyLowerVectorsPass(
    linalg::LinalgVectorLoweringOptions opt =
        linalg::LinalgVectorLoweringOptions(),
    const linalg::LinalgTransformationFilter &filter =
        linalg::LinalgTransformationFilter());

/// Create a LinalgStrategyRemoveMarkersPass.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgStrategyRemoveMarkersPass();

void registerPasses();

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
