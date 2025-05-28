// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMS_H_

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir::bufferization {
struct OneShotBufferizationOptions;
} // namespace mlir::bufferization

namespace mlir::iree_compiler {

/// Common helper class for tracking lowering configs through pattern
/// applications.
class ConfigTrackingListener : public RewriterBase::Listener {
public:
  ConfigTrackingListener() = default;
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;
};

/// Fold a tensor::ExpandShapeOp into a consumer `mapScatterOp`, by linearizing
/// and then delinearizing the source indices of the `mapScatterOp`s index
/// transformation.
IREE::LinalgExt::MapScatterOp
foldExpandShapeIntoMapScatter(RewriterBase &rewriter,
                              tensor::ExpandShapeOp expandShapeOp,
                              IREE::LinalgExt::MapScatterOp mapScatterOp);

/// Fold a tensor::CollapseShapeOp into a consumer `mapScatterOp`, by
/// linearizing and then delinearizing the source indices of the
/// `mapScatterOp`s index transformation.
IREE::LinalgExt::MapScatterOp
foldCollapseShapeIntoMapScatter(RewriterBase &rewriter,
                                tensor::CollapseShapeOp collapseShapeOp,
                                IREE::LinalgExt::MapScatterOp mapScatterOp);

using IGEMMConfigFn =
    std::function<LogicalResult(linalg::GenericOp, IREE::LinalgExt::Im2colOp)>;
using IGEMMControlFn = std::function<bool(Operation *)>;

/// Converts conv_2d ops into linalg_ext.im2col + matmul, and sets a lowering
/// configuration on the matmul.
LogicalResult convertToIGEMMAndSetConfig(
    FunctionOpInterface funcOp,
    std::optional<IGEMMConfigFn> configFn = std::nullopt,
    std::optional<IGEMMControlFn> controlFn = std::nullopt);

/// Eliminates tensor.empty ops to avoid buffer allocations.
LogicalResult eliminateEmptyTensors(
    RewriterBase &rewriter, Operation *op,
    const bufferization::OneShotBufferizationOptions &options);

/// Bufferizes the given op with One-Shot Bufferize.
LogicalResult
runIREEOneShotBufferize(Operation *op,
                        const IREEOneShotBufferizationOptions &options,
                        bufferization::BufferizationState &state);

/// For a given operation within a dispatch, tile and distribute the operation
/// to workgroups as well as tile + fuse its producers. Returns the
/// generated tiled and fused ops, as well as the loops used for distribution.
struct IREETileAndFuseResult {
  SmallVector<Operation *> tiledAndFusedOps;
  SmallVector<scf::ForOp> loops;
  SmallVector<Value> workgroupCount;
};

FailureOr<IREETileAndFuseResult>
tileAndFuseDispatchUsingSCFForOp(RewriterBase &rewriter, TilingInterface op,
                                 linalg::LinalgTilingOptions tilingOptions);

/// Result of the tiled operation.
struct IREETilingResult {
  SmallVector<Operation *> tiledOps;
  SmallVector<Value> tiledValues;
  SmallVector<scf::ForOp> loops;
  SmallVector<Value> workgroupCount;
  // TODO(ravishankarm): Cleanup the following returns. We should not need
  // these.
  llvm::SmallBitVector tiledLoops;
  SmallVector<OpFoldResult> tileOffsets;
  SmallVector<OpFoldResult> tileSizes;
};
FailureOr<IREETilingResult>
tileDispatchUsingSCFForOp(RewriterBase &rewriter, TilingInterface op,
                          linalg::LinalgTilingOptions options);

/// Populate patterns related to clean up the IR after tile and distribute
/// to workgroups.
void populateTileAndDistributeToWorkgroupsCleanupPatterns(
    RewritePatternSet &patterns);

/// Populate IREE patterns related to resolving
/// `memref.extract_strided_metadata`.
void populateIREEResolveExtractStridedMetadataPatterns(
    RewritePatternSet &patterns, bool allowSubviewExpansion = false);

/// Populate patterns that replaces maximumf/minimumf with minumf/maxnumf ops.
/// This is supposed to be used for targets which have faulty codegen
/// for maximumf/minimumf ops, e.g. LLVM NVIDIA-PTX.
void populateReplaceSlowMinMaxOpsPatterns(RewritePatternSet &patterns);

void populateSwapExtractWithExpandPattern(RewritePatternSet &patterns);

/// Populate patterns to fold relayout operations into map_scatter ops.
void populateCombineRelayoutOpPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMS_H_
