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

namespace IREE::VectorExt {
class VectorLayoutInterface;
} // namespace IREE::VectorExt

/// Propagates vector layouts through `root` and its nested operations.
/// Populates `layouts` with the resolved layout for every vector-typed value.
///
/// Layout anchors are `iree_vector_ext.to_layout` ops in the IR. These must
/// be present before calling this function — they seed the analysis. On GPUs,
/// anchors are typically placed on loads. IR without anchors is considered
/// ill-formed for vector distribution.
///
/// Starting from anchors, layouts are inferred for the rest of the IR using
/// op-specific rules (elementwise, transpose, contract, scf.for, etc.).
///
/// Example — single anchor:
///
///    %read = vector.transfer_read ...
///    %anchored = iree_vector_ext.to_layout %read to layout(L)
///    %c = arith.mulf %anchored, %b   --> inferred to layout L
///    %e = arith.divf %b, %anchored   --> inferred to layout L
///
/// When a value receives conflicting layouts from different anchors, the
/// analysis resolves the conflict by inserting a `to_layout` conversion.
/// Cheap ops (constants, create_mask, step) are cloned per use site instead.
/// The caller is responsible for lowering the inserted `to_layout` ops.
///
/// Example — conflict resolution:
///
///    %a = ... to_layout ... to layout(L)
///    %b = ... to_layout ... to layout(L')
///    // %a and %b have different layouts but feed into the same op.
///    // The analysis inserts a conversion on %b:
///    %resolved = iree_vector_ext.to_layout %b to layout(L)
///    arith.addf %a, %resolved   --> layout L
void propagateVectorLayoutInfo(
    Operation *root,
    llvm::MapVector<Value, IREE::VectorExt::VectorLayoutInterface> &layouts);

/// Transform a `scf.for` loop with a strictly positive step
///   for %i = %lb to %ub step %s
/// into a 0-based loop with step 1
///   for %ii = 0 to ceildiv(%ub - %lb, %s) step 1
/// Insert an `affine.apply` operation to compute the denormalized index value.
LogicalResult normalizeLoopBounds(RewriterBase &rewriter, scf::ForOp forOp);

/// Transform a `scf.forall` loop with a strictly positive steps
///   forall (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1)
/// into a 0-based loop with step 1 (normalized)
///   forall (%i, %j) in (ceildiv(%ub0 - %lb0, %s0), ceildiv(%ub1 - %lb1, %s1))
/// Insert `affine.apply` operations to compute the denormalized index values.
LogicalResult normalizeLoopBounds(RewriterBase &rewriter,
                                  scf::ForallOp forallOp);

/// Move memref.expand_shape and memref.collapse_shape ops nested within the
/// `op` up to just after their last operand producer.
void moveUpMemrefReshapeOps(RewriterBase &rewriter, Operation *op);

/// Populate patterns that fold tensor.expand/collapse_shape into the memref
/// of iree_codegen.load_from_buffer or iree_codegen.store_to_buffer ops.
void populateFoldTensorReshapeIntoBufferPatterns(RewritePatternSet &patterns);

/// Populate patterns that fold reshaping and bitcasting ops into the source
/// hal.interface.binding.subspan.
void populateReshapeToInterfaceTensorPatterns(RewritePatternSet &patterns);

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

/// Populate pattern to convert `tensor.extract_slice(tensor.expand_shape)` to
/// `tensor.expand_shape(tensor.extract_slice)`. The optional `controlFn` can be
/// used to restrict when the pattern applies.
void populateSwapExtractWithExpandPattern(
    RewritePatternSet &patterns, linalg::ControlFusionFn controlFn = nullptr);

/// Populate pattern to fold `tensor.extract_slice(linalg.broadcast)` into the
/// broadcast input when the extract_slice undoes the broadcast.
void populateFoldExtractSliceOfBroadcastPattern(RewritePatternSet &patterns);

/// Populate pattern to convert `tensor.extract_slice(tensor.collapse_shape)` to
/// `tensor.collapse_shape(tensor.extract_slice)`.
void populateSwapExtractWithCollapsePattern(RewritePatternSet &patterns);

/// Populate patterns to fold relayout operations into map_store ops. If a
/// `padDistributionConfigFn` is passed, then the tensor.pad folding pattern
/// will be added, using the padDistributionConfigFn for distribution.
void populateCombineRelayoutOpPatterns(
    RewritePatternSet &patterns,
    PadDistributionConfigFn padDistributionConfigFn = nullptr);

/// Populate patterns to fuse tilable consumers of forall ops into it.
void populateFuseTilableForallConsumersPattern(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Utilities for iteration space expansion transformations
//===----------------------------------------------------------------------===//

/// Helper struct to hold the expand/collapse shape ops created for dimension
/// expansion or blocking transformations.
struct ReshapeOps {
  tensor::ExpandShapeOp expandShapeOp;
  tensor::CollapseShapeOp collapseShapeOp;
};

/// Populate patterns to remove optimization barriers.
void populateRemoveOptimizationBarrierPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMS_H_
