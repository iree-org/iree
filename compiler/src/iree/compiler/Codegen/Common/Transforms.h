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
                        const IREEOneShotBufferizationOptions &options);

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

void fuseProducersOfSlices(RewriterBase &rewriter,
                           std::queue<Operation *> &worklist,
                           scf::SCFTileAndFuseOptions &options,
                           MutableArrayRef<LoopLikeOpInterface> loops);

/// Consider the following case
///
/// ```mlir
/// %0:2 = linalg.generic {
///     indexing_maps = [....,
///                      affine_map<(d0, d1, d2) -> (d0, d1),
///                      affine_map<(d0, d1, d2) -> (d0, d1)>]}
/// %1 = linalg.generic ins(%0#0, %0#1) {
///     indexing_maps = [affine_map<(d0, d1) -> (d0, d1),
///                      affine_map<(d0, d1) -> (d0, d1)]}
/// ```
///
/// After tiling the first op we get
///
/// ```
/// %0:2 = scf.forall ... {
///   %1:2 = linalg.generic {
///       indexing_maps = [....,
///                        affine_map<(d0, d1, d2) -> (d0, d1),
///                        affine_map<(d0, d1, d2) -> (d0, d1)>]}
///   }
/// }
/// %2 = linalg.generic ins(%0#0, %0#1) {
///     indexing_maps = [affine_map<(d0, d1) -> (d0, d1),
///                      affine_map<(d0, d1) -> (d0, d1)]}
/// ```
///
/// Due to a quirk of the fusion of consumers, fusing this consumer into the
/// loop results in
///
/// ```
/// %0:2 = scf.forall ... {
///   %1:2 = linalg.generic {
///       indexing_maps = [....,
///                        affine_map<(d0, d1, d2) -> (d0, d1),
///                        affine_map<(d0, d1, d2) -> (d0, d1)>]}
///   %2 = tensor.extract_slice %0#1 [...]
///   %3 = linalg.generic ins(%1#0, %2) {
///       indexing_maps = [affine_map<(d0, d1) -> (d0, d1),
///                        affine_map<(d0, d1) -> (d0, d1)]}
///   }
/// }
/// ```
///
/// This is an SSA violation because of `%0#1` being used in the loop. This
/// needs to be fixed upstream, but for cases where
/// 1. The root operation produces results using an identity indexing map (when
/// ignoring the iteration space dimensions corresponding to the reduction
/// loops)
/// 2. For all consumers of the results of the root operation, access the data
/// using identity indexing map then for each consumer fusion step it is valid
/// to replace all uses of slices of the outer loop that occur within the loop
/// with the correponding tiled result value.
/// This is a workaround till upstream transformation can fix this issue. The
/// following method is testing if such a case exists to implement the
/// work-around.
bool warForConsumerFusionSSAViolation(
    Operation *rootOp,
    const llvm::SmallDenseSet<Operation *> &tiledAndFusedOps);

/// Starting from `op` walk all operands backwards to find all
/// potentially fusible operations, i.e. operations that implement
/// the `TilingInterface`.
void collectTiledAndFusedOps(Operation *rootOp,
                             llvm::SmallDenseSet<Operation *> &result);

// Fuse all consumers of the given `tiledOp` into the surrounding scf.forall.
// Returns a list of new `tensor.extract_slice` ops with new fusion
// opportunities, as well as the new surrounding `scf.forall` (because consumer
// fusion replaces the loop).
FailureOr<std::queue<Operation *>>
fuseConsumers(RewriterBase &rewriter, Operation *tiledOp,
              MutableArrayRef<LoopLikeOpInterface> loops,
              bool useWARForConsumerFusionSSAViolation);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMS_H_
