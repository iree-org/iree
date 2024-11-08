// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TILELARGETENSORSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct TileLargeTensorsPass final
    : public impl::TileLargeTensorsPassBase<TileLargeTensorsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Finds the largest factor of |val| less than or equal to the given
/// |upperBound|. All 2 element factorizations of the value must include a
/// term less than or equal to floor(sqrt(val)), so we search until we find
/// the first factor whose reciprocal is <= the upper bound.
int64_t getLargestFactorLessThan(int64_t val, int64_t upperBound) {
  assert(val >= 1);
  for (int64_t i = 1, e = std::sqrt(val); i <= e; ++i) {
    if (val % i == 0 && val / i <= upperBound) {
      return val / i;
    }
  }
  return 1;
}

/// Helper to tile and greedily fuse the given operation. This does not yield
/// any fused operation and only replaces the tiling root. Because this pass is
/// primarily concerned with managing large vector sizes, we only handle linalg
/// ops here.
/// TODO: Handle all vectorizable ops that might yield a large vector.
///
/// If tiling fails this returns silently (tiling is best effort). Later
/// verification steps will throw an error if distribution does not occur.
static void tileToMaxVectorSize(RewriterBase &rewriter,
                                linalg::LinalgOp linalgOp,
                                int64_t maxVectorSize) {
  assert(maxVectorSize >= 1 && "maximum vector size must be at least 1");
  SmallVector<int64_t> staticTileSizes = linalgOp.getStaticLoopRanges();
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  // Collect the total statically known parallel iterations of the linalg op.
  // We expect this to be the minimum required vector size for the op
  // because outputs should reflect the full parallel iteration space.
  int64_t staticNumTrips = 1;
  for (auto [size, type] : llvm::zip_equal(staticTileSizes, iteratorTypes)) {
    // Skip reduction iterators.
    if (type == utils::IteratorType::reduction) {
      continue;
    }
    if (ShapedType::isDynamic(size)) {
      // Tile all dynamic dims to 1 as well to enable new vectorization
      // opportunities. This also ensures all entries in staticTileSizes are
      // static.
      // TODO: This may want to be a pass option in case strategies like
      // masked vectorization are employed for dynamic shapes.
      size = 1;
    } else {
      staticNumTrips *= size;
    }
  }

  int64_t expectedMinVectorSize = staticNumTrips;
  int64_t lastParallelDim = 0;
  for (int64_t i = 0, e = staticTileSizes.size() - 1; i < e; ++i) {
    if (iteratorTypes[i] == utils::IteratorType::reduction) {
      continue;
    }
    lastParallelDim = i;
    // While we exceed the maximum vector size, set the tile size for all
    // loops except the inner most to 1. This assumes that the only dimension
    // that can be meaningfully vectorized is the inner most which is not always
    // true. Considering this is fallback logic, this is fine.
    if (expectedMinVectorSize > maxVectorSize) {
      // These two quantities are always divisible. Also staticTileSizes[i] is
      // always static since we set all dynamic entries to 1.
      expectedMinVectorSize /= staticTileSizes[i];
      staticTileSizes[i] = 1;
    }
  }
  if (iteratorTypes.back() == utils::IteratorType::parallel) {
    lastParallelDim = staticTileSizes.size() - 1;
  }

  // For the inner most loop, pick the largest static integer factor that is
  // less than the maximum vector size. This might not be a great approximation
  // and we may opt for a smaller default in the future.
  if (expectedMinVectorSize > maxVectorSize) {
    staticTileSizes[lastParallelDim] =
        getLargestFactorLessThan(expectedMinVectorSize, maxVectorSize);
  }

  // Check if nothing to do.
  if (staticTileSizes == linalgOp.getStaticLoopRanges()) {
    return;
  }

  rewriter.setInsertionPoint(linalgOp);
  SmallVector<OpFoldResult> tileSizes =
      getAsIndexOpFoldResult(rewriter.getContext(), staticTileSizes);

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);

  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    // Always fuse tilable ops but never yield a replacement.
    if (!isa<TilingInterface>(originalProducer.getOwner())) {
      return std::nullopt;
    }
    return scf::SCFTileAndFuseOptions::ControlFnResult{
        /*yieldProducerReplacement=*/false};
  };
  tileAndFuseOptions.setFusionControlFn(controlFn);

  FailureOr<scf::SCFTileAndFuseResult> tiledResults =
      scf::tileConsumerAndFuseProducersUsingSCF(
          rewriter, cast<TilingInterface>(&*linalgOp), tileAndFuseOptions);
  if (failed(tiledResults)) {
    return;
  }

  // Perform the replacement of the tiling root.
  for (OpResult res : linalgOp->getResults()) {
    if (auto replacement = tiledResults->replacements.lookup(res)) {
      rewriter.replaceAllUsesWith(res, replacement);
    }
  }

  if (linalgOp->use_empty()) {
    rewriter.eraseOp(linalgOp);
  }
}

/// Recursively process the given region and tile all linalg operations that
/// are too large. The assumption is that all operations have been sufficiently
/// tiled or lowered by this point and this is a fallback to avoid large vector
/// sizes.
static void processRegion(RewriterBase &rewriter, Region *region,
                          int64_t maxVectorSize) {
  // Process the region blocks in reverse.
  for (Block &block : llvm::reverse(region->getBlocks())) {
    // Save a reversed list of operations within the block. Ops will be
    // greedily tiled + fused in reverse so that if a producer can be fused
    // with a consumer we only distribute the producer once via fusion.
    SmallVector<Operation *> targetOps =
        llvm::map_to_vector(llvm::reverse(block.getOperations()),
                            [](Operation &op) { return &op; });
    // Skip all unused ops (possibly from tiling).
    for (Operation *op : targetOps) {
      if (op->use_empty()) {
        continue;
      }

      // Try to greedily tile + fuse linalg ops.
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        tileToMaxVectorSize(rewriter, linalgOp, maxVectorSize);
        continue;
      }

      // Else recursively process all nested operations.
      for (auto &region : op->getRegions()) {
        processRegion(rewriter, &region, maxVectorSize);
      }
    }
  }
}

void TileLargeTensorsPass::runOnOperation() {
  auto funcOp = getOperation();

  IRRewriter rewriter(funcOp->getContext());
  for (auto &region : funcOp->getRegions()) {
    processRegion(rewriter, &region, maxVectorSize);
  }
}

} // namespace mlir::iree_compiler
