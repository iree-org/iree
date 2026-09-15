// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TILELARGETENSORSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct TileLargeTensorsPass final
    : impl::TileLargeTensorsPassBase<TileLargeTensorsPass> {
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

/// Tries to compute a provable static upper bound for loop dim |dim| of
/// |linalgOp| through the operand dims mapped to it. This mirrors the vector
/// size inference in GenericVectorization (inferSizesFromIR) so that both
/// stages agree on which dynamic dims masked vectorization can handle.
static std::optional<int64_t>
getProvenLoopDimUpperBound(linalg::LinalgOp linalgOp, unsigned dim) {
  SmallVector<std::pair<Value, unsigned>> operandDimPairs;
  linalgOp.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
  for (auto [operand, operandDim] : operandDimPairs) {
    FailureOr<DimBoundSize> bound =
        computeDimUpperBound(operand, operandDim, /*vscaleRange=*/std::nullopt,
                             RoundUpVscaleMultiple::No);
    if (succeeded(bound) && !bound->scalable) {
      return bound->baseSize;
    }
  }
  return std::nullopt;
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
                                TilingInterface tilingInterfaceOp,
                                ArrayRef<int64_t> bounds, int64_t maxVectorSize,
                                bool allowMaskedDynamicDims) {
  assert(maxVectorSize >= 1 && "maximum vector size must be at least 1");
  SmallVector<int64_t> staticTileSizes(bounds);
  SmallVector<utils::IteratorType> iteratorTypes =
      tilingInterfaceOp.getLoopIteratorTypes();
  auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingInterfaceOp.getOperation());
  // Provable static upper bounds of dynamic parallel dims. With
  // |allowMaskedDynamicDims|, bounded dims stay untiled (tile size 0, "no
  // tiling" like reduction dims) so masked vectorization handles the partial
  // tile remainders downstream.
  SmallVector<std::optional<int64_t>> dynamicDimUBs(staticTileSizes.size());

  // Collect the total statically known parallel iterations of the linalg op.
  // We expect this to be the minimum required vector size for the op
  // because outputs should reflect the full parallel iteration space.
  int64_t staticNumTrips = 1;
  for (int64_t i = 0, e = staticTileSizes.size(); i < e; ++i) {
    int64_t &size = staticTileSizes[i];
    // Skip tiling of reduction iterators.
    if (iteratorTypes[i] == utils::IteratorType::reduction) {
      size = 0;
      continue;
    }
    if (!ShapedType::isDynamic(size)) {
      staticNumTrips *= size;
      continue;
    }
    if (allowMaskedDynamicDims && linalgOp) {
      dynamicDimUBs[i] = getProvenLoopDimUpperBound(linalgOp, i);
    }
    if (dynamicDimUBs[i]) {
      size = 0;
      staticNumTrips *= *dynamicDimUBs[i];
      continue;
    }
    // Tile dynamic dims without a proven bound to 1 to enable new
    // vectorization opportunities. This also keeps all tiled entries in
    // staticTileSizes static.
    size = 1;
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
      // Dims kept dynamic for masked vectorization contribute their proven
      // upper bound instead of a tile size.
      int64_t dimSize = staticTileSizes[i];
      if (dimSize == 0) {
        if (!dynamicDimUBs[i]) {
          continue; // Zero-trip dim; nothing useful to shrink.
        }
        dimSize = *dynamicDimUBs[i];
      }
      expectedMinVectorSize /= dimSize;
      staticTileSizes[i] = 1;
    }
  }
  if (iteratorTypes.back() == utils::IteratorType::parallel) {
    lastParallelDim = staticTileSizes.size() - 1;
  }

  // For the inner most loop, pick the largest static integer factor that is
  // less than the maximum vector size. This might not be a great approximation
  // and we may opt for a smaller default in the future. For a dynamic dim kept
  // for masked vectorization, tile to a factor of its proven bound instead;
  // the remaining partial tile is masked downstream.
  if (expectedMinVectorSize > maxVectorSize) {
    int64_t dimSize =
        dynamicDimUBs[lastParallelDim].value_or(expectedMinVectorSize);
    staticTileSizes[lastParallelDim] =
        getLargestFactorLessThan(dimSize, maxVectorSize);
  }

  // Check if nothing to do. Dims kept dynamic for masked vectorization carry
  // tile size 0 ("no tiling") and count as unchanged for this comparison.
  bool nothingToDo = true;
  for (int64_t i = 0, e = staticTileSizes.size(); i < e; ++i) {
    int64_t effective = (staticTileSizes[i] == 0 && dynamicDimUBs[i])
                            ? bounds[i]
                            : staticTileSizes[i];
    if (effective != bounds[i]) {
      nothingToDo = false;
      break;
    }
  }
  if (nothingToDo) {
    return;
  }

  rewriter.setInsertionPoint(tilingInterfaceOp);
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
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                tileAndFuseOptions);
  if (failed(tiledResults)) {
    return;
  }

  // Perform the replacement of the tiling root.
  for (OpResult res : tilingInterfaceOp->getResults()) {
    if (auto replacement = tiledResults->replacements.lookup(res)) {
      rewriter.replaceAllUsesWith(res, replacement);
    }
  }

  if (tilingInterfaceOp->use_empty()) {
    rewriter.eraseOp(tilingInterfaceOp);
  }
}

/// Recursively process the given region and tile all linalg operations that
/// are too large. The assumption is that all operations have been sufficiently
/// tiled or lowered by this point and this is a fallback to avoid large vector
/// sizes.
static void processRegion(RewriterBase &rewriter, Region *region,
                          int64_t maxVectorSize, bool allowMaskedDynamicDims) {
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

        // Skip copies, transposes, and fills. This is based on an expectation
        // that such ops are introduced carefully and don't represent
        // significant computation anyway. Equivalent generics are still tiled
        // as they typically arise organically. Fills in particular are almost
        // never found on their own and will be fused when tiling if need be.
        if (isa<linalg::TransposeOp, linalg::CopyOp, linalg::FillOp>(op)) {
          continue;
        }
        // Nothing to do for ops without parallel loops.
        if (linalgOp.getNumParallelLoops() == 0) {
          continue;
        }
        SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
        tileToMaxVectorSize(rewriter, cast<TilingInterface>(&*linalgOp), bounds,
                            maxVectorSize, allowMaskedDynamicDims);
        continue;
      }

      // map_store creates a lot of register pressure because it carries an
      // index for every element in the input. It is better to use a smaller
      // vector size for map_store to avoid register spills, because there is
      // not much benefit in using a larger vector size anyway. For now, keep
      // the heuristic simple and just use a quarter of the maxVectorSize.
      if (auto mapStoreOp = dyn_cast<IREE::LinalgExt::MapStoreOp>(op)) {
        ArrayRef<int64_t> bounds = mapStoreOp.getInputType().getShape();
        tileToMaxVectorSize(rewriter, mapStoreOp, bounds,
                            std::max<int64_t>(maxVectorSize / 4, 1),
                            allowMaskedDynamicDims);
        continue;
      }

      // Else recursively process all nested operations.
      for (auto &region : op->getRegions()) {
        processRegion(rewriter, &region, maxVectorSize, allowMaskedDynamicDims);
      }
    }
  }
}

void TileLargeTensorsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();

  IRRewriter rewriter(funcOp->getContext());
  for (auto &region : funcOp->getRegions()) {
    processRegion(rewriter, &region, maxVectorSize, allowMaskedDynamicDims);
  }
}

} // namespace mlir::iree_compiler
