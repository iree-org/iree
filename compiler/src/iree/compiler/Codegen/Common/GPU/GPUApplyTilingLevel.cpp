// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-tiling-level"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYTILINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUApplyTilingLevelPass final
    : impl::GPUApplyTilingLevelPassBase<GPUApplyTilingLevelPass> {
  using GPUApplyTilingLevelPassBase::GPUApplyTilingLevelPassBase;
  void runOnOperation() override;
};
} // namespace

/// This collects the set of operations to tile + fuse starting from the given
/// root |op| and walking up to its producers. Stops at operations given by
/// |exclude| which are expected to receive their own independent tiling for the
/// given level.
static llvm::SmallDenseSet<Operation *>
collectTiledAndFusedOps(Operation *op,
                        llvm::SmallDenseSet<TilingInterface> exclude) {
  SmallVector<Operation *> worklist;
  llvm::SmallDenseSet<Operation *> producers;
  worklist.push_back(op);
  producers.insert(op);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      auto producer = operand.get().getDefiningOp<TilingInterface>();
      if (!producer || producers.contains(producer) ||
          exclude.contains(producer))
        continue;
      worklist.push_back(producer);
      producers.insert(producer);
    }
  }
  return producers;
}

/// Apply a tile and fuse transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
static LogicalResult
applyTileAndFuseToEachRoot(RewriterBase &rewriter,
                           llvm::SmallDenseSet<TilingInterface> &payloadOps,
                           bool threadTiling) {
  MLIRContext *context = rewriter.getContext();
  unsigned tilingLevel =
      threadTiling ? static_cast<unsigned>(IREE::GPU::TilingLevel::Thread)
                   : static_cast<unsigned>(IREE::GPU::TilingLevel::Reduction);
  for (TilingInterface tilingInterfaceOp : payloadOps) {
    mlir::DominanceInfo dominanceInfo(tilingInterfaceOp);

    llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
        collectTiledAndFusedOps(tilingInterfaceOp, payloadOps);
    llvm::DenseSet<Operation *> yieldReplacementsFor;
    for (auto op : tiledAndFusedOps) {
      if (llvm::any_of(op->getUsers(), [&](Operation *user) {
            return dominanceInfo.properlyDominates(tilingInterfaceOp, user);
          })) {
        yieldReplacementsFor.insert(op);
      }
    }

    rewriter.setInsertionPoint(tilingInterfaceOp);
    SmallVector<OpFoldResult> tileSizes =
        getLoweringConfig(tilingInterfaceOp)
            .getTilingLevelSizes(rewriter, tilingLevel, tilingInterfaceOp);

    // Pad the tile sizes with zero.
    auto zero = rewriter.getIndexAttr(0);
    int64_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
    if (tileSizes.size() > numLoops) {
      return failure();
    }
    while (tileSizes.size() < numLoops) {
      tileSizes.push_back(zero);
    }

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    if (threadTiling) {
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

      // TODO: Add some helpers to construct this based on the enum type rather
      // than doing it here.
      SmallVector<DeviceMappingAttrInterface> mapping;
      for (auto [idx, size] : llvm::enumerate(tileSizes)) {
        if (!isConstantIntValue(size, 0)) {
          unsigned mappingId =
              static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx;
          mapping.push_back(gpu::GPUThreadMappingAttr::get(
              context, static_cast<gpu::MappingId>(mappingId)));
        }
      }
      tilingOptions.setMapping(mapping);
    }

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);

    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand) {
          Operation *owner = originalProducer.getOwner();
          bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
          bool shouldFuse = false;
          if (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
            shouldFuse = !payloadOps.contains(tilingOwner);
          }
          // Do not fuse destination operands.
          shouldFuse &= !isDestinationOperand;
          return std::make_tuple(shouldFuse, yieldProducerReplacement);
        };
    tileAndFuseOptions.setFusionControlFn(controlFn);

    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                  tileAndFuseOptions);
    if (failed(tiledResults)) {
      return failure();
    }

    // Perform the replacement of tiled and fused values.
    SmallVector<Operation *> opsToReplace{tilingInterfaceOp};
    llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    for (Operation *toReplace : opsToReplace) {
      for (OpResult res : toReplace->getResults())
        if (auto replacement = tiledResults->replacements.lookup(res)) {
          Operation *replacementOp = replacement.getDefiningOp();
          rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return dominanceInfo.properlyDominates(replacementOp, user);
          });
        }

      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }
  }
  return success();
}

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(Operation *funcOp, IREE::GPU::TilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;
  unsigned opaqueLevel = static_cast<unsigned>(tilingLevel);
  funcOp->walk([&](TilingInterface target) {
    // TODO: This would probably be easier with a lowering config interface
    // method that checks whether a particular level is tiled.
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (!loweringConfig.getStaticTilingLevelSizes(opaqueLevel, target)
               .empty()) {
        targets.insert(target);
      }
    }
  });
  return targets;
}

void GPUApplyTilingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  if (tilingLevel != IREE::GPU::TilingLevel::Reduction &&
      tilingLevel != IREE::GPU::TilingLevel::Thread) {
    funcOp.emitError() << "unsupported tiling level: "
                       << IREE::GPU::stringifyEnum(tilingLevel) << "\n";
    return signalPassFailure();
  }

  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);
  bool useThread = tilingLevel == IREE::GPU::TilingLevel::Thread;

  IRRewriter rewriter(funcOp);
  if (failed(applyTileAndFuseToEachRoot(rewriter, targetOps, useThread))) {
    funcOp.emitError() << "tiling of level "
                       << IREE::GPU::stringifyEnum(tilingLevel) << " failed\n";
    return signalPassFailure();
  }

  MLIRContext *context = &getContext();

  // Apply cleanup patterns.
  {
    RewritePatternSet patterns(context);
    // Merge consecutive insert/extract slice ops to simplify later loop
    // hoisting patterns.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "tiling cleanup failed\n";
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
