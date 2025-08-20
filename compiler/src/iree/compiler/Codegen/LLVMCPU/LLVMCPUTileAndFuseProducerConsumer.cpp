// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-tile-root-and-fuse-producers-consumers"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUTILEANDFUSEPRODUCERCONSUMERPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

/// Returns the operation that has workgroup tiling level and `level` tiling
/// level in lowering config.
/// Returns nullptr if there is not exactly one op that meets the conditions.
static Operation *getRootOp(ArrayRef<Operation *> computeOps,
                            IREE::CPU::TilingLevel level) {
  Operation *rootOp = nullptr;
  for (Operation *op : computeOps) {
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
        getLoweringConfig(op);
    if (loweringConfig && loweringConfig.hasWorkgroupTilingLevel() &&
        loweringConfig.hasTilingLevel(level)) {
      if (rootOp) {
        return nullptr;
      }
      rootOp = op;
    }
  }
  return rootOp;
}

/// Returns the last operation that has `level` tiling level in lowering config.
/// Returns nullptr if the operation does not exist.
static Operation *getLastAnchorOp(ArrayRef<Operation *> computeOps,
                                  IREE::CPU::TilingLevel level) {
  for (Operation *op : llvm::reverse(computeOps)) {
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
        getLoweringConfig(op);
    if (loweringConfig && loweringConfig.hasTilingLevel(level)) {
      return op;
    }
  }
  return nullptr;
}

/// Implementation of tile root and fuse producers and consumers greedily. Tile
/// the root operation and fuse the producers of the root operation then
/// consumers (finds any missing fusion opportunities, then apply producer
/// fusion). If `onlyFuseProducerInputOperands` is set, only fuse producer input
/// operands.
static FailureOr<Operation *>
tileRootAndFuseProducerConsumer(IRRewriter &rewriter, TilingInterface rootOp,
                                int64_t tilingLevel,
                                bool onlyFuseProducerInputOperands) {
  auto *context = rewriter.getContext();
  mlir::DominanceInfo dominanceInfo(rootOp);
  llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
  collectTiledAndFusedOps(rootOp, tiledAndFusedOps);

  llvm::DenseSet<Operation *> yieldReplacementsFor;
  for (auto op : tiledAndFusedOps) {
    // If an op result is used after `rootOp`, yield a replacement---unless the
    // op using the result will also later be fused.
    // For example:
    //     A
    //    / \
    //   |  [B]
    //    \  /
    //     C
    // Assuming we're doing producer-consumer fusion from B, as C uses A, and B
    // does not properly dominate C, we will yield replacements for A. That is,
    // unless C will later be fused through consumer fusion.
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          return dominanceInfo.properlyDominates(rootOp, user) &&
                 !tiledAndFusedOps.contains(user);
        })) {
      yieldReplacementsFor.insert(op);
    }
  }

  int64_t numLoops = rootOp.getLoopIteratorTypes().size();
  auto tileSizesAttr = dyn_cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
      getLoweringConfig(rootOp).getTilingLevelAttr(tilingLevel));
  SmallVector<int64_t> tileSizes(tileSizesAttr.getSizes());
  SmallVector<bool> tileScalableFlags(tileSizesAttr.getScalableFlags());
  tileSizes.resize(numLoops, 0);
  tileScalableFlags.resize(numLoops, false);

  scf::SCFTilingOptions tilingOptions;
  setSCFTileSizes(tilingOptions, rootOp, std::move(tileSizes),
                  std::move(tileScalableFlags));

  // onlyFuseProducerInputOperands implies reduction tiling.
  if (!onlyFuseProducerInputOperands) {
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  }

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);

  RewritePatternSet cleanupPatterns(context);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(cleanupPatterns, context);
  tensor::DimOp::getCanonicalizationPatterns(cleanupPatterns, context);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(cleanupPatterns);
  tensor::populateBubbleUpExtractSliceOpPatterns(cleanupPatterns);
  // When fusing pads we do not want to generate zeroSliceGuards when doing
  // workgroup tiling. In `GPUApplyTilingLevelPass` we do have an option called
  // `allowZeroSlices` that can control this but we do not want these
  // generated if workgroup tiling is happening first.
  cleanupPatterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
      context, [](tensor::ExtractSliceOp) { return /*zeroSliceGuard=*/false; });
  tileAndFuseOptions.cleanupPatterns =
      FrozenRewritePatternSet(std::move(cleanupPatterns));

  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    Operation *owner = originalProducer.getOwner();
    bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
    // Do not fuse destination operands if onlyFuseProducerInputOperands is
    // true.
    bool shouldFuse = !(onlyFuseProducerInputOperands && isDestinationOperand);
    if (shouldFuse) {
      return scf::SCFTileAndFuseOptions::ControlFnResult{
          yieldProducerReplacement};
    }
    return std::nullopt;
  };
  tileAndFuseOptions.setFusionControlFn(controlFn);
  rewriter.setInsertionPoint(rootOp);

  FailureOr<scf::SCFTileAndFuseResult> tiledResults =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, rootOp,
                                                tileAndFuseOptions);
  if (failed(tiledResults)) {
    return failure();
  }

  // Perform the replacement of tiled and fused values.
  for (auto [origValue, replacement] : tiledResults->replacements) {
    rewriter.replaceAllUsesWith(origValue, replacement);
  }

  FailureOr<Operation *> rootTiledOp = tiledResults->tiledAndFusedOps.front();

  if (failed(rootTiledOp)) {
    return failure();
  }
  SmallVector<LoopLikeOpInterface> tilingLoops = tiledResults->loops;

  if (!onlyFuseProducerInputOperands) {
    FailureOr<std::queue<Operation *>> newFusionOpportunities =
        fuseConsumersIntoForall(rewriter, *rootTiledOp, tilingLoops,
                                [&tiledAndFusedOps](Operation *op) {
                                  return tiledAndFusedOps.contains(op);
                                });

    if (failed(newFusionOpportunities)) {
      LDBG() << "failed to fuse consumers, skip";
      return tiledResults->tiledAndFusedOps.front();
    }

    // Because we restrict to at most a single tilable consumer for yielding
    // a replacement, no new fusion opportunities will yield a replacement,
    // meaning there is no need to run consumer fusion again afterwards.
    // TODO: run producer and consumer fusion in one worklist.
    fuseProducersOfSlices(rewriter, *newFusionOpportunities, tileAndFuseOptions,
                          tilingLoops);
  }

  return tiledResults->tiledAndFusedOps.front();
}

namespace {
/// This pass starts with the first TilingInterface operation that has
/// lowering_config attribute, tiles the op and fuses its  consumers and
/// producers recursively. If the `onlyFuseProducerInputOperands` is set, it
/// only fuses producer input operands and disables consumer fusion. The
/// `tilingLevel` must be specified. It picks the `tilingLevel`-th list as
/// tiling sizes from lowering_config.
struct LLVMCPUTileAndFuseProducerConsumer
    : impl::LLVMCPUTileAndFuseProducerConsumerPassBase<
          LLVMCPUTileAndFuseProducerConsumer> {
  using impl::LLVMCPUTileAndFuseProducerConsumerPassBase<
      LLVMCPUTileAndFuseProducerConsumer>::
      LLVMCPUTileAndFuseProducerConsumerPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override;
};

void LLVMCPUTileAndFuseProducerConsumer::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  IRRewriter rewriter(funcOp);

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  Operation *anchorOp;
  if (anchorOnRootOp) {
    anchorOp = getRootOp(computeOps, tilingLevel);
  } else {
    // TODO(hanchung): Support the case that anchor op is root op's producer. It
    // is not common that the producer op has additional iteration dimensions,
    // but it should be handled like consumers. I.e., the additional dimensions
    // should be tiled. It is easier to support after we have solid lowering
    // config propagation.
    Operation *rootOp =
        getRootOp(computeOps, IREE::CPU::TilingLevel::DistributionTiles);
    anchorOp = getLastAnchorOp(computeOps, tilingLevel);
    if (rootOp &&
        llvm::find(computeOps, rootOp) > llvm::find(computeOps, anchorOp)) {
      LDBG() << "anchor op that is rootOp's producer is not supported";
      return;
    }
  }
  if (!anchorOp) {
    LDBG() << "unable to find an anchor operation that has "
           << IREE::CPU::getTilingLevelName(tilingLevel) << " config";
    return;
  }

  if (failed(tileRootAndFuseProducerConsumer(
          rewriter, cast<TilingInterface>(anchorOp), tilingLevel,
          onlyFuseProducerInputOperands))) {
    funcOp.emitError() << "tiling of level "
                       << IREE::CPU::getTilingLevelName(tilingLevel)
                       << " failed\n";
    return signalPassFailure();
  }

  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  scf::ForallOp::getCanonicalizationPatterns(patterns, context);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  // Pull in tensor dialect canonicalization patterns to fold tensor.cast
  // into producers when possible.
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    LDBG() << "----- cleanup failed -----";
    return signalPassFailure();
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileAndFuseProducerConsumerPass(
    IREE::CPU::TilingLevel tilingLevel) {
  LLVMCPUTileAndFuseProducerConsumerPassOptions options;
  options.tilingLevel = tilingLevel;
  options.onlyFuseProducerInputOperands = false;
  return std::make_unique<LLVMCPUTileAndFuseProducerConsumer>(options);
}
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileRootAndFuseInputOperandsPass(
    IREE::CPU::TilingLevel tilingLevel) {
  LLVMCPUTileAndFuseProducerConsumerPassOptions options;
  options.tilingLevel = tilingLevel;
  options.onlyFuseProducerInputOperands = true;
  return std::make_unique<LLVMCPUTileAndFuseProducerConsumer>(options);
}
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileLastOpAndFuseProducerConsumerPass(
    IREE::CPU::TilingLevel tilingLevel) {
  LLVMCPUTileAndFuseProducerConsumerPassOptions options;
  options.tilingLevel = tilingLevel;
  options.anchorOnRootOp = false;
  return std::make_unique<LLVMCPUTileAndFuseProducerConsumer>(options);
}
} // namespace mlir::iree_compiler
