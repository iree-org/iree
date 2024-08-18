// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-tile-root-and-fuse-producers-consumers"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUTILEROOTANDFUSEPRODUCERCONSUMERPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

/// Starting from `op` walk all operands backwards to find all
/// potentially fusable operations, i.e. operations that implement
/// the `TilingInterface`.
static void collectTiledAndFusedOps(Operation *rootOp,
                                    llvm::SmallDenseSet<Operation *> &result) {
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);
  result.insert(rootOp);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !isa<TilingInterface>(producer) ||
          result.count(producer))
        continue;
      worklist.push_back(producer);
      result.insert(producer);
    }
  }
}

/// Tile the root operation and fuse the producers of the root operation.
/// If `onlyFuseProducerInputOperands` is set, only fuse producer input
/// operands. Returns the tiled operation to be used for fusing consumers.
FailureOr<Operation *>
tileRootAndFuseProducers(IRRewriter &rewriter, TilingInterface rootOp,
                         int64_t tilingLevel,
                         bool onlyFuseProducerInputOperands) {
  mlir::DominanceInfo dominanceInfo(rootOp);
  llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
  collectTiledAndFusedOps(rootOp, tiledAndFusedOps);

  llvm::DenseSet<Operation *> yieldReplacementsFor;
  for (auto op : tiledAndFusedOps) {
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          return dominanceInfo.properlyDominates(rootOp, user);
        })) {
      yieldReplacementsFor.insert(op);
    }
  }

  SmallVector<OpFoldResult> tileSizes =
      getLoweringConfig(rootOp).getTilingLevelSizes(rewriter, tilingLevel,
                                                    rootOp);

  // Pad the tile sizes with zero.
  auto zero = rewriter.getIndexAttr(0);
  int64_t numLoops = rootOp.getLoopIteratorTypes().size();
  if (tileSizes.size() > numLoops) {
    LLVM_DEBUG(llvm::dbgs()
               << "tile sizes size " << tileSizes.size()
               << " exceeds the number of loops " << numLoops << "\n");
    return failure();
  }
  tileSizes.resize(numLoops, zero);

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);

  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand) {
        Operation *owner = originalProducer.getOwner();
        bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
        // Do not fuse destination operands if onlyFuseProducerInputOperands is
        // true.
        bool shouldFuse =
            !(onlyFuseProducerInputOperands && isDestinationOperand);
        return std::make_tuple(shouldFuse, yieldProducerReplacement);
      };
  tileAndFuseOptions.setFusionControlFn(controlFn);

  FailureOr<scf::SCFTileAndFuseResult> tiledResults =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, rootOp,
                                                tileAndFuseOptions);
  if (failed(tiledResults)) {
    return failure();
  }

  // Perform the replacement of tiled and fused values.
  SmallVector<Operation *> opsToReplace{rootOp};
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

  return tiledResults->tiledAndFusedOps.front();
}

static void fuseConsumers(RewriterBase &rewriter, Operation *tiledOp) {

  //  Typically, the consumers of the tiled operation are slices of the
  //  results of the tiled operation. These are expressed in IR using
  //  `tensor.insert_slice` operations, whose outputs are the operands of the
  //  untiled operation. Create a worklist of these `tensor.insert_siices`
  //  operations. If the consumers of the source of the `tensor.insert_slices`
  //  can be tiled such that the tiled value is generated in-place, that
  //  effectively tiles + fuses the operations.
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::queue<tensor::InsertSliceOp> &candidates) {
    for (auto *userOp : fusedOp->getResults().getUsers()) {
      if (auto sliceOp = llvm::dyn_cast<tensor::InsertSliceOp>(userOp)) {
        candidates.push(sliceOp);
      }
    }
  };

  // Collect the candidate slices which can be potential consumers that can be
  // fused.
  std::queue<tensor::InsertSliceOp> candidates;
  addCandidateSlices(tiledOp, candidates);

  while (!candidates.empty()) {

    // Traverse the slices in BFS fashion.
    tensor::InsertSliceOp candidateSliceOp = candidates.front();
    candidates.pop();

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        mlir::scf::tileAndFuseConsumerOfSlice(rewriter, candidateSliceOp);
    if (failed(fusedResult)) {
      LLVM_DEBUG(llvm::dbgs() << "failed to fuse consumer of slice: "
                              << candidateSliceOp << "\n");
      continue;
    }

    // Replace the original consumer operation with the tiled implementation.
    rewriter.replaceOp(fusedResult->origConsumerOperand->getOwner(),
                       fusedResult->tiledOps.front());

    // The result of the fused conumers might themselved be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    addCandidateSlices(fusedResult->tiledAndFusedConsumerOperand->getOwner(),
                       candidates);
  }
}

/// Implementation of tile root and fuse producers and consumers greedily.
/// If `onlyFuseProducerInputOperands` is set, only fuse producer input operands
/// and disable consumer fusion.
static LogicalResult tileRootAndFuse(IRRewriter &rewriter,
                                     TilingInterface rootOp,
                                     int64_t tilingLevel,
                                     bool onlyFuseProducerInputOperands) {

  FailureOr<Operation *> tiledOp = tileRootAndFuseProducers(
      rewriter, rootOp, tilingLevel, onlyFuseProducerInputOperands);

  if (failed(tiledOp))
    return failure();

  if (!onlyFuseProducerInputOperands)
    fuseConsumers(rewriter, tiledOp.value());

  return success();
}

/// This pass starts with the first TilingInterface operation that has
/// lowering_config attribute, tiles the op and fuses its  consumers and
/// producers recursively. If the `onlyFuseProducerInputOperands` is set, it
/// only fuses producer input operands and disables consumer fusion. The
/// `tilingLevel` must be specified. It picks the `tilingLevel`-th list as
/// tiling sizes from lowering_config.
struct LLVMCPUTileRootAndFuseProducerConsumer
    : impl::LLVMCPUTileRootAndFuseProducerConsumerPassBase<
          LLVMCPUTileRootAndFuseProducerConsumer> {
  using impl::LLVMCPUTileRootAndFuseProducerConsumerPassBase<
      LLVMCPUTileRootAndFuseProducerConsumer>::
      LLVMCPUTileRootAndFuseProducerConsumerPassBase;
  explicit LLVMCPUTileRootAndFuseProducerConsumer(
      int64_t tilingLevel, bool onlyFuseProducerInputOperands) {
    this->tilingLevel = tilingLevel;
    this->onlyFuseProducerInputOperands = onlyFuseProducerInputOperands;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void LLVMCPUTileRootAndFuseProducerConsumer::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  IRRewriter rewriter(funcOp);

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);

  if (failed(rootOp)) {
    funcOp.emitError() << "not able to find the root operation\n";
    return signalPassFailure();
  }

  IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
      getLoweringConfig(rootOp.value());
  if (!loweringConfig) {
    funcOp.emitError() << "not able to find the lowering config\n";
    return signalPassFailure();
  }

  if (!loweringConfig.hasTilingLevel(tilingLevel)) {
    funcOp.emitError()
        << "not able to find the lowering config with the tiling level "
        << tilingLevel.getValue() << "\n";
    return signalPassFailure();
  }

  if (failed(tileRootAndFuse(
          rewriter, dyn_cast<TilingInterface>(rootOp.value()),
          tilingLevel.getValue(), onlyFuseProducerInputOperands.getValue()))) {
    funcOp.emitError() << "tiling of level " << tilingLevel.getValue()
                       << " failed\n";
    return signalPassFailure();
  }

  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  // Pull in tensor dialect canonicalization patterns to fold tensor.cast
  // into producers when possible.
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
    return signalPassFailure();
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileRootAndFuseProducerConsumer(int64_t tilingLevel) {
  return std::make_unique<LLVMCPUTileRootAndFuseProducerConsumer>(
      tilingLevel, /*onlyFuseProducerInputOperands=*/false);
}
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileRootAndFuseInputOperands(int64_t tilingLevel) {
  return std::make_unique<LLVMCPUTileRootAndFuseProducerConsumer>(
      tilingLevel, /*onlyFuseProducerInputOperands=*/true);
}
} // namespace mlir::iree_compiler
