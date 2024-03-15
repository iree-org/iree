// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-tile-matmul-and-fuse-img2col"

namespace mlir::iree_compiler {

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

LogicalResult applyTileAndFuse(RewriterBase &rewriter, Operation *rootOp,
                               DominanceInfo &dominanceInfo,
                               scf::SCFTilingOptions options) {
  llvm::SmallDenseSet<Operation *> origTiledAndFusedOps;
  collectTiledAndFusedOps(rootOp, origTiledAndFusedOps);
  auto isIgnoredUser = [&](Operation *user,
                           LoopLikeOpInterface outerMostTiledLoop) {
    return origTiledAndFusedOps.count(user) || isa<tensor::DimOp>(user);
  };

  // The rest of this method is similar to
  // scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp, except that this
  // replaces DPS out operand with iter_arg when they use the same init
  // operands.

  // 1. Tile the consumer.
  SmallVector<OpResult> yieldedValuesToOrigValues;
  SmallVector<Operation *> tiledOps;
  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCF(rewriter, cast<TilingInterface>(rootOp), options);
  if (failed(tilingResult)) {
    return failure();
  }
  yieldedValuesToOrigValues.append(rootOp->result_begin(),
                                   rootOp->result_end());
  // A map from untiled value to scf.for iter_arg. The iter_arg is used for DPS
  // init operand if they use the same init operands.
  llvm::DenseMap<Value, Value> mapToIterArg;

  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(rootOp)) {
    for (auto [init, iterArg] : llvm::zip_equal(
             dpsOp.getDpsInits(),
             cast<scf::ForOp>(tilingResult->loops.back().getOperation())
                 .getRegionIterArgs())) {
      mapToIterArg[init] = iterArg;
    }
  }
  tiledOps.append(tilingResult->tiledOps);

  // 2. Tiling each operation results in generation of slices. The source of
  // these slices could be producers that can be fused into the tiled loops by
  // computing the slices of these producers in-place. This results in more
  // slices created for operands of the "fused producer". This open up more
  // opportunities for fusion. Use a worklist to fuse greedily.
  auto addCandidateSlices =
      [&](Operation *fusedOp, std::deque<tensor::ExtractSliceOp> &candidates) {
        for (OpOperand &operand : fusedOp->getOpOperands()) {
          auto sliceOp = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
          if (!sliceOp)
            continue;
          // ========= HERE IS THE ONLY DIFFERENCE FROM LLVMCPUTileAndFuse ===
          // TODO: This should probably only fuse the input slices and
          // not the output slices.
          if (!sliceOp.getSource().getDefiningOp<linalg::GenericOp>())
            continue;
          candidates.push_back(sliceOp);

          auto dpsOp = dyn_cast<DestinationStyleOpInterface>(fusedOp);
          if (!dpsOp)
            continue;

          if (dpsOp.isDpsInit(&operand) &&
              mapToIterArg.contains(sliceOp.getSource())) {
            rewriter.startOpModification(sliceOp);
            sliceOp.getSourceMutable().assign(
                mapToIterArg[sliceOp.getSource()]);
            rewriter.finalizeOpModification(sliceOp);
          }
        }
      };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tilingResult->tiledOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // Materialize the slice of the producer in place.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        scf::tileAndFuseProducerOfSlice(rewriter, candidateSliceOp,
                                        tilingResult->loops);
    if (!fusedProducer)
      continue;

    // Check if the fused producer has other uses that require the value
    // to be yielded from within the tiled loop.
    OpResult untiledProducer = fusedProducer->origProducer;
    if (llvm::any_of(untiledProducer.getUsers(), [&](Operation *user) {
          return !isIgnoredUser(user, tilingResult->loops.front()) &&
                 !tilingResult->loops.front()->isAncestor(user);
          ;
        })) {
      if (failed(scf::yieldReplacementForFusedProducer(
              rewriter, candidateSliceOp, fusedProducer.value(),
              tilingResult->loops))) {
        return failure();
      }
      yieldedValuesToOrigValues.push_back(untiledProducer);
    }

    // Add more fusion candidates to the worklist.
    for (auto tiledOp : fusedProducer->tiledOps) {
      addCandidateSlices(tiledOp, candidates);
      tiledOps.push_back(tiledOp);
    }
  }

  auto outermostLoop = cast<scf::ForOp>(tilingResult->loops.front());
  for (auto [index, origVal] : llvm::enumerate(yieldedValuesToOrigValues)) {
    Value replacement = outermostLoop.getResult(index);
    rewriter.replaceUsesWithIf(origVal, replacement, [&](OpOperand &use) {
      return !isIgnoredUser(use.getOwner(), outermostLoop) &&
             dominanceInfo.properlyDominates(outermostLoop, use.getOwner());
    });
  }
  return success();
}

namespace {

class LLVMGPUTileMatmulAndFuseImg2ColPass
    : public LLVMGPUTileMatmulAndFuseImg2ColBase<
          LLVMGPUTileMatmulAndFuseImg2ColPass> {
public:
  LLVMGPUTileMatmulAndFuseImg2ColPass(int64_t tilingLevel = -1) {
    this->tilingLevel.setValue(tilingLevel);
  }

  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override;
};

} // namespace

void LLVMGPUTileMatmulAndFuseImg2ColPass::runOnOperation() {
  if (tilingLevel == -1) {
    LLVM_DEBUG(llvm::dbgs() << "tilingLevel not set, skip tiling\n");
    return;
  }

  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<IREE::Codegen::LoweringConfigAttr> loweringConfig =
      getLoweringConfig(computeOps);
  if (failed(loweringConfig)) {
    LLVM_DEBUG(llvm::dbgs() << "can't find lowering_config, skip tiling\n");
    return;
  }

  IRRewriter rewriter(ctx);
  SmallVector<OpFoldResult> tileSizes = llvm::map_to_vector(
      loweringConfig->getTileSizeVals(tilingLevel),
      [&](int64_t val) -> OpFoldResult { return rewriter.getIndexAttr(val); });
  auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);

  linalg::LinalgOp matmul;
  auto found = funcOp->walk([&](linalg::LinalgOp op) {
    if (op.getNumReductionLoops() == 0) {
      return WalkResult::advance();
    }
    if (op.getNumReductionLoops() != 1) {
      return WalkResult::interrupt();
    }
    if (matmul) {
      return WalkResult::interrupt();
    }
    matmul = op;
    return WalkResult::advance();
  });

  if (found.wasInterrupted()) {
    LLVM_DEBUG(llvm::dbgs() << "skip, expect a single matmul\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "candidate: " << matmul << "\n");

  DominanceInfo dominanceInfo(funcOp);
  if (failed(applyTileAndFuse(rewriter, matmul, dominanceInfo, options))) {
    LLVM_DEBUG(llvm::dbgs() << "----- tile and fuse failed -----\n");
    return signalPassFailure();
  }

  {
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(ctx);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    ctx->getLoadedDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
        patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUTileMatmulAndFuseImg2ColPass(int tilingLevel) {
  return std::make_unique<LLVMGPUTileMatmulAndFuseImg2ColPass>(tilingLevel);
}

} // namespace mlir::iree_compiler
