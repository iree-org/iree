// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-tile-and-fuse"

namespace mlir {
namespace iree_compiler {
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

/// This pass starts with the last TilingInterface operation, tiles the op and
/// fuses its producers recursively. The `tilingLevel` must be specified. It
/// picks the `tilingLevel`-th list as tiling sizes from lowering_config.
struct LLVMCPUTileAndFusePass : LLVMCPUTileAndFuseBase<LLVMCPUTileAndFusePass> {
  LLVMCPUTileAndFusePass(int64_t tilingLevel = -1) {
    this->tilingLevel.setValue(tilingLevel);
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};

LogicalResult applyTileAndFuse(RewriterBase &rewriter, Operation *rootOp,
                               scf::SCFTilingOptions options) {
  llvm::SmallDenseSet<Operation *> origTiledAndFusedOps;
  collectTiledAndFusedOps(rootOp, origTiledAndFusedOps);
  auto isIgnoredUser = [&](Operation *user, scf::ForOp outerMostTiledLoop) {
    return origTiledAndFusedOps.count(user) || isa<tensor::DimOp>(user) ||
           outerMostTiledLoop->isAncestor(user);
  };

  // The rest of this method is similar to
  // scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp, except that also
  // yields replacements for values of the fused producer.

  // 1. Tile the consumer.
  SmallVector<OpResult> yieldedValuesToOrigValues;
  SmallVector<Operation *> tiledOps;
  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCFForOp(rewriter, cast<TilingInterface>(rootOp), options);
  if (failed(tilingResult)) {
    return failure();
  }
  yieldedValuesToOrigValues.append(rootOp->result_begin(),
                                   rootOp->result_end());
  tiledOps.append(tilingResult->tiledOps);

  // 2. Tiling each operation results in generation of slices. The source of
  // these slices could be producers that can be fused into the tiled loops by
  // computing the slices of these producers in-place. This results in more
  // slices created for operands of the "fused producer". This open up more
  // opportunities for fusion. Use a worklist to fuse greedily.
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::deque<tensor::ExtractSliceOp> &candidates) {
    for (Value operand : fusedOp->getOperands())
      if (auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>())
        candidates.push_back(sliceOp);
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
        tileAndFuseProducerOfSlice(rewriter, candidateSliceOp,
                                   tilingResult->loops);
    if (!fusedProducer) continue;

    // Check if the fused producer has other uses that require the value
    // to be yielded from within the tiled loop.
    OpResult untiledProducer = fusedProducer->origProducer;
    if (llvm::any_of(untiledProducer.getUsers(), [&](Operation *user) {
          return !isIgnoredUser(user, tilingResult->loops.front());
        })) {
      yieldReplacementForFusedProducer(rewriter, candidateSliceOp,
                                       fusedProducer.value(),
                                       tilingResult->loops);
      yieldedValuesToOrigValues.push_back(untiledProducer);
    }

    // Add more fusion candidates to the worklist.
    for (auto tiledOp : fusedProducer->tiledOps) {
      addCandidateSlices(tiledOp, candidates);
      tiledOps.push_back(tiledOp);
    }
  }

  scf::ForOp outermostLoop = tilingResult->loops.front();
  for (auto [index, origVal] : llvm::enumerate(yieldedValuesToOrigValues)) {
    Value replacement = outermostLoop.getResult(index);
    rewriter.replaceUsesWithIf(origVal, replacement, [&](OpOperand &use) {
      return !isIgnoredUser(use.getOwner(), outermostLoop);
    });
  }

  return success();
}

void LLVMCPUTileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  TilingInterface consumerOp;
  funcOp.walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface op) {
    // Find the next consumer op if it does not have loops.
    if (op.getLoopIteratorTypes().empty()) return WalkResult::advance();
    consumerOp = op;
    return WalkResult::interrupt();
  });
  if (!consumerOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no consumer op -----\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "consumerOp: " << consumerOp << "\n");
  LLVM_DEBUG(llvm::dbgs() << "tilingLevel: " << tilingLevel << "\n");

  FailureOr<IREE::Codegen::LoweringConfigAttr> maybeLoweringConfig =
      getLoweringConfig(getComputeOps(funcOp));
  if (failed(maybeLoweringConfig)) {
    LLVM_DEBUG(llvm::dbgs() << "can't find lowering_config, skip TileAndFuse");
    return;
  }

  int numLoops = consumerOp.getLoopIteratorTypes().size();
  SmallVector<int64_t> tilingSizes =
      maybeLoweringConfig.value().getTileSizeVals(tilingLevel);
  if (numLoops > tilingSizes.size()) {
    tilingSizes.append(numLoops - tilingSizes.size(), 0);
  }
  tilingSizes.resize(numLoops);

  if (llvm::all_of(tilingSizes, [&](int64_t size) { return size == 0; })) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, all zeros -----\n");
    return;
  }

  auto options = scf::SCFTilingOptions().setTileSizes(tilingSizes);
  IRRewriter rewriter(context);
  if (failed(applyTileAndFuse(rewriter, consumerOp, options))) {
    LLVM_DEBUG(llvm::dbgs() << "----- tile and fuse failed -----\n");
    return signalPassFailure();
  }

  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  // Pull in tensor dialect canonicalization patterns to fold tensor.cast
  // into producers when possible.
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
    return signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUTileAndFusePass(
    int64_t tilingLevel) {
  return std::make_unique<LLVMCPUTileAndFusePass>(tilingLevel);
}

}  // namespace iree_compiler
}  // namespace mlir
