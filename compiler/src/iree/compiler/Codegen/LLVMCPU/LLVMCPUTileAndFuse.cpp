// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
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

#define DEBUG_TYPE "iree-llvmcpu-tile-and-fuse"

namespace mlir::iree_compiler {

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

/// Tiling of `tensor.pad` operation generates
///
/// ```mlir
/// scf.if {
///   ...
/// } else {
///    tensor.pad
/// }
/// ```
///
/// For IREEs use case we dont need this. So this folds away the `if` condition.
/// Note this is a fairly hacky workaround, but the current pad operation
/// semantics force us down this path.
static FailureOr<tensor::PadOp>
foldIfGeneratedFromPadding(RewriterBase &rewriter, tensor::PadOp untiledPadOp,
                           tensor::PadOp tiledPadOp) {
  auto ifOp = dyn_cast<scf::IfOp>(tiledPadOp->getParentOp());
  if (!ifOp) {
    return failure();
  };
  Block *block = tiledPadOp->getBlock();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, ifOp, /*blockArgs=*/{});
  rewriter.replaceOp(ifOp, results);
  rewriter.eraseOp(terminator);
  return tiledPadOp;
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
                    linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};

LogicalResult applyTileAndFuse(RewriterBase &rewriter, Operation *rootOp,
                               DominanceInfo &dominanceInfo,
                               scf::SCFTilingOptions options) {
  llvm::SmallDenseSet<Operation *> origTiledAndFusedOps;
  collectTiledAndFusedOps(rootOp, origTiledAndFusedOps);
  auto isIgnoredUser = [&](Operation *user,
                           LoopLikeOpInterface outerMostTiledLoop) {
    return origTiledAndFusedOps.count(user) ||
           isa<tensor::DimOp, IREE::Codegen::UKernelGenericOp>(user);
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

  // WAR for `if` ops generating `scf.if` operations.
  if (auto rootPadOp = dyn_cast<tensor::PadOp>(rootOp)) {
    assert(tilingResult->tiledOps.size() == 1 &&
           "expected tiling of `pad` op to return only one operation");
    FailureOr<Operation *> replacementTiledOp = foldIfGeneratedFromPadding(
        rewriter, rootPadOp, cast<tensor::PadOp>(tilingResult->tiledOps[0]));
    if (!failed(replacementTiledOp)) {
      tilingResult->tiledOps[0] = replacementTiledOp.value();
    }
  } else if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(rootOp)) {
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

void LLVMCPUTileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  TilingInterface consumerOp;
  funcOp.walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface op) {
    // Find the next consumer op if it does not have loops.
    if (op.getLoopIteratorTypes().empty())
      return WalkResult::advance();
    consumerOp = op;
    return WalkResult::interrupt();
  });
  if (!consumerOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no consumer op -----\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "consumerOp: " << consumerOp << "\n");
  LLVM_DEBUG(llvm::dbgs() << "tilingLevel: " << tilingLevel << "\n");

  // If `consumerOp` has its own lowering config, we prefer using it. Otherwise,
  // fallback to find a lowering_config from other operations.
  SmallVector<int64_t> tileSizes;
  SmallVector<bool> tileScalableFlags;
  if (auto loweringConfig = getLoweringConfig(consumerOp)) {
    tileSizes = loweringConfig.getTileSizeVals(tilingLevel);
    tileScalableFlags = loweringConfig.getScalableTileFlagVals(tilingLevel);
  } else {
    FailureOr<IREE::Codegen::LoweringConfigAttr> maybeLoweringConfig =
        getLoweringConfig(getComputeOps(funcOp));
    if (failed(maybeLoweringConfig)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "can't find lowering_config, skip TileAndFuse");
      return;
    }
    tileSizes = maybeLoweringConfig.value().getTileSizeVals(tilingLevel);
    tileScalableFlags =
        maybeLoweringConfig.value().getScalableTileFlagVals(tilingLevel);
  }

  if (llvm::all_of(tileSizes, [&](int64_t size) { return size == 0; })) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, all zeros -----\n");
    return;
  }

  scf::SCFTilingOptions options{};
  setSCFTileSizes(options, consumerOp, std::move(tileSizes),
                  std::move(tileScalableFlags));

  IRRewriter rewriter(context);
  DominanceInfo dominanceInfo(funcOp);
  if (failed(applyTileAndFuse(rewriter, consumerOp, dominanceInfo, options))) {
    LLVM_DEBUG(llvm::dbgs() << "----- tile and fuse failed -----\n");
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
createLLVMCPUTileAndFusePass(int64_t tilingLevel) {
  return std::make_unique<LLVMCPUTileAndFusePass>(tilingLevel);
}

} // namespace mlir::iree_compiler
