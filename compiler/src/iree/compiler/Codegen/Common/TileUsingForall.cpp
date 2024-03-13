// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-tile-using-forall"

namespace mlir::iree_compiler {

static Operation *cloneAndFuseFirstUse(RewriterBase &rewriter,
                                       Operation *producerOp,
                                       Operation *containingOp) {
  LLVM_DEBUG(llvm::dbgs() << "Try to fuse an use by cloning\n");

  // Gather all uses inside the containing op.
  SmallVector<OpOperand *> uses;
  for (OpResult result : producerOp->getOpResults()) {
    for (OpOperand &use : result.getUses()) {
      if (containingOp->isProperAncestor(use.getOwner())) {
        uses.push_back(&use);
        continue;
      }
      // Cannot clone and fuse if the use is by the containing op itself: fail
      // immediately.
      if (containingOp == use.getOwner()) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "producer op use by containing op cannot be fused by cloning");
        return nullptr;
      }
    }
  }

  // Check for a non-empty list of fusion opportunities.
  if (uses.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "no fusion opportunity by cloning");
    return nullptr;
  }

  // Clone and fuse inside the containing op.
  Operation *fusedOp = nullptr;
  OpOperand *use = uses.front();
  // Parallel insert slice is not a valid clone destination.
  // TODO: Generalize to other type of ops.
  assert(!isa<tensor::ParallelInsertSliceOp>(use->getOwner()) &&
         "Parallel insert slice is not a valid clone destination");
  unsigned resultNumber = cast<OpResult>(use->get()).getResultNumber();
  LLVM_DEBUG(llvm::dbgs() << "resultNumber: " << resultNumber << "\n");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(use->getOwner());
  fusedOp = rewriter.clone(*producerOp);
  rewriter.modifyOpInPlace(
      use->getOwner(), [&] { use->set(fusedOp->getOpResult(resultNumber)); });

  return fusedOp;
}

/// Add new operands to the forall op for users of the producerOp
/// that are dominated by the containing scf.forall op.
static Operation *replaceForAllWithNewSignature(
    RewriterBase &rewriter, Operation *producerOp,
    Operation *containingOp, TilingResult &tileAndFuseResult,
    int64_t resultNumber, SmallVector<OpFoldResult> &offsets,
    SmallVector<OpFoldResult> &sizes) {

  // Count number of users not including the containing op
  SetVector<Operation *> dominatedUsers;
  DominanceInfo domInfo(containingOp);
  for (Operation *user : producerOp->getResult(resultNumber).getUsers()) {
    if (!containingOp->isAncestor(user) &&
        (domInfo.dominates(containingOp, user))) {
      dominatedUsers.insert(user);
    }
  }
  if (dominatedUsers.empty())
    return nullptr;

  // Create new scf.forall op
  auto forallOp = cast<scf::ForallOp>(containingOp);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  // Get new output
  Location loc = forallOp.getLoc();
  auto genericOp = dyn_cast<linalg::GenericOp>(producerOp);
  if (!genericOp)
    return nullptr;
  SmallVector<Value> outputs = genericOp.getOutputs();
  SmallVector<Value> newOuts(forallOp.getOutputs());
  newOuts.push_back(outputs[resultNumber]);

  // Create new scf.forall op
  auto newforallOp = rewriter.create<scf::ForallOp>(
      loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), newOuts, forallOp.getMapping());
  rewriter.eraseBlock(newforallOp.getBody());
  newforallOp.getRegion().takeBody(forallOp.getRegion());

  // Add additional block argument for new value being returned
  // and replaces all uses of the new output with corresponding bbArg
  // inside the scf.forall to enable fusion into this new scf.forall.
  newforallOp.getBody()->addArgument(newOuts.back().getType(),
                                     newOuts.back().getLoc());
  auto bbArgs = newforallOp.getBody()->getArguments();
  rewriter.replaceUsesWithIf(newOuts.back(), bbArgs.back(),
                             [&](OpOperand &use) {
                               Operation *op = use.getOwner();
                               return newforallOp->isProperAncestor(op);
                             });

  // Fix terminator
  scf::InParallelOp terminatorOp = newforallOp.getTerminator();
  SmallVector<Operation *> yieldingOps = llvm::to_vector<4>(llvm::map_range(
      terminatorOp.getYieldingOps(), [](Operation &op) { return &op; }));
  Operation *firstYieldOp = yieldingOps.front();
  rewriter.setInsertionPoint(firstYieldOp);
  Value src = tileAndFuseResult.tiledValues[0];
  Value dst = newforallOp.getRegionIterArgs().back();
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  rewriter.create<tensor::ParallelInsertSliceOp>(firstYieldOp->getLoc(), src,
                                                 dst, offsets, sizes, strides);

  for (auto result : llvm::enumerate(forallOp.getResults())) {
    rewriter.replaceAllUsesWith(result.value(),
                                newforallOp->getResult(result.index()));
  }
  rewriter.replaceUsesWithIf(producerOp->getResult(resultNumber),
                             newforallOp->getResults().back(),
                             [&](OpOperand &use) {
                               Operation *user = use.getOwner();
                               return dominatedUsers.contains(user);
                             });
  return newforallOp;
}

/// Find the first "extract" user of `producerOp` and tile it right before its
/// use. The tiled op is fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
/// If tiled op has uses that are dominated by `containingOp`, return
/// a new `containingOp` with results of the fused op appended to
/// results of the `containingOp` or nullptr if there are no dominated uses.
static std::tuple<SmallVector<Operation *>, Operation *>
tileAndFuseFirstExtractUse(RewriterBase &rewriter, Operation *producerOp,
                           Operation *containingOp) {
  LLVM_DEBUG(llvm::dbgs() << "Try to fuse a direct extract use\n");
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    LLVM_DEBUG(llvm::dbgs()
        << "producer is not a TileableInterface: " << *producerOp);
    return {};
  }

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto it = llvm::find_if(tileableProducer->getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (it == tileableProducer->getUsers().end()) {
    LLVM_DEBUG(llvm::dbgs()
        << "could not find fusion opportunity for: " << *tileableProducer);
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*it);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  int64_t resultNumber =
      cast<OpResult>(sliceOpToTile.getSource()).getResultNumber();
  LLVM_DEBUG(llvm::dbgs() << "resultNumber: " << resultNumber << "\n");

  SmallVector<OpFoldResult> offsets = sliceOpToTile.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOpToTile.getMixedSizes();

  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducer.generateResultTileValue(rewriter, resultNumber, offsets,
                                               sizes);

  if (failed(tileAndFuseResult)) {
    LLVM_DEBUG(llvm::dbgs()
        << "failed to tile producer op: " << *tileableProducer);
    return {};
  }

#ifndef NDEBUG
  for (auto *tiledOp : tileAndFuseResult->tiledOps) {
    LLVM_DEBUG(llvm::dbgs() << "tiledProducer: " << *tiledOp << "\n");
  }
#endif

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  if (failed(maybeRankReduced)) {
    LLVM_DEBUG(llvm::dbgs()
        << "shape types don't match (missing canonicalization?):\nTiledOp: "
        << tileAndFuseResult->tiledValues[0]
        << "\nSliceOp: " << sliceOpToTile.getOperation() << '\n');
    return {};
  }
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  // Add new outputs to containing op, if required
  Operation *newContainingOp = replaceForAllWithNewSignature(
      rewriter, producerOp, containingOp, *tileAndFuseResult,
      resultNumber, offsets, sizes);

  return std::make_tuple(tileAndFuseResult->tiledOps, newContainingOp);
}

/// First, find the first "scf::ForallOp" user of `producerOp` and ensure
/// it is exactly the `containingOp`, otherwise bail.
/// Then, find the first "extract" user of the tied block argument and tile it
/// right before its "extract" use. The tiled op is fused under the
/// `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static SmallVector<Operation *>
tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    RewriterBase &rewriter, Operation *producerOp,
    Operation *containingOp) {
  LLVM_DEBUG(
      llvm::dbgs() << "Try to fuse an extract use through block argument\n");

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    LLVM_DEBUG(llvm::dbgs()
               << "producer is not a TileableInterface: " << *producerOp);
    return {};
  }

  // Search the first use by a "scf::ForallOp" user.
  scf::ForallOp forallOp;
  auto itProducerUses =
      llvm::find_if(tileableProducer->getUses(), [&](OpOperand &use) {
        forallOp = dyn_cast<scf::ForallOp>(use.getOwner());
        return forallOp;
      });
  // If it's not from the containing op, return.
  if (!forallOp || forallOp != containingOp) {
    LLVM_DEBUG(llvm::dbgs()
        << "could not find a use by the containing op: " << *tileableProducer);
    return {};
  }

  // Search the producer slices accessed within the containing
  // operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples.
  //   Maybe evolve into an interface.
  OpOperand *pUse = &(*itProducerUses);
  BlockArgument bbArg = forallOp.getTiedBlockArgument(pUse);

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto itBBArgUsers = llvm::find_if(bbArg.getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (itBBArgUsers == bbArg.getUsers().end()) {
    LLVM_DEBUG(llvm::dbgs()
               << "could not find fusion opportunity for bbArg: " << bbArg);
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*itBBArgUsers);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Replace the use in the tileableProducer before tiling: clone, replace and
  // then tile.
  int64_t resultNumber = cast<OpResult>(pUse->get()).getResultNumber();
  LLVM_DEBUG(llvm::dbgs() << "resultNumber: " << resultNumber << "\n");

  // Gather destination tensors.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(
          rewriter, tileableProducer->getLoc(), tileableProducer,
          destinationTensors))) {
    LLVM_DEBUG(llvm::dbgs() << "failed to get destination tensors for: "
                            << *tileableProducer);
    return {};
  }

  IRMapping bvm;
  bvm.map(destinationTensors[resultNumber], bbArg);
  auto tileableProducerClone =
      cast<TilingInterface>(rewriter.clone(*tileableProducer, bvm));
  auto scopeGuard =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(tileableProducerClone); });

  // Tile the producer.
  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducerClone.generateResultTileValue(
          rewriter, resultNumber, sliceOpToTile.getMixedOffsets(),
          sliceOpToTile.getMixedSizes());
  if (failed(tileAndFuseResult)) {
    LLVM_DEBUG(llvm::dbgs()
        << "failed to tile producer op: " << *tileableProducer);
    return {};
  }

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  assert(succeeded(maybeRankReduced) && "unexpected shape");
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  // Replace the use in containingOp.
  rewriter.modifyOpInPlace(containingOp, [&]() {
    containingOp->setOperand(pUse->getOperandNumber(),
                             destinationTensors.front());
  });

  return tileAndFuseResult->tiledOps;
}

namespace {
class TileUsingForallPass : public TileUsingForallBase<TileUsingForallPass> {
public:
  TileUsingForallPass(int64_t tilingLevel = -1) {
    this->tilingLevel.setValue(tilingLevel);
  }

  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override;
};

} // namespace

void TileUsingForallPass::runOnOperation() {
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
  auto first = dyn_cast<TilingInterface>(computeOps.back());
  SmallVector<OpFoldResult> mixedNumThreads;
  SmallVector<OpFoldResult> mixedTileSizes =
      llvm::map_to_vector(loweringConfig->getTileSizeVals(tilingLevel),
                          [&](int64_t size) -> OpFoldResult {
                            return rewriter.getIndexAttr(size);
                          });

  Attribute bX = gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimX);
  Attribute bY = gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimY);
  Attribute bZ = gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimZ);
  auto mapping = rewriter.getArrayAttr({bZ, bY, bX});

  rewriter.setInsertionPoint(first);
  FailureOr<linalg::ForallTilingResult> maybeTilingResult = failure();
  if (!mixedNumThreads.empty()) {
    maybeTilingResult =
        linalg::tileToForallOp(rewriter, first, mixedNumThreads, mapping);
  } else {
    maybeTilingResult = linalg::tileToForallOpUsingTileSizes(
        rewriter, first, mixedTileSizes, mapping);
  }
  if (failed(maybeTilingResult)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to tile using forall op");
    return signalPassFailure();
  }
  rewriter.replaceOp(first, maybeTilingResult->tileOp->getResults());

  {
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(ctx);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    ctx->getLoadedDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
        patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
      return signalPassFailure();
    }
  }

  auto forallOp = cast<scf::ForallOp>(maybeTilingResult->tileOp);
  computeOps.clear();
  for (auto op:  getComputeOps(funcOp)) {
    if (op->getParentOp() == forallOp)
      continue;
    computeOps.push_back(op);
  }
  SetVector<Operation *> remainingProducers(computeOps.begin(),
                                            computeOps.end());
  auto getNextProducer = [&]() -> FailureOr<Operation *> {
    for (const auto &it : enumerate(remainingProducers)) {
      Operation *producerOp = it.value();
      // The containing op may be a user of producerOp: use isAncestor.
      int64_t numUsesInContainingOp =
          llvm::count_if(producerOp->getUsers(), [&](Operation *op) {
            return forallOp->isAncestor(op);
          });
      // TODO: When resolving the TODO below (no duplicate ops), take an op
      // that has no use among the remaining producers. This is a topological
      // sorting.
      if (numUsesInContainingOp > 0) {
        if (numUsesInContainingOp == 1)
          remainingProducers.erase(remainingProducers.begin() + it.index());
        return producerOp;
      }
    }
    return failure();
  };
  while (!remainingProducers.empty()) {
    auto nextProducer = getNextProducer();
    if (failed(nextProducer)) {
      break;
    }
    Operation *producerOp = *nextProducer;

    // TODO: If there are multiple uses of the producer in the containing op,
    // we currently tile/clone the op multiple times (once per use). In some
    // cases, we can tile/clone once and reuse the value for each use.
    // Futhermore, producers should then be traversed according to a
    // topological sorting.
    auto [tiledOps, newContainingOp] =
        tileAndFuseFirstExtractUse(rewriter, producerOp, forallOp);
    if (!tiledOps.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "\nFused a direct extract use\n" << *forallOp);
      if (newContainingOp) {
        // Update handles associated with the containing op so we don't need to
        // invalidate them. This is a hack to support better composability
        // between tiling and fusion while a proper mechanism is being
        // investigated.
        //
        // DO NOT replicate this elsewhere unless you understand what you are
        // doing.
        rewriter.eraseOp(forallOp);
        forallOp = cast<scf::ForallOp>(newContainingOp);
      }
      continue;
    }

    SmallVector<Operation *> tiledContainingOpOperand =
        tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
            rewriter, producerOp, forallOp);
    if (!tiledContainingOpOperand.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\nFused an extract use through block argument\n"
                 << *forallOp);
      continue;
    }

    Operation *cloned = cloneAndFuseFirstUse(rewriter, producerOp, forallOp);
    if (cloned) {
      LLVM_DEBUG(llvm::dbgs() << "\nFused an use by cloning\n" << *forallOp);
      continue;
    }
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileUsingForallPass(int32_t tilingLevel) {
  return std::make_unique<TileUsingForallPass>(tilingLevel);
}

} // namespace mlir::iree_compiler
