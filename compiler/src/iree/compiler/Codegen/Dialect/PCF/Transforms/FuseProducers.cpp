// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-pcf-fuse-producers"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_FUSEPRODUCERSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct FuseProducersPass final
    : impl::FuseProducersPassBase<FuseProducersPass> {
  void runOnOperation() override;
};

struct FuseProducerIntoGenericOp
    : public OpRewritePattern<IREE::PCF::GenericOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    ProducerFusionParams params;
    if (failed(matchTilableProducer(rewriter, genericOp, params))) {
      return failure();
    }
    fuseTilableProducer(rewriter, genericOp, params);
    return success();
  }
};

struct FuseProducerIntoLoopOp : public OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    ProducerFusionParams params;
    if (failed(matchTilableProducer(rewriter, loopOp, params))) {
      return failure();
    }
    fuseTilableProducer(rewriter, loopOp, params);
    return success();
  }
};

void FuseProducersPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<FuseProducerIntoGenericOp, FuseProducerIntoLoopOp>(
      &getContext());
  populatePCFDropUnusedResultPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

//===---------------------------------------------------------------------===//
// Producer fusion impls
//===---------------------------------------------------------------------===//

/// For a given result of a scoped op, find all pcf.read_slice ops on the
/// corresponding sref region argument that read the init value.
template <typename OpTy>
static LogicalResult
lookupConsumerSlices(OpResult result,
                     SmallVectorImpl<PCF::ReadSliceOp> &slices) {
  OpTy owner = cast<OpTy>(result.getOwner());
  Value tiedArg = owner.getRegionRefArgs()[result.getResultNumber()];

  // Collect all read_slice ops on this sref arg. Write_slice users are
  // allowed but not tracked. Any other user type prevents fusion.
  SmallVector<PCF::ReadSliceOp> reads;
  for (Operation *user : tiedArg.getUsers()) {
    if (isa<PCF::WriteSliceOp>(user)) {
      continue;
    }
    auto readOp = dyn_cast<PCF::ReadSliceOp>(user);
    if (!readOp || !readOp.hasUnitStride()) {
      return failure();
    }
    reads.push_back(readOp);
  }

  // With SyncOnReturn semantics, the only case where a read does not see the
  // init value is when there are overlapping reads and writes, which is
  // explicitly undefined behavior. We may therefore optimize as though all
  // reads see the init value.
  auto srefType = cast<PCF::ShapedRefType>(tiedArg.getType());
  if (!srefType.isReturnOnlySync()) {
    return failure();
  }

  slices.append(reads.begin(), reads.end());
  return success();
}

template <typename OpTy>
static LogicalResult matchTilableProducerImpl(RewriterBase &rewriter,
                                              OpTy scopedOp,
                                              ProducerFusionParams &params) {
  for (int64_t i = 0, e = scopedOp->getNumResults(); i < e; ++i) {
    OpOperand *tiedInit = scopedOp.getTiedInit(i);
    if (!tiedInit) {
      continue;
    }

    Value initValue = tiedInit->get();
    Operation *defOp = initValue.getDefiningOp();
    if (!defOp) {
      continue;
    }
    auto producer = dyn_cast<TilingInterface>(defOp);
    if (!producer) {
      continue;
    }

    // Must be DPS with single result.
    auto dpsProducer = dyn_cast<DestinationStyleOpInterface>(defOp);
    if (!dpsProducer || defOp->getNumResults() != 1) {
      continue;
    }

    // Verify dominance of producer's operands.
    DominanceInfo dominanceInfo(scopedOp->getParentOp());
    bool allDominate = llvm::all_of(defOp->getOperands(), [&](Value v) {
      return dominanceInfo.dominates(v, scopedOp);
    });
    if (!allDominate) {
      continue;
    }

    // Find read_slice ops on the corresponding sref arg.
    SmallVector<PCF::ReadSliceOp> readSlices;
    if (failed(
            lookupConsumerSlices<OpTy>(scopedOp->getOpResult(i), readSlices))) {
      continue;
    }

    if (readSlices.empty()) {
      continue;
    }

    params.resultIndex = i;
    params.producer = defOp;
    params.readSlices = std::move(readSlices);
    return success();
  }
  return rewriter.notifyMatchFailure(scopedOp, "no fusable producer found");
}

template <typename OpTy>
static void fuseTilableProducerImpl(RewriterBase &rewriter, OpTy scopedOp,
                                    const ProducerFusionParams &params) {
  auto producer = cast<TilingInterface>(params.producer);

  // Step 1: Replace the init with the producer's DPS init.
  auto dpsProducer = cast<DestinationStyleOpInterface>(params.producer);
  Value producerDPSInit = dpsProducer.getDpsInits()[0];

  OpOperand *tiedInit = scopedOp.getTiedInit(params.resultIndex);
  rewriter.modifyOpInPlace(scopedOp,
                           [&]() { tiedInit->assign(producerDPSInit); });

  // Step 2: At each read_slice, generate tiled producer.
  for (PCF::ReadSliceOp readSlice : params.readSlices) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(readSlice);
    Location loc = readSlice.getLoc();

    SmallVector<OpFoldResult> offsets = readSlice.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = readSlice.getMixedSizes();

    FailureOr<TilingResult> tiledResult = producer.generateResultTileValue(
        rewriter, /*resultNumber=*/0, offsets, sizes);
    assert(succeeded(tiledResult) && "unexpected tiling failure");

    Value replacement = tiledResult->tiledValues[0];

    // generateResultTileValue produces a tensor. If the read_slice returns a
    // vector, extract it with vector.transfer_read.
    if (auto vectorType = dyn_cast<VectorType>(readSlice.getType())) {
      auto tensorType = cast<RankedTensorType>(replacement.getType());
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      SmallVector<Value> indices(tensorType.getRank(), zero);
      SmallVector<bool> inBounds(tensorType.getRank(), true);
      replacement = vector::TransferReadOp::create(
          rewriter, loc, vectorType, replacement, indices,
          /*padding=*/std::nullopt, inBounds);
    }

    rewriter.replaceOp(readSlice, replacement);
  }

  // Step 3: Erase the original producer if it has no other uses.
  if (params.producer->use_empty()) {
    rewriter.eraseOp(params.producer);
  }
}

} // namespace

//===---------------------------------------------------------------------===//
// Public API Specializations
//===---------------------------------------------------------------------===//

LogicalResult matchTilableProducer(RewriterBase &rewriter,
                                   PCF::GenericOp genericOp,
                                   ProducerFusionParams &params) {
  return matchTilableProducerImpl(rewriter, genericOp, params);
}

LogicalResult matchTilableProducer(RewriterBase &rewriter, PCF::LoopOp loopOp,
                                   ProducerFusionParams &params) {
  return matchTilableProducerImpl(rewriter, loopOp, params);
}

void fuseTilableProducer(RewriterBase &rewriter, PCF::GenericOp genericOp,
                         const ProducerFusionParams &params) {
  return fuseTilableProducerImpl(rewriter, genericOp, params);
}

void fuseTilableProducer(RewriterBase &rewriter, PCF::LoopOp loopOp,
                         const ProducerFusionParams &params) {
  return fuseTilableProducerImpl(rewriter, loopOp, params);
}

} // namespace mlir::iree_compiler::IREE::PCF
