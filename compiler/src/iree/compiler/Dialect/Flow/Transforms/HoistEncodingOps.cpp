// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-hoist-encoding-ops"

namespace mlir::iree_compiler::IREE::Flow {
#define GEN_PASS_DEF_HOISTENCODINGOPSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

static LogicalResult
bubbleUpSetEncodingThroughBroadcastOp(RewriterBase &rewriter,
                                      IREE::Encoding::SetEncodingOp encodingOp,
                                      linalg::BroadcastOp broadcastOp) {
  if (!broadcastOp->hasOneUse()) {
    return failure();
  }
  RankedTensorType encodedType = encodingOp.getResultType();
  auto encoding = cast<IREE::Encoding::EncodingAttr>(encodedType.getEncoding());

  // Create new encoding and set encoding on broadcast input.
  AffineMap bcastMap = encoding.getBcastMap().getAffineMap();
  bcastMap = bcastMap.dropResults(broadcastOp.getDimensions());
  Value input = broadcastOp.getInput();
  auto newEncoding = encoding.clone(bcastMap);
  auto inputType = cast<RankedTensorType>(input.getType());
  auto resType = RankedTensorType::get(inputType.getShape(),
                                       inputType.getElementType(), newEncoding);
  Location loc = broadcastOp->getLoc();
  IREE::Encoding::SetEncodingOp newSetEncoding =
      rewriter.create<IREE::Encoding::SetEncodingOp>(loc, resType, input);

  // Create encoded broadcast op.
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, encodingOp.getSource());
  Value encodedBCastInit = rewriter.create<tensor::EmptyOp>(
      loc, mixedSizes, encodedType.getElementType(), encoding);
  SmallVector<Value> encodedBCastOperands = {newSetEncoding.getResult(),
                                             encodedBCastInit};
  auto encodedBroadcast = clone(
      rewriter, broadcastOp, encodingOp.getResultType(), encodedBCastOperands);

  rewriter.replaceOp(encodingOp, encodedBroadcast);
  return success();
}

static LogicalResult
bubbleUpSetEncodingThroughGenericOp(RewriterBase &rewriter,
                                    IREE::Encoding::SetEncodingOp encodingOp,
                                    linalg::GenericOp genericOp) {
  if (!genericOp->hasOneUse()) {
    return failure();
  }
  if (genericOp.getNumDpsInits() != 1) {
    return failure();
  }
  if (genericOp.getNumReductionLoops() != 0) {
    return failure();
  }
  AffineMap outputMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (!outputMap.isIdentity()) {
    return rewriter.notifyMatchFailure(genericOp, "output map not identity");
  }

  RankedTensorType encodedType = encodingOp.getResultType();
  auto encoding = cast<IREE::Encoding::EncodingAttr>(encodedType.getEncoding());
  AffineMap bcastMap = encoding.getBcastMap().getAffineMap();
  if (outputMap.getNumDims() != bcastMap.getNumDims()) {
    return rewriter.notifyMatchFailure(
        genericOp, "output map dims do not match bcast_map dims");
  }

  // Set encodings on each input
  Location loc = genericOp->getLoc();
  SmallVector<Value> encodedOperands;
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    // Compute the new bcastMap from the operand's indexing map.
    AffineMap operandMap = genericOp.getMatchingIndexingMap(operand);
    AffineMap newBcastMap = operandMap.compose(bcastMap);

    // Create new encoding and set encoding on the operand.
    auto newEncoding = encoding.clone(newBcastMap);
    auto operandType = cast<RankedTensorType>(operand->get().getType());
    auto resType = RankedTensorType::get(
        operandType.getShape(), operandType.getElementType(), newEncoding);
    Value encodedInput = rewriter.create<IREE::Encoding::SetEncodingOp>(
        loc, resType, operand->get());
    encodedOperands.push_back(encodedInput);
  }

  // Create encoded generic op.
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, encodingOp.getSource());
  Value encodedInit = rewriter.create<tensor::EmptyOp>(
      loc, mixedSizes, encodedType.getElementType(), encoding);
  encodedOperands.push_back(encodedInit);
  auto encodedGenericOp =
      clone(rewriter, genericOp, encodingOp.getResultType(), encodedOperands);

  rewriter.replaceOp(encodingOp, encodedGenericOp);
  return success();
}

static LogicalResult bubbleUpSetEncoding(RewriterBase &rewriter,
                                         OpOperand &operand) {
  auto setEncoding = cast<Encoding::SetEncodingOp>(operand.getOwner());
  Operation *producer = operand.get().getDefiningOp();
  if (!producer) {
    return failure();
  }

  // Only bubble through dequantization ops and broadcasting ops for now.
  if (!isDequantizationLikeOp(producer) && !isBroadcastingOp(producer)) {
    return failure();
  }

  return TypeSwitch<Operation *, LogicalResult>(producer)
      .Case<linalg::GenericOp>([&](linalg::GenericOp genericOp) {
        return bubbleUpSetEncodingThroughGenericOp(rewriter, setEncoding,
                                                   genericOp);
      })
      .Case<linalg::BroadcastOp>([&](linalg::BroadcastOp broadcastOp) {
        return bubbleUpSetEncodingThroughBroadcastOp(rewriter, setEncoding,
                                                     broadcastOp);
      })
      .Default([](Operation *op) { return failure(); });
}

namespace {
/// Pass declaration.
struct HoistEncodingOpsPass
    : public IREE::Flow::impl::HoistEncodingOpsPassBase<HoistEncodingOpsPass> {
  using IREE::Flow::impl::HoistEncodingOpsPassBase<
      HoistEncodingOpsPass>::HoistEncodingOpsPassBase;
  void runOnOperation() override;
};

/// Pattern to bubble SetEncoding ops upwards through producers. This pattern
/// runs until bubbling is not possible, or until the SetEncoding op is outside
/// of a dispatch.
struct HoistSetEncodingOp
    : public OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using OpRewritePattern<IREE::Encoding::SetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    if (isNonNullAndOutsideDispatch(encodingOp)) {
      return failure();
    }
    // Try to hoist the SetEncodingOp out of the dispatch region.
    if (succeeded(hoistOutOfDispatch(rewriter, encodingOp))) {
      return success();
    }

    // Otherwise, try to bubble the SetEncodingOp past its producer op.
    return bubbleUpSetEncoding(rewriter, encodingOp->getOpOperand(0));
  }
};

/// Pattern to bubble SetEncoding ops upwards through producers. This pattern
/// runs until bubbling is not possible, or until the SetEncoding op is outside
/// of a dispatch.
struct HoistUnsetEncodingOp
    : public OpRewritePattern<IREE::Encoding::UnsetEncodingOp> {
  using OpRewritePattern<IREE::Encoding::UnsetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Encoding::UnsetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    if (isNonNullAndOutsideDispatch(encodingOp)) {
      return failure();
    }
    if (!encodingOp->hasOneUse()) {
      return failure();
    }
    // First check for extract_slice op, and hoist the extract_slice.
    SmallVector<Operation *> users(encodingOp->getUsers());
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(users.front());
    if (!extractSliceOp) {
      return hoistOutOfDispatch(rewriter, encodingOp);
    }
    if (!extractSliceOp->hasOneUse()) {
      return failure();
    }
    if (failed(hoistOutOfDispatch(rewriter, extractSliceOp))) {
      return failure();
    }
    return hoistOutOfDispatch(rewriter, encodingOp);
  }
};
} // namespace

/// Create dispatch.region Ops based on a fusion heuristic.
void HoistEncodingOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  IRRewriter rewriter(ctx);

  RewritePatternSet patterns(ctx);
  patterns.insert<HoistSetEncodingOp>(ctx);
  patterns.insert<HoistUnsetEncodingOp>(ctx);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::IREE::Flow
