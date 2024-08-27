// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
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

#define DEBUG_TYPE "iree-dispatch-creation-hoist-encoding-ops"

namespace mlir::iree_compiler::DispatchCreation {
#define GEN_PASS_DEF_HOISTENCODINGOPSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

static AffineMap getBcastMapOrIdentity(RewriterBase &rewriter,
                                       RankedTensorType encodedType) {
  auto encoding = cast<IREE::Encoding::EncodingAttr>(encodedType.getEncoding());
  AffineMapAttr bcastMapAttr = encoding.getBcastMap();
  return bcastMapAttr ? bcastMapAttr.getAffineMap()
                      : rewriter.getMultiDimIdentityMap(encodedType.getRank());
}

/// Bubbles a SetEncodingOp up through a linalg::GenericOp. The `genericOp`
/// must:
///  1. Have a single result.
///  2. Have single use.
///  3. Have all parallel iterators.
///  4. Have an identity output indexing map.
///  5. Have a tensor.empty init operand.
///  6. Have as many indexing map dims as there are results in the encoding's
///     bcast_map.
///
/// This function creates SetEncoding ops on all of the inputs to the
/// `genericOp`, and replaces the op with an encoded version. If any of
/// the above conditions are false, then it returns failure.
///
/// Note: The bcast_map on the set_encoding op must be identity or absent.
///       The implementation should work for cases where it is not, but it is
///       unexpected in IREE compilation to find such cases, and it will not
///       be well tested.
static LogicalResult
bubbleUpSetEncodingThroughGenericOp(RewriterBase &rewriter,
                                    IREE::Encoding::SetEncodingOp encodingOp,
                                    linalg::GenericOp genericOp) {
  if (!genericOp->hasOneUse()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "genericOp must have one use");
  }
  if (genericOp.getNumDpsInits() != 1) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "genericOp must have a single init");
  }
  if (genericOp.getNumReductionLoops() != 0) {
    return rewriter.notifyMatchFailure(
        genericOp, "genericOp must have all parallel loops");
  }
  if (!genericOp.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "init operand must be tensor.empty");
  }
  AffineMap outputMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (!outputMap.isIdentity()) {
    return rewriter.notifyMatchFailure(genericOp, "output map not identity");
  }

  RankedTensorType encodedType = encodingOp.getResultType();
  AffineMap bcastMap = getBcastMapOrIdentity(rewriter, encodedType);
  if (!bcastMap.isIdentity()) {
    return rewriter.notifyMatchFailure(genericOp, "bcast_map map not identity");
  }
  if (outputMap.getNumDims() != bcastMap.getNumResults()) {
    return rewriter.notifyMatchFailure(
        genericOp, "output map numDims do not match bcast_map numResults");
  }

  // Set encodings on each input
  Location loc = genericOp->getLoc();
  SmallVector<Value> encodedOperands;
  auto encoding = cast<IREE::Encoding::EncodingAttr>(encodedType.getEncoding());
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
  auto setEncoding = cast<IREE::Encoding::SetEncodingOp>(operand.getOwner());
  auto producer = operand.get().getDefiningOp<linalg::GenericOp>();
  if (!producer) {
    return failure();
  }
  // Only bubble through dequantization ops and broadcasting ops for now.
  if (!IREE::LinalgExt::isBitExtendOp(producer) &&
      !IREE::LinalgExt::isBroadcastingOp(producer)) {
    return failure();
  }
  return bubbleUpSetEncodingThroughGenericOp(rewriter, setEncoding, producer);
}

namespace {
/// Pass declaration.
struct HoistEncodingOpsPass
    : public impl::HoistEncodingOpsPassBase<HoistEncodingOpsPass> {
  using Base::Base;
  void runOnOperation() override;
};

/// Pattern to bubble SetEncoding ops upwards through producers. This pattern
/// runs until bubbling is not possible, or until the SetEncoding op is outside
/// of a dispatch.
struct BubbleUpSetEncodingOp
    : public OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using OpRewritePattern<IREE::Encoding::SetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    if (IREE::Flow::isNonNullAndOutsideDispatch(encodingOp)) {
      return failure();
    }
    // Fail if the encodingOp is not in the same dispatch as its producer.
    Operation *producer = encodingOp.getSource().getDefiningOp();
    if (!producer) {
      return failure();
    }
    auto dispatch = producer->getParentOfType<IREE::Flow::DispatchRegionOp>();
    if (!dispatch ||
        dispatch !=
            encodingOp->getParentOfType<IREE::Flow::DispatchRegionOp>()) {
      return failure();
    }

    return bubbleUpSetEncoding(rewriter, encodingOp->getOpOperand(0));
  }
};

} // namespace

/// Create dispatch.region Ops based on a fusion heuristic.
void HoistEncodingOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  RewritePatternSet bubblingPatterns(ctx);
  bubblingPatterns.insert<BubbleUpSetEncodingOp>(ctx);
  if (failed(
          applyPatternsAndFoldGreedily(funcOp, std::move(bubblingPatterns)))) {
    return signalPassFailure();
  }

  SmallVector<IREE::Encoding::SetEncodingOp> candidates;
  funcOp->walk([&](IREE::Encoding::SetEncodingOp setEncodingOp) {
    if (setEncodingOp->getParentOfType<IREE::Flow::DispatchRegionOp>()) {
      candidates.push_back(setEncodingOp);
    }
  });
  IRRewriter rewriter(ctx);
  for (auto setEncodingOp : candidates) {
    if (failed(IREE::Flow::hoistOutOfDispatch(rewriter, setEncodingOp))) {
      return signalPassFailure();
    }
  }

  RewritePatternSet cleanPatterns(ctx);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(cleanPatterns);
  IREE::Flow::DispatchRegionOp::getCanonicalizationPatterns(cleanPatterns, ctx);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(cleanPatterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
