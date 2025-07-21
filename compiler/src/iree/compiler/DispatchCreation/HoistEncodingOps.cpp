// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <queue>

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
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
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::DispatchCreation {
#define GEN_PASS_DEF_HOISTENCODINGOPSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

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
  auto encoding = cast<IREE::Encoding::EncodingAttr>(encodedType.getEncoding());
  AffineMap lastMap = encoding.getLastMapForOperandIndex();
  if (outputMap.getNumDims() != lastMap.getNumResults()) {
    return rewriter.notifyMatchFailure(
        genericOp,
        "output map numDims do not match last encoding map numResults");
  }

  // Set encodings on each input
  Location loc = genericOp->getLoc();
  SmallVector<Value> encodedOperands;
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    // Append the operand's indexing map to the encoding's user indexing maps.
    AffineMap operandMap = genericOp.getMatchingIndexingMap(operand);

    // Create new encoding and set encoding on the operand.
    IREE::Encoding::EncodingAttr newEncoding =
        encoding.cloneWithNewOperandIndexingMap(operandMap);
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

/// Returns true if the op is hoistable outside dispatches, which indicates that
/// the ops can be either mappable to Flow ops or get hoisted to globals.
static bool isHoistableOp(Operation *op) {
  if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
    ArrayRef<int64_t> srcShape = sliceOp.getSourceType().getShape();
    return IREE::Flow::isOffsetSizeAndStrideMappableToFlow(offsets, sizes,
                                                           strides, srcShape);
  }
  // ConstExprHoistingPolicy has an assumption that any root op is not hoistable
  // because they are already hoisted. This is not the case when the parent op
  // is a constant-like op, so we have a special rule here.
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    return true;
  }
  return false;
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

/// Pattern to sink UnsetEncoding ops down through consumers.
struct SinkUnsetEncodingOp
    : public OpRewritePattern<IREE::Encoding::UnsetEncodingOp> {
  using OpRewritePattern<IREE::Encoding::UnsetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Encoding::UnsetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    if (!encodingOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(encodingOp, "has multiple uses");
    }
    OpOperand *consumerOperand = &(*encodingOp->getUses().begin());
    Operation *consumer = consumerOperand->getOwner();
    if (IREE::Flow::isNonNullAndOutsideDispatch(encodingOp) ||
        IREE::Flow::isNonNullAndOutsideDispatch(consumer)) {
      return rewriter.notifyMatchFailure(
          encodingOp, "expected that both operations are inside dispatch");
    }

    auto propagationAttrInterface =
        dyn_cast<IREE::Encoding::EncodingPropagationAttrInterface>(
            encodingOp.getSourceType().getEncoding());
    if (!propagationAttrInterface ||
        !propagationAttrInterface.isPropagableDown(consumerOperand)) {
      return rewriter.notifyMatchFailure(
          encodingOp,
          "the propagation attribute interface isn't defined or the "
          "target isn't propagable");
    }
    // Get the encoding attributes for the operands and results of the
    // operation.
    FailureOr<IREE::Encoding::PropagationEncoding> propagationEncodings =
        propagationAttrInterface.generateSinkingEncodings(consumerOperand);
    if (failed(propagationEncodings)) {
      return rewriter.notifyMatchFailure(encodingOp,
                                         "not able to determine propagation "
                                         "attributes for operands and results");
    }
    auto propagationResult =
        dyn_cast<IREE::Encoding::EncodingPropagationOpInterface>(consumer);
    if (!propagationResult) {
      return rewriter.notifyMatchFailure(
          encodingOp, "encoding propagation op interface isn't defined");
    }
    // Propagate the set encoding and generate the new encoding operations.
    rewriter.setInsertionPointAfter(encodingOp);
    FailureOr<IREE::Encoding::PropagationResult> maybeResult =
        propagationResult.propagateEncoding(
            rewriter, *propagationEncodings,
            cast<OpResult>(encodingOp.getResult()));
    if (failed(maybeResult)) {
      return rewriter.notifyMatchFailure(
          encodingOp, "not able to propagate encodings and find replacement");
    }
    rewriter.replaceOp(consumer, maybeResult->replacements);
    return success();
  }
};

} // namespace

/// Create dispatch.region Ops based on a fusion heuristic.
void HoistEncodingOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ModuleOp moduleOp = getOperation();

  RewritePatternSet bubblingPatterns(ctx);
  bubblingPatterns.insert<BubbleUpSetEncodingOp>(ctx);
  bubblingPatterns.insert<SinkUnsetEncodingOp>(ctx);
  GreedyRewriteConfig config;
  config.enableConstantCSE(false);
  if (failed(applyPatternsGreedily(moduleOp, std::move(bubblingPatterns),
                                   config))) {
    return signalPassFailure();
  }

  const auto &constExprs = getAnalysis<IREE::Util::ConstExprAnalysis>();
  IREE::Util::ConstExprHoistingPolicy policy(constExprs, /*threshold=*/0);
  policy.initialize();

  // Each element indicates ops that are expected to be hoisted. It is valid to
  // follow the order in the vector to hoist the ops. The last operation is
  // expected to be the corresponding SetEncoding op.
  SmallVector<SmallVector<Operation *>> candidates;
  moduleOp->walk([&](IREE::Encoding::SetEncodingOp setEncodingOp) {
    if (!setEncodingOp->getParentOfType<IREE::Flow::DispatchRegionOp>()) {
      return;
    }
    // Avoid hoisting set encodings on scalar tensors as they will likely not
    // resolve into anything other than scalar values and are just broadcast
    // ops.
    auto inputType =
        cast<RankedTensorType>(setEncodingOp.getSource().getType());
    if (inputType.getRank() == 0) {
      return;
    }
    // Avoid hoisting set encodings that are using the padding encodings.
    Attribute encoding = setEncodingOp.getResultType().getEncoding();
    if (isa_and_nonnull<IREE::Encoding::PaddingAttr>(encoding)) {
      return;
    }

    bool isHoistable = true;
    SetVector<Operation *> opsWithinDispatch, seen;
    std::queue<Operation *> worklist;
    worklist.push(setEncodingOp);
    seen.insert(setEncodingOp);
    while (!worklist.empty()) {
      Operation *op = worklist.front();
      worklist.pop();
      opsWithinDispatch.insert(op);
      for (auto input : op->getOperands()) {
        auto inputOp = input.getDefiningOp();
        if (inputOp &&
            inputOp->getParentOfType<IREE::Flow::DispatchRegionOp>() &&
            !seen.contains(inputOp)) {
          if (!isHoistableOp(inputOp)) {
            isHoistable = false;
          }
          worklist.push(inputOp);
          seen.insert(inputOp);
        }
      }
    }
    if (isHoistable) {
      candidates.push_back(llvm::to_vector(llvm::reverse(opsWithinDispatch)));
      return;
    }

    // The ops are hoistable if they are const-exprs.
    const IREE::Util::ConstExprAnalysis::ConstValueInfo *constInfo =
        constExprs.lookup(setEncodingOp.getSource());
    if (!constInfo) {
      LDBG("Non-hoistable op (failed to get constInfo): " << setEncodingOp);
      return;
    }
    if (policy.getDecision(constInfo)->getOutcome() ==
        IREE::Util::ConstExprHoistingPolicy::ENABLE_HOIST) {
      candidates.push_back(llvm::to_vector(llvm::reverse(opsWithinDispatch)));
      return;
    }
    LDBG("Non-hoistable op: " << setEncodingOp);
  });

  IRRewriter rewriter(ctx);
  for (ArrayRef<Operation *> hoistableOps : candidates) {
    LDBG("Hoisting the ops for " << *hoistableOps.back());
    for (Operation *op : hoistableOps) {
      if (failed(IREE::Flow::hoistOutOfDispatch(rewriter, op))) {
        op->emitOpError("failed to hoist the op out of dispatch");
        return signalPassFailure();
      }
    }
  }

  RewritePatternSet cleanPatterns(ctx);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(cleanPatterns);
  IREE::Flow::DispatchRegionOp::getCanonicalizationPatterns(cleanPatterns, ctx);
  if (failed(
          applyPatternsGreedily(moduleOp, std::move(cleanPatterns), config))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
