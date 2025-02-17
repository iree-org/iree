// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SetEncoding.cpp -------------------------------------===//
// Sets the encoding for compute operations to allow execution of the
// operations in tiled layouts.
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-set-encoding"

namespace mlir::iree_compiler::DispatchCreation {
#define GEN_PASS_DEF_SETENCODINGPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

using IREE::Encoding::EncodingAttr;

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

Value setEncoding(OpBuilder &builder, Location loc, Value source,
                  EncodingAttr encodingAttr) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto resultType = RankedTensorType::get(
      sourceType.getShape(), sourceType.getElementType(), encodingAttr);
  return builder.create<IREE::Encoding::SetEncodingOp>(loc, resultType, source);
};

static Value unsetEncoding(OpBuilder &builder, Location loc, Value source,
                           SmallVector<OpFoldResult> sizes) {
  SmallVector<Value> dynamicSizesVec;
  SmallVector<int64_t> staticSizesVec;
  dispatchIndexOpFoldResults(sizes, dynamicSizesVec, staticSizesVec);

  auto sourceType = cast<RankedTensorType>(source.getType());
  auto unsetEncodingReturnType =
      RankedTensorType::get(sourceType.getShape(), sourceType.getElementType());
  return builder.create<IREE::Encoding::UnsetEncodingOp>(
      loc, unsetEncodingReturnType, source, dynamicSizesVec);
}

/// Given a LinalgOp and one of its OpOperands, return the element type,
/// inferring unsignedness from the body of the LinalgOp
static Type getContractionInputTypeWithSignedness(OpBuilder &builder,
                                                  linalg::LinalgOp linalgOp,
                                                  OpOperand *operand) {
  assert(linalg::isaContractionOpInterface(linalgOp));
  assert(operand->getOwner() == linalgOp.getOperation());
  auto elemType = getElementTypeOrSelf(operand->get().getType());
  // Infer if unsigned from body ops
  Value blockArg = linalgOp.getMatchingBlockArgument(operand);
  for (auto bodyCastOp : blockArg.getParentBlock()->getOps<arith::ExtUIOp>()) {
    if (bodyCastOp->getOperand(0) == blockArg) {
      return builder.getIntegerType(elemType.getIntOrFloatBitWidth(),
                                    /*isSigned=*/false);
    }
  }
  return elemType;
}

/// Returns true iff the linalgOp has a body like a regular matmul, i.e.
/// yield(add(out, mul(cast(in0), cast(in1))))
static bool hasMatmulLikeBody(linalg::LinalgOp linalgOp) {
  auto outBlockArg =
      linalgOp.getMatchingBlockArgument(linalgOp.getDpsInitOperand(0));
  auto yieldOp =
      dyn_cast<linalg::YieldOp>(outBlockArg.getParentBlock()->getTerminator());
  if (!yieldOp) {
    return false;
  }
  Operation *addOp = yieldOp->getOperand(0).getDefiningOp();
  if (!addOp || !isa<arith::AddIOp, arith::AddFOp>(addOp)) {
    return false;
  }
  Value addLhs = addOp->getOperand(0);
  Value addRhs = addOp->getOperand(1);
  Operation *addLhsOp = addLhs.getDefiningOp();
  Operation *addRhsOp = addRhs.getDefiningOp();
  if (!(addLhsOp && addRhs == outBlockArg) &&
      !(addRhsOp && addLhs == outBlockArg)) {
    return false;
  }
  Operation *mulOp = addLhsOp ? addLhsOp : addRhsOp;
  if (!isa<arith::MulFOp, arith::MulIOp>(mulOp)) {
    return false;
  }
  Value mulLhs = mulOp->getOperand(0);
  Value mulRhs = mulOp->getOperand(1);
  auto mulLhsOp = mulLhs.getDefiningOp<CastOpInterface>();
  auto mulRhsOp = mulRhs.getDefiningOp<CastOpInterface>();
  if (!isa<BlockArgument>(mulLhs) && !mulLhsOp && !isa<BlockArgument>(mulRhs) &&
      !mulRhsOp) {
    return false;
  }
  if ((mulLhsOp && !isa<BlockArgument>(mulLhsOp->getOperand(0))) ||
      (mulRhsOp && !isa<BlockArgument>(mulRhsOp->getOperand(0)))) {
    return false;
  }
  return true;
}

/// Not all contractions are supported by data tiling, so return true if:
///   1) linalgOp has contraction indexingMaps.
///   2) There are not more than one of each contraction dimension
///   3) There is and M or N dimension, and there is a K dimension
///   4) linalgOp has the same body as an ordinary int or float matmul
///
/// These restrictions are required because data tiling currently creates
/// an Mmt4DOp or BatchMmt4DOp on the packed inputs.
///
/// TODO(#16176): Loosen restrictions on contraction ops once data tiling
/// can support more cases.
static LogicalResult isSupportedContractionOp(PatternRewriter &rewriter,
                                              linalg::LinalgOp linalgOp) {
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "Expected isaContractionOpInterface");
  }
  auto cDims = linalg::inferContractionDims(linalgOp);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expected {|Batch|, |M|, |N|, |K|} <= 1");
  }
  if ((cDims->n.empty() && cDims->m.empty()) || cDims->k.empty()) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expected M or N dims and K dim to not be empty");
  }
  if (!hasMatmulLikeBody(linalgOp)) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expected op to have a matmul body, i.e. yield(add(out, "
                  "mul(cast(in0), cast(in1))))");
  }
  return success();
}

namespace {

class SetContractionOpEncoding final
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  explicit SetContractionOpEncoding(MLIRContext *ctx, int64_t factor)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx), padFactor(factor) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasPureTensorSemantics()) {
      return failure();
    }
    if (getCompilationInfo(linalgOp)) {
      return rewriter.notifyMatchFailure(
          linalgOp, "the op has preset compilation strategy, skip SetEncoding");
    }
    if (failed(isSupportedContractionOp(rewriter, linalgOp))) {
      return failure();
    }

    auto inputs = linalgOp.getDpsInputs();
    auto outputs = linalgOp.getDpsInits();

    auto hasEncoding = [](Value operand) -> bool {
      auto type = llvm::dyn_cast<RankedTensorType>(operand.getType());
      return type && type.getEncoding();
    };
    if (llvm::any_of(inputs, hasEncoding) ||
        llvm::any_of(outputs, hasEncoding)) {
      return failure();
    }
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    Value out = outputs[0];

    Type lhsElemType = getContractionInputTypeWithSignedness(
        rewriter, linalgOp, linalgOp.getDpsInputOperand(0));
    Type rhsElemType = getContractionInputTypeWithSignedness(
        rewriter, linalgOp, linalgOp.getDpsInputOperand(1));
    Type outElemType = getContractionInputTypeWithSignedness(
        rewriter, linalgOp, linalgOp.getDpsInitOperand(0));

    if (!lhsElemType || !rhsElemType || !outElemType) {
      return failure();
    }
    SmallVector<Type> elemTypes = {lhsElemType, rhsElemType, outElemType};

    auto narrowDim = IREE::Encoding::getMatmulNarrowDim(linalgOp, padFactor);

    Location loc = linalgOp.getLoc();
    SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();

    auto opType = IREE::Encoding::EncodingOpType::matmul;
    auto setEncodingWrapper = [&](Value src, int64_t operandIndex) -> Value {
      SmallVector<int64_t> roundDimsTo(3, padFactor);
      if (narrowDim.isM()) {
        roundDimsTo[0] = llvm::PowerOf2Ceil(narrowDim.size);
      }
      if (narrowDim.isN()) {
        roundDimsTo[1] = llvm::PowerOf2Ceil(narrowDim.size);
      }
      auto encoding = EncodingAttr::get(linalgOp.getContext(), operandIndex,
                                        opType, elemTypes, maps,
                                        /*bcastMap=*/std::nullopt, roundDimsTo);
      return setEncoding(rewriter, loc, src, encoding);
    };
    Value encodedLhs = setEncodingWrapper(lhs, IREE::Encoding::MATMUL_LHS);
    Value encodedRhs = setEncodingWrapper(rhs, IREE::Encoding::MATMUL_RHS);
    Value encodedOut = setEncodingWrapper(out, IREE::Encoding::MATMUL_RESULT);
    Value opTiled = clone(rewriter, linalgOp, encodedOut.getType(),
                          ValueRange{encodedLhs, encodedRhs, encodedOut})
                        ->getResult(0);

    // Sizes are computed by original output size.
    SmallVector<OpFoldResult> outSizes =
        tensor::getMixedSizes(rewriter, loc, out);
    Value result = unsetEncoding(rewriter, loc, opTiled, outSizes);

    rewriter.replaceOp(linalgOp, result);
    return success();
  }

private:
  int64_t padFactor = 32;
};

/// Pattern to fold a `linalg.fill` -> `iree_encoding.set_encoding`
/// operation into a `linalg.fill` of the encoded type.
struct FoldFillWithSetEncoding final
    : OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = encodingOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    // Create a new fill op, with outs being defined by a new `tensor.empty` op.
    RankedTensorType encodingType = encodingOp.getResultType();
    Location loc = fillOp.getLoc();
    SmallVector<OpFoldResult> dimValues =
        tensor::getMixedSizes(rewriter, loc, fillOp.getOutputs()[0]);
    auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, dimValues, encodingType.getElementType(),
        encodingType.getEncoding());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(encodingOp, fillOp.getInputs(),
                                                ValueRange{newEmptyOp});
    return success();
  }
};

struct SetEncodingPass final : impl::SetEncodingPassBase<SetEncodingPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SetContractionOpEncoding>(context, padFactor);
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    patterns.add<FoldFillWithSetEncoding>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    GreedyRewriteConfig config;
    config.cseConstants = false;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
