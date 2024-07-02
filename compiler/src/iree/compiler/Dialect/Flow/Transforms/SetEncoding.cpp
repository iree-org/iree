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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-set-encoding"

namespace mlir::iree_compiler::IREE::Flow {
#define GEN_PASS_DEF_SETENCODINGPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

constexpr int64_t kPadSize = 16;

struct MatmulNarrowSizes {
  std::optional<int64_t> M, N;
};

// Returns the minimum of static sizes of the M/N-dimensions in the types of the
// Ouput.
static MatmulNarrowSizes getMatmulNarrowSizes(ShapedType outType,
                                              linalg::LinalgOp linalgOp) {
  linalg::ContractionDimensions cDims =
      linalg::inferContractionDims(linalgOp).value();
  auto map = linalgOp.getIndexingMapsArray().back();
  auto getOutputSizeAtDimPos = [&](unsigned dimPos) -> int64_t {
    return outType.getDimSize(
        map.getResultPosition(getAffineDimExpr(dimPos, linalgOp->getContext()))
            .value());
  };
  // M or N can be empty instead of having an explicit dim size of 1 for matvec
  // and vecmat, so set to 1 if empty.
  int64_t M = cDims.m.empty() ? 1 : getOutputSizeAtDimPos(cDims.m[0]);
  int64_t N = cDims.n.empty() ? 1 : getOutputSizeAtDimPos(cDims.n[0]);

  MatmulNarrowSizes narrow;
  // Threshold below which a M/N size is considered "narrow", making it
  // eligible for a narrow tile size during materialization. This value should
  // be at least as large as the actual M/N tile sizes that we choose on any
  // target in CPUMaterializeEncodingPass. If it is smaller, we will miss
  // opportunities to select optimized narrow tiles for narrow matmuls.
  // If it is larger, everything will work fine, but the IR will be a bit more
  // verbose as more narrow_matmul_{M,N} optional parameters will be specified.
  const int64_t kNarrowThreshold = 16;
  if (!ShapedType::isDynamic(M) && M < kNarrowThreshold) {
    narrow.M = M;
  }
  if (!ShapedType::isDynamic(N) && N < kNarrowThreshold) {
    narrow.N = N;
  }

  // Only pick 1 if both are present
  if (narrow.M && narrow.N) {
    if (*narrow.M <= *narrow.N) {
      narrow.N.reset();
    } else {
      narrow.M.reset();
    }
  }

  return narrow;
}

static bool isDefiningExtendingCastOp(linalg::LinalgOp op) {
  auto isExtending = [](CastOpInterface castOp) -> bool {
    auto inElemType = getElementTypeOrSelf(castOp->getOperandTypes()[0]);
    auto outElemType = getElementTypeOrSelf(castOp->getResultTypes()[0]);
    return inElemType.getIntOrFloatBitWidth() <
           outElemType.getIntOrFloatBitWidth();
  };

  if (op.getBlock()->getOperations().size() != 2 || !isElementwise(op)) {
    return false;
  }
  auto yieldOp = cast<linalg::YieldOp>(op.getBlock()->getTerminator());
  auto castOp = yieldOp->getOperand(0).getDefiningOp<CastOpInterface>();
  if (!castOp) {
    return false;
  }
  Value castIn = castOp->getOperand(0);
  if (isa<BlockArgument>(castIn) &&
      cast<BlockArgument>(castIn).getArgNumber() != 0) {
    return false;
  }
  return isExtending(castOp);
}

/// Given a LinalgOp and one of its OpOperands, return the element type,
/// inferring unsignedness from the body of the LinalgOp
static Type getContractionInputTypeWithSignedness(OpBuilder &builder,
                                                  linalg::LinalgOp linalgOp,
                                                  OpOperand *operand) {
  assert(linalg::isaContractionOpInterface(linalgOp));
  auto elemType = getElementTypeOrSelf(operand->get().getType());
  // Infer if unsigned from body ops
  Value blockArg = linalgOp.getMatchingBlockArgument(operand);
  for (auto bodyCastOp : blockArg.getParentBlock()->getOps<arith::ExtUIOp>()) {
    if (bodyCastOp->getOperand(0) == blockArg) {
      return builder.getIntegerType(elemType.getIntOrFloatBitWidth(),
                                    /*isSigned=*/false);
    }
  }
  auto producer = operand->get().getDefiningOp<linalg::LinalgOp>();
  if (isDequantizationLikeOp(operand->get().getDefiningOp()) && producer &&
      isDefiningExtendingCastOp(producer)) {
    return getElementTypeOrSelf(
        producer.getDpsInputOperand(0)->get().getType());
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
  auto addOp = yieldOp->getOperand(0).getDefiningOp();
  if (!addOp || !isa<arith::AddIOp, arith::AddFOp>(addOp)) {
    return false;
  }
  auto addLhs = addOp->getOperand(0);
  auto addRhs = addOp->getOperand(1);
  auto addLhsOp = addLhs.getDefiningOp();
  auto addRhsOp = addRhs.getDefiningOp();
  if (!(addLhsOp && addRhs == outBlockArg) &&
      !(addRhsOp && addLhs == outBlockArg)) {
    return false;
  }
  Operation *mulOp = addLhsOp ? addLhsOp : addRhsOp;
  if (!isa<arith::MulFOp, arith::MulIOp>(mulOp)) {
    return false;
  }
  auto mulLhs = mulOp->getOperand(0);
  auto mulRhs = mulOp->getOperand(1);
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
static LogicalResult isSupportedContractionOp(RewriterBase &rewriter,
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

static Value unsetEncodingAndExtractSlice(OpBuilder &builder, Location loc,
                                          Value source,
                                          SmallVector<OpFoldResult> sizes) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto unsetEncodingReturnType =
      RankedTensorType::get(sourceType.getShape(), sourceType.getElementType());
  auto unsetEncoding = builder
                           .create<IREE::Encoding::UnsetEncodingOp>(
                               loc, unsetEncodingReturnType, source)
                           .getResult();
  auto rank = sourceType.getRank();
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  return builder.create<tensor::ExtractSliceOp>(loc, unsetEncoding, offsets,
                                                sizes, strides);
}

static void setEncodingsOnContractions(RewriterBase &rewriter,
                                       DispatchRegionOp regionOp) {
  SmallVector<linalg::LinalgOp> candidates;
  regionOp.getBody().walk([&](linalg::LinalgOp op) {
    if (failed(isSupportedContractionOp(rewriter, op))) {
      return;
    }
    candidates.push_back(op);
  });

  for (auto op : candidates) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    Location loc = op.getLoc();
    auto inputs = op.getDpsInputs();
    auto outputs = op.getDpsInits();
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    Value out = outputs[0];

    Type lhsElemType = getContractionInputTypeWithSignedness(
        rewriter, op, op.getDpsInputOperand(0));
    Type rhsElemType = getContractionInputTypeWithSignedness(
        rewriter, op, op.getDpsInputOperand(1));
    Type outElemType = getContractionInputTypeWithSignedness(
        rewriter, op, op.getDpsInitOperand(0));
    SmallVector<Type> elemTypes = {lhsElemType, rhsElemType, outElemType};

    MatmulNarrowSizes narrowSizes =
        getMatmulNarrowSizes(cast<ShapedType>(out.getType()), op);

    SmallVector<AffineMap> maps = op.getIndexingMapsArray();

    auto setEncodingWrapper = [&](Value src, int64_t operandIndex) -> Value {
      auto srcType = cast<RankedTensorType>(src.getType());
      AffineMap bcastMap = rewriter.getMultiDimIdentityMap(srcType.getRank());
      // Set pad size for M, N, and K dimensions.
      SmallVector<int64_t> roundDimsTo(3, kPadSize);
      auto encodingAttr = Encoding::EncodingAttr::get(
          op.getContext(), operandIndex, elemTypes, srcType, narrowSizes.M,
          narrowSizes.N, maps, bcastMap, roundDimsTo);
      auto resType = RankedTensorType::get(
          srcType.getShape(), srcType.getElementType(), encodingAttr);
      return rewriter.create<IREE::Encoding::SetEncodingOp>(loc, resType, src);
    };

    Value encodedLhs = setEncodingWrapper(lhs, IREE::Encoding::MATMUL_LHS);
    Value encodedRhs = setEncodingWrapper(rhs, IREE::Encoding::MATMUL_RHS);
    Value encodedOut = setEncodingWrapper(out, IREE::Encoding::MATMUL_RESULT);
    Value opTiled = clone(rewriter, op, encodedOut.getType(),
                          ValueRange{encodedLhs, encodedRhs, encodedOut})
                        ->getResult(0);
    SmallVector<OpFoldResult> outSizes =
        tensor::getMixedSizes(rewriter, loc, out);
    Value result =
        unsetEncodingAndExtractSlice(rewriter, loc, opTiled, outSizes);
    rewriter.replaceOp(op, result);
  }
}

namespace {
/// Pass declaration.
struct SetEncodingPass
    : public IREE::Flow::impl::SetEncodingPassBase<SetEncodingPass> {
  using IREE::Flow::impl::SetEncodingPassBase<
      SetEncodingPass>::SetEncodingPassBase;
  void runOnOperation() override;
};

/// Pattern to fold a `linalg.fill` -> `iree_encoding.set_encoding`
/// operation into a `linalg.fill` of the encoded type.
struct FoldFillWithSetEncoding
    : public OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using OpRewritePattern<IREE::Encoding::SetEncodingOp>::OpRewritePattern;

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
} // namespace

/// Create dispatch.region Ops based on a fusion heuristic.
void SetEncodingPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(ctx);
  funcOp->walk([&](DispatchRegionOp regionOp) {
    setEncodingsOnContractions(rewriter, regionOp);
  });

  RewritePatternSet patterns(ctx);
  linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
  patterns.insert<FoldFillWithSetEncoding>(ctx);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::IREE::Flow
