// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SetEncoding.cpp -------------------------------------===//
// Sets the encoding for compute operations to allow execution of the
// operations in tiled layouts.
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir::iree_compiler::GlobalOptimization {

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

/// Pads `value` enough for any actual tile sizes that could result from
/// materialization of `encodingAttr`.
static Value pad(OpBuilder &builder, Location loc, Value source,
                 IREE::LinalgExt::EncodingAttr encodingAttr) {
  RankedTensorType sourceType = source.getType().cast<RankedTensorType>();
  Type elemType = sourceType.getElementType();
  size_t rank = sourceType.getRank();
  RankedTensorType tensorTypeWithEncoding =
      RankedTensorType::get(sourceType.getShape(), elemType, encodingAttr);
  SmallVector<OpFoldResult> lowPad(rank, builder.getIndexAttr(0));
  SmallVector<Type> resultTypes(rank, builder.getIndexType());

  ValueRange encodingPaddingSizes =
      builder
          .create<IREE::LinalgExt::UpperBoundTileSizeOp>(
              loc, resultTypes, TypeAttr::get(tensorTypeWithEncoding))
          .getResults();
  SmallVector<OpFoldResult> highPad(rank);
  AffineExpr tileExpr, shapeExpr;
  bindSymbols(builder.getContext(), tileExpr, shapeExpr);
  AffineExpr highPadExpr = shapeExpr.ceilDiv(tileExpr) * tileExpr - shapeExpr;
  for (size_t i = 0; i < rank; ++i) {
    highPad[i] = affine::makeComposedFoldedAffineApply(
        builder, loc, highPadExpr,
        getAsOpFoldResult({encodingPaddingSizes[i],
                           builder.create<tensor::DimOp>(loc, source, i)}));
  }

  Value zero = builder.create<arith::ConstantOp>(loc, elemType,
                                                 builder.getZeroAttr(elemType));
  return builder.create<tensor::PadOp>(loc, /*resultType=*/nullptr, source,
                                       lowPad, highPad, zero);
}

Value setEncoding(OpBuilder &builder, Location loc, Value source,
                  IREE::LinalgExt::EncodingAttr encodingAttr) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
      sourceType.getShape(), sourceType.getElementType(), encodingAttr);
  return builder.create<IREE::LinalgExt::SetEncodingOp>(loc, resultType,
                                                        source);
};

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
  return narrow;
}

static IREE::LinalgExt::EncodingAttr
makeEncoding(OpBuilder &builder, IREE::LinalgExt::EncodingRole role,
             TypeRange operandTypes, Type originalType,
             MatmulNarrowSizes narrow, ArrayAttr indexingMaps) {
  auto *context = builder.getContext();
  auto roleAttr = IREE::LinalgExt::EncodingRoleAttr::get(context, role);
  SmallVector<Attribute> elemTypeAttrs =
      llvm::map_to_vector(operandTypes, [](auto t) {
        return TypeAttr::get(t.template cast<ShapedType>().getElementType())
            .template cast<Attribute>();
      });
  auto operandElemTypesAttr = ArrayAttr::get(context, elemTypeAttrs);
  auto originalTypeAttr =
      originalType ? TypeAttr::get(originalType) : TypeAttr{};
  auto getAttr = [&](std::optional<int64_t> x) {
    return x ? builder.getIndexAttr(*x) : IntegerAttr();
  };
  return IREE::LinalgExt::EncodingAttr::get(
      context, roleAttr, operandElemTypesAttr, originalTypeAttr,
      getAttr(narrow.M), getAttr(narrow.N), indexingMaps);
}

// Creates a linalg::GenericOp that performs an element-wise cast of the same
// type as performed in `castOp`, and returns the result enceoded with
// `encodingAttr`. The element type of `encoded` is expected to be the same as
// the element type of the input to `castOp`, which can be a CastOpInterface op
// on a tensor or single element.
static Value castEncodedResult(OpBuilder &builder, Location loc, Value encoded,
                               CastOpInterface castOp,
                               IREE::LinalgExt::EncodingAttr encodingAttr) {
  auto genericOp = castOp->getParentOfType<linalg::GenericOp>();
  NamedAttrList castAttrs = genericOp
                                ? linalg::getPrunedAttributeList(genericOp)
                                : castOp->getAttrs();
  return createGenericElementwiseCastOp(builder, loc, encoded, castOp,
                                        castAttrs, encodingAttr);
}

static Value
padAndSetEncoding(OpBuilder &builder, Location loc, Value source,
                  IREE::LinalgExt::EncodingRole role, TypeRange operandTypes,
                  MatmulNarrowSizes narrow, ArrayAttr indexingMaps,
                  std::optional<CastOpInterface> castOp = std::nullopt) {
  Value padSource = castOp ? source.getDefiningOp()->getOperand(0) : source;
  // No need to specify original_type in the encoding poadded to pad(), because
  // the operand there is the `source` tensor, so it will default to reading its
  // original shape.
  auto encodingForPad =
      makeEncoding(builder, role, operandTypes,
                   /*originalType=*/Type{}, narrow, indexingMaps);
  Value padded = pad(builder, loc, padSource, encodingForPad);
  // For setEncoding() below, we potentially need to specify an encoding with an
  // explicit original_type, because the operand there is the padded tensor
  // returned by pad() above, but we want setEncoding to be aware of the
  // original source tensor shape, not the padded tensor shape. To limit IR
  // verbosity, we only specify the original original_type when it differs from
  // the tensor type that the encoding is applied to.
  auto encodingForSetEncoding = encodingForPad;
  if (padded.getType() != padSource.getType()) {
    encodingForSetEncoding = makeEncoding(
        builder, role, operandTypes, padSource.getType(), narrow, indexingMaps);
  }
  Value encoded = setEncoding(builder, loc, padded, encodingForSetEncoding);
  if (castOp) {
    encoded = castEncodedResult(builder, loc, encoded, castOp.value(),
                                encodingForSetEncoding);
  }
  return encoded;
}

static Value unsetEncodingAndExtractSlice(OpBuilder &builder, Location loc,
                                          Value source,
                                          SmallVector<OpFoldResult> sizes) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  auto unsetEncodingReturnType =
      RankedTensorType::get(sourceType.getShape(), sourceType.getElementType());
  auto unsetEncoding = builder
                           .create<IREE::LinalgExt::UnsetEncodingOp>(
                               loc, unsetEncodingReturnType, source)
                           .getResult();
  auto rank = sourceType.getRank();
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  return builder.create<tensor::ExtractSliceOp>(loc, unsetEncoding, offsets,
                                                sizes, strides);
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
/// TODO (#16176): Loosen restrictions on contraction ops once data tiling
/// can support more cases.
static LogicalResult isSupportedContractionOp(PatternRewriter &rewriter,
                                              linalg::LinalgOp linalgOp) {
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

struct setContractionOpEncoding
    : public OpInterfaceRewritePattern<linalg::ContractionOpInterface> {
  using OpInterfaceRewritePattern<
      linalg::ContractionOpInterface>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::ContractionOpInterface op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
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
    Value origLhs = inputs[0];
    Value origRhs = inputs[1];
    Value origOut = outputs[0];

    auto getElemType = [](Value v) -> Type {
      if (auto tensorType = llvm::dyn_cast<RankedTensorType>(v.getType())) {
        return tensorType.getElementType();
      }
      return {};
    };
    std::optional<CastOpInterface> maybeLhsCastOp =
        getDefiningNonI1ExtendingCastOp(origLhs);
    std::optional<CastOpInterface> maybeRhsCastOp =
        getDefiningNonI1ExtendingCastOp(origRhs);
    Type lhsElemType = maybeLhsCastOp ? getCastElemType(origLhs).value()
                                      : getElemType(origLhs);
    Type rhsElemType = maybeRhsCastOp ? getCastElemType(origRhs).value()
                                      : getElemType(origRhs);
    Type outElemType = getElemType(origOut);

    if (!lhsElemType || !rhsElemType || !outElemType) {
      return failure();
    }

    MatmulNarrowSizes narrowSizes =
        getMatmulNarrowSizes(origOut.getType().cast<ShapedType>(), linalgOp);

    Location loc = linalgOp.getLoc();
    SmallVector<Type> operandTypes(linalgOp->getOperandTypes());
    operandTypes[0] =
        cast<RankedTensorType>(operandTypes[0]).clone(lhsElemType);
    operandTypes[1] =
        cast<RankedTensorType>(operandTypes[1]).clone(rhsElemType);
    auto maps = linalgOp.getIndexingMaps();
    Value encodedLhs = padAndSetEncoding(
        rewriter, loc, origLhs, IREE::LinalgExt::EncodingRole::LHS,
        operandTypes, narrowSizes, maps, maybeLhsCastOp);
    Value encodedRhs = padAndSetEncoding(
        rewriter, loc, origRhs, IREE::LinalgExt::EncodingRole::RHS,
        operandTypes, narrowSizes, maps, maybeRhsCastOp);
    Value encodedOut = padAndSetEncoding(rewriter, loc, origOut,
                                         IREE::LinalgExt::EncodingRole::RESULT,
                                         operandTypes, narrowSizes, maps);
    Value opTiled;
    opTiled = clone(rewriter, linalgOp, encodedOut.getType(),
                    ValueRange{encodedLhs, encodedRhs, encodedOut})
                  ->getResult(0);

    // Sizes are computed by original output size.
    FailureOr<SmallVector<OpFoldResult>> origOutSizes =
        IREE::LinalgExt::getDims(rewriter, loc, origOut);
    if (failed(origOutSizes)) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "failed to get shape of result");
    }

    Value result = unsetEncodingAndExtractSlice(rewriter, loc, opTiled,
                                                origOutSizes.value());

    rewriter.replaceOp(linalgOp, result);
    return success();
  }
};

/// Pattern to fold a `linalg.fill` -> `iree_linalg_ext.set_encoding`
/// operation into a `linalg.fill` of the encoded type.
struct FoldFillWithSetEncoding
    : public OpRewritePattern<IREE::LinalgExt::SetEncodingOp> {
  using OpRewritePattern<IREE::LinalgExt::SetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::SetEncodingOp encodingOp,
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

struct SetEncodingPass : public SetEncodingBase<SetEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void SetEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  {
    RewritePatternSet patterns(context);
    patterns.insert<setContractionOpEncoding>(context);
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    patterns.insert<FoldFillWithSetEncoding>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> createSetEncodingPass() {
  return std::make_unique<SetEncodingPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
