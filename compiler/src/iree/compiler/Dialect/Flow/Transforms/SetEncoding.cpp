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
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

using IREE::LinalgExt::TensorEncoding;

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

// Returns the element type of `t` if it is a `ShapedType`, else return
// `t` itself.
static Type getElementTypeOrType(Type t) {
  if (auto shapedType = t.dyn_cast<ShapedType>()) {
    return shapedType.getElementType();
  }
  return t;
}

/// Returns a constant 0 of type `elementType`.
static FailureOr<Value> getZero(OpBuilder &builder, Location loc,
                                Type elementType) {
  Attribute zeroVal =
      TypeSwitch<Type, Attribute>(elementType)
          .Case<FloatType>([&](FloatType floatType) -> Attribute {
            return builder.getFloatAttr(floatType, 0);
          })
          .Case<IntegerType>([&](IntegerType intType) -> Attribute {
            return builder.getIntegerAttr(intType, 0);
          })
          .Default([](Type type) { return nullptr; });
  if (!zeroVal) return failure();
  return builder.create<arith::ConstantOp>(loc, zeroVal, elementType)
      .getResult();
}

/// Pads `value` to `padding` if needed. If no padding is specified,
/// return `value` itself.
static FailureOr<Value> padIfNeeded(
    OpBuilder &builder, Location loc, Value value,
    std::optional<int64_t> padding = std::nullopt) {
  if (!padding) return value;

  OpFoldResult paddingOfr = builder.getIndexAttr(padding.value());
  FailureOr<SmallVector<OpFoldResult>> shape =
      LinalgExt::getDims(builder, loc, value);
  if (failed(shape)) {
    return failure();
  }

  OpFoldResult zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> lowPad(shape->size(), zero);
  SmallVector<OpFoldResult> highPad(shape->size(), zero);

  // The low padding is always zero. The high padding is
  // shape.ceildDiv(padding) - shape.
  AffineExpr paddingExpr, shapeExpr;
  bindSymbols(builder.getContext(), paddingExpr, shapeExpr);
  AffineExpr highPadExpr =
      shapeExpr.ceilDiv(paddingExpr) * paddingExpr - shapeExpr;
  for (auto shape : llvm::enumerate(shape.value())) {
    highPad[shape.index()] = affine::makeComposedFoldedAffineApply(
        builder, loc, highPadExpr, {paddingOfr, shape.value()});
  }

  // If all high padding evaluate to 0, then nothing to do.
  if (llvm::all_of(highPad, [](OpFoldResult ofr) {
        return isConstantIntValue(ofr, 0);
      })) {
    return value;
  }

  FailureOr<Value> zeroVal =
      getZero(builder, loc, getElementTypeOrSelf(value.getType()));
  if (failed(zeroVal)) {
    return failure();
  }
  auto padOp = builder.create<tensor::PadOp>(loc, /*resultType=*/nullptr, value,
                                             lowPad, highPad, zeroVal.value());
  return padOp.getResult();
}

namespace {

/// Rewrites the matmul op to work on tensors with encoding. Optionally
/// also pads the operands.
struct SetMatmulEncoding : public OpRewritePattern<linalg::MatmulOp> {
  SetMatmulEncoding(MLIRContext *context, int64_t padding,
                    PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        padding(padding) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasTensorSemantics()) return failure();
    auto inputs = matmulOp.getDpsInputOperands();
    auto outputs = matmulOp.getDpsInitOperands();
    auto hasEncoding = [](OpOperand *operand) -> bool {
      auto type = operand->get().getType().dyn_cast<RankedTensorType>();
      return type && type.getEncoding();
    };
    if (llvm::any_of(inputs, hasEncoding) ||
        llvm::any_of(outputs, hasEncoding)) {
      return failure();
    }

    Value origLhs = inputs[0]->get();
    Value origRhs = inputs[1]->get();
    Value origOut = outputs[0]->get();

    auto getElemType = [](Value v) -> Type {
      if (auto tensorType = v.getType().dyn_cast<RankedTensorType>()) {
        return tensorType.getElementType();
      }
      return {};
    };
    Type lhsElemType = getElemType(origLhs);
    Type rhsElemType = getElemType(origRhs);
    Type outElemType = getElemType(origOut);

    if (!lhsElemType || !rhsElemType || !outElemType) {
      return failure();
    }

    TensorEncoding lhsEncoding;
    TensorEncoding rhsEncoding;
    TensorEncoding outEncoding;

    if (lhsElemType.isF32() && rhsElemType.isF32() && outElemType.isF32()) {
      lhsEncoding = TensorEncoding::MATMUL_F32F32F32_LHS;
      rhsEncoding = TensorEncoding::MATMUL_F32F32F32_RHS;
      outEncoding = TensorEncoding::MATMUL_F32F32F32_RESULT;
    } else if (lhsElemType.isSignlessInteger(8) &&
               rhsElemType.isSignlessInteger(8) &&
               outElemType.isSignlessInteger(32)) {
      lhsEncoding = TensorEncoding::MATMUL_I8I8I32_LHS;
      rhsEncoding = TensorEncoding::MATMUL_I8I8I32_RHS;
      outEncoding = TensorEncoding::MATMUL_I8I8I32_RESULT;
    } else {
      return rewriter.notifyMatchFailure(
          matmulOp,
          "unhandled combination of (lhs, rhs, result) element types");
    }

    Location loc = matmulOp.getLoc();

    // Set encoding for LHS (pad if necessary)
    FailureOr<Value> paddedLhs = padIfNeeded(rewriter, loc, origLhs, padding);
    if (failed(paddedLhs)) {
      return rewriter.notifyMatchFailure(matmulOp, "failed to pad lhs");
    }

    // Set encoding for RHS (pad if necessary)
    FailureOr<Value> paddedRhs = padIfNeeded(rewriter, loc, origRhs, padding);
    if (failed(paddedRhs)) {
      return rewriter.notifyMatchFailure(matmulOp, "failed to pad rhs");
    }

    // Set encoding for OUTS (pad if necessary)
    FailureOr<Value> paddedOut = padIfNeeded(rewriter, loc, origOut, padding);
    if (failed(paddedOut)) {
      return rewriter.notifyMatchFailure(matmulOp, "failed to pad output");
    }

    Value encodedLhs = rewriter.create<IREE::LinalgExt::SetEncodingOp>(
        loc, paddedLhs.value(), lhsEncoding);
    Value encodedRhs = rewriter.create<IREE::LinalgExt::SetEncodingOp>(
        loc, paddedRhs.value(), rhsEncoding);
    Value encodedOut = rewriter.create<IREE::LinalgExt::SetEncodingOp>(
        loc, paddedOut.value(), outEncoding);

    auto matmulTiled = rewriter.create<linalg::MatmulOp>(
        loc, encodedOut.getType(), ValueRange{encodedLhs, encodedRhs},
        encodedOut);
    auto unsetEncoding = rewriter.create<IREE::LinalgExt::UnsetEncodingOp>(
        loc, matmulTiled.getResult(0));

    Value replacement = unsetEncoding.getResult();
    // If the output was padded, extract the actual output.
    if (paddedOut.value() != origOut) {
      auto replacementRank =
          replacement.getType().cast<RankedTensorType>().getRank();
      // Offsets are all 0.
      OpFoldResult zero = rewriter.getIndexAttr(0);
      SmallVector<OpFoldResult> offsets(replacementRank, zero);
      // Strides are all 1.
      OpFoldResult one = rewriter.getIndexAttr(1);
      SmallVector<OpFoldResult> strides(replacementRank, one);

      // Sizes are computed by original output size.
      FailureOr<SmallVector<OpFoldResult>> sizes =
          LinalgExt::getDims(rewriter, loc, origOut);
      if (failed(sizes)) {
        return rewriter.notifyMatchFailure(matmulOp,
                                           "failed to get shape of result");
      }
      replacement = rewriter.create<tensor::ExtractSliceOp>(
          loc, replacement, offsets, sizes.value(), strides);
    }

    rewriter.replaceOp(matmulOp, replacement);
    return success();
  }

 private:
  int64_t padding;
};

/// Pattern to fold a `linalg.fill` -> `iree_linalg_ext.set_encoding`
/// operation into a `linalg.fill` of the encoded type.
struct FoldFillWithSetEncoding
    : public OpRewritePattern<IREE::LinalgExt::SetEncodingOp> {
  using OpRewritePattern<IREE::LinalgExt::SetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = encodingOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp) return failure();

    // Create a new fill op, with outs being defined by a new `tensor.empty` op.
    RankedTensorType encodingType = encodingOp.getResultType();
    Location loc = fillOp.getLoc();
    SmallVector<OpFoldResult> dimValues =
        tensor::createDimValues(rewriter, loc, fillOp.getOutputs()[0]);
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
}  // namespace

void SetEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  {
    RewritePatternSet patterns(context);
    patterns.insert<SetMatmulEncoding>(context, defaultPadding);
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    patterns.insert<FoldFillWithSetEncoding>(context);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> createSetEncodingPass() {
  return std::make_unique<SetEncodingPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
