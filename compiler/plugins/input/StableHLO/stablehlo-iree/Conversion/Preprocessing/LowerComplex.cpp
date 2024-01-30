// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements passes to convert complex operations to equivalent real value
// operations. This does not include removing complex values from function
// argument or return types.

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h"
#include "stablehlo-iree/Conversion/Preprocessing/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LOWERCOMPLEX
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h.inc"

namespace {

struct ConvertComplexDot final : OpRewritePattern<mlir::stablehlo::DotOp> {
  using OpRewritePattern::OpRewritePattern;

  // Will decompose stablehlo::DotOp with complex parameters down to
  // four Dot operations in the following fashion:
  //   result.real = lhs.real <DOT> rhs.real - lhs.imag <DOT> rhs.imag
  //   result.imag = lhs.imag <DOT> rhs.real + lhs.real <DOT> rhs.imag
  //   result = complex(result.real, result.imag)
  LogicalResult matchAndRewrite(mlir::stablehlo::DotOp dot,
                                PatternRewriter &rewriter) const override {
    ArrayAttr precision = dot.getPrecisionConfigAttr();
    TypedValue<TensorType> lhs = dot.getLhs();
    TypedValue<TensorType> rhs = dot.getRhs();
    ShapedType lhsType = lhs.getType();
    ShapedType rhsType = rhs.getType();
    if (!isa<ComplexType>(lhsType.getElementType()) ||
        !isa<ComplexType>(rhsType.getElementType())) {
      return rewriter.notifyMatchFailure(dot, "lhs/rhs types are not complex");
    }

    Location loc = dot.getLoc();
    Value lhsReal = rewriter.create<mlir::stablehlo::RealOp>(loc, lhs);
    Value lhsImag = rewriter.create<mlir::stablehlo::ImagOp>(loc, lhs);
    Value rhsReal = rewriter.create<mlir::stablehlo::RealOp>(loc, rhs);
    Value rhsImag = rewriter.create<mlir::stablehlo::ImagOp>(loc, rhs);
    TensorType resultType = dot.getType();
    Type newType = mlir::hlo::createRealType(resultType);

    Value realComponent = rewriter.create<mlir::stablehlo::SubtractOp>(
        loc,
        rewriter.create<mlir::stablehlo::DotOp>(loc, newType, lhsReal, rhsReal,
                                                precision),
        rewriter.create<mlir::stablehlo::DotOp>(loc, newType, lhsImag, rhsImag,
                                                precision));
    Value imagComponent = rewriter.create<mlir::stablehlo::AddOp>(
        loc,
        rewriter.create<mlir::stablehlo::DotOp>(loc, newType, lhsReal, rhsImag,
                                                precision),
        rewriter.create<mlir::stablehlo::DotOp>(loc, newType, lhsImag, rhsReal,
                                                precision));
    Value result = rewriter.create<mlir::stablehlo::ComplexOp>(
        loc, resultType, realComponent, imagComponent);
    rewriter.replaceOp(dot, result);
    return success();
  }
};

struct LowerComplex final : impl::LowerComplexBase<LowerComplex> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populatePreprocessingComplexPatterns(ctx, &patterns);
    populateCanonicalizationPatterns(ctx, &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// Get a constant splat for the given value of type. Requires value to be of
// type static shaped RankedTensorType.
template <typename T>
ElementsAttr getSplat(Builder *b, RankedTensorType ty, T constant) {
  Type elementTy = getElementTypeOrSelf(ty);

  if (elementTy.isSignlessInteger()) {
    return DenseElementsAttr::get(ty, b->getIntegerAttr(elementTy, constant));
  }

  if (isa<FloatType>(elementTy)) {
    return DenseElementsAttr::get(ty, b->getFloatAttr(elementTy, constant));
  }

  if (auto complexTy = dyn_cast<ComplexType>(elementTy)) {
    auto complexElementTy = complexTy.getElementType();
    if (complexElementTy.isF32())
      return DenseElementsAttr::get(ty,
                                    static_cast<std::complex<float>>(constant));
    if (complexElementTy.isF64())
      return DenseElementsAttr::get(
          ty, static_cast<std::complex<double>>(constant));
  }
  llvm_unreachable("unhandled element type");
}

template <typename T>
ElementsAttr getSplat(Builder *b, Value val, T constant) {
  return getSplat(b, cast<RankedTensorType>(val.getType()), constant);
}
} // end anonymous namespace

namespace {
#include "stablehlo-iree/Conversion/Preprocessing/ComplexLoweringPatterns.h.inc"
} // end anonymous namespace

void populatePreprocessingComplexPatterns(MLIRContext *context,
                                          RewritePatternSet *patterns) {
  patterns->add<ConvertComplexDot>(context);
  populateWithGenerated(*patterns);
}
} // namespace mlir::iree_compiler::stablehlo
