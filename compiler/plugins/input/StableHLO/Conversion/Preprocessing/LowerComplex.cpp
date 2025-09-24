// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements passes to convert complex operations to equivalent real value
// operations. This does not include removing complex values from function
// argument or return types.

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LOWERCOMPLEX
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h.inc"

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
    TypedValue<RankedTensorType> lhs = dot.getLhs();
    TypedValue<RankedTensorType> rhs = dot.getRhs();
    ShapedType lhsType = lhs.getType();
    ShapedType rhsType = rhs.getType();
    if (!isa<ComplexType>(lhsType.getElementType()) ||
        !isa<ComplexType>(rhsType.getElementType())) {
      return rewriter.notifyMatchFailure(dot, "lhs/rhs types are not complex");
    }

    Location loc = dot.getLoc();
    Value lhsReal = mlir::stablehlo::RealOp::create(rewriter, loc, lhs);
    Value lhsImag = mlir::stablehlo::ImagOp::create(rewriter, loc, lhs);
    Value rhsReal = mlir::stablehlo::RealOp::create(rewriter, loc, rhs);
    Value rhsImag = mlir::stablehlo::ImagOp::create(rewriter, loc, rhs);
    TensorType resultType = dot.getType();
    Type newType = mlir::hlo::createRealType(resultType);

    Value realComponent = mlir::stablehlo::SubtractOp::create(
        rewriter, loc,
        mlir::stablehlo::DotOp::create(rewriter, loc, newType, lhsReal, rhsReal,
                                       precision),
        mlir::stablehlo::DotOp::create(rewriter, loc, newType, lhsImag, rhsImag,
                                       precision));
    Value imagComponent = mlir::stablehlo::AddOp::create(
        rewriter, loc,
        mlir::stablehlo::DotOp::create(rewriter, loc, newType, lhsReal, rhsImag,
                                       precision),
        mlir::stablehlo::DotOp::create(rewriter, loc, newType, lhsImag, rhsReal,
                                       precision));
    Value result = mlir::stablehlo::ComplexOp::create(
        rewriter, loc, resultType, realComponent, imagComponent);
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
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/ComplexLoweringPatterns.h.inc"
} // end anonymous namespace

void populatePreprocessingComplexPatterns(MLIRContext *context,
                                          RewritePatternSet *patterns) {
  patterns->add<ConvertComplexDot>(context);
  populateWithGenerated(*patterns);
}
} // namespace mlir::iree_compiler::stablehlo
