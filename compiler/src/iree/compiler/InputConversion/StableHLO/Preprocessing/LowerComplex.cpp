// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements passes to convert complex operations to equivalent real value
// operations. This does not include removing complex values from function
// argument or return types.

#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h"
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LOWERCOMPLEX
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h.inc"

namespace {
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
}  // end anonymous namespace

namespace {
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/ComplexLoweringPatterns.h.inc"
}  // end anonymous namespace

void populatePreprocessingComplexPatterns(MLIRContext * /*context*/,
                                          RewritePatternSet *patterns) {
  populateWithGenerated(*patterns);
}
}  // namespace mlir::iree_compiler::stablehlo
