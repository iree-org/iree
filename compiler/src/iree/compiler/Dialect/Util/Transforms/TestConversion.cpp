// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/Conversion/FuncToUtil/Patterns.h"
#include "iree/compiler/Dialect/Util/Conversion/MemRefToUtil/Patterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_TESTCONVERSIONPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

static Value buildUnrealizedConversionCastOp(OpBuilder &builder, Type toType,
                                             ValueRange inputs, Location loc) {
  return builder.create<UnrealizedConversionCastOp>(loc, toType, inputs)
      .getResult(0);
}

class TestConversionPass
    : public impl::TestConversionPassBase<TestConversionPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    auto *context = &getContext();
    ConversionTarget conversionTarget(*context);
    conversionTarget.addLegalDialect<arith::ArithDialect>();
    conversionTarget.addLegalDialect<IREE::Util::UtilDialect>();
    conversionTarget.addLegalOp<UnrealizedConversionCastOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    if (widenIntegers) {
      // Promote all integers < 32bit to 32bit to test type conversion on
      // for tests that are sensitive to that.
      typeConverter.addConversion([](IntegerType type) {
        if (type.getWidth() < 32) {
          return IntegerType::get(type.getContext(), 32);
        }
        return type;
      });
    }

    typeConverter.addTargetMaterialization(buildUnrealizedConversionCastOp);

    RewritePatternSet patterns(&getContext());
    populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                   patterns);
    if (structuralConversion) {
      populateGenericStructuralConversionPatterns(context, conversionTarget,
                                                  typeConverter, patterns);
    } else {
      populateFuncToUtilPatterns(context, conversionTarget, typeConverter,
                                 patterns, getOperation());
    }
    populateMemRefToUtilPatterns(context, conversionTarget, typeConverter,
                                 patterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      getOperation()->emitError() << "conversion to util failed";
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
