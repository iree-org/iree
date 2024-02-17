// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/Conversion/MemRefToUtil/Patterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Util {

namespace {

class TestConversionPass : public TestConversionBase<TestConversionPass> {
public:
  TestConversionPass() = default;
  TestConversionPass(const TestConversionPass &) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, mlir::arith::ArithDialect,
                    math::MathDialect, mlir::affine::AffineDialect,
                    memref::MemRefDialect>();
  }

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

    RewritePatternSet patterns(&getContext());
    populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                   patterns);
    populateGenericStructuralConversionPatterns(context, conversionTarget,
                                                typeConverter, patterns);
    populateMemRefToUtilPatterns(context, conversionTarget, typeConverter,
                                 patterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      getOperation()->emitError() << "conversion to util failed";
      return signalPassFailure();
    }
  }

  Option<bool> widenIntegers{
      *this, "widen-integers",
      llvm::cl::desc("Tests type conversion by widening integers to i32")};
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTestConversionPass() {
  return std::make_unique<TestConversionPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
