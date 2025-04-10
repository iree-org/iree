// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/FuncToUtil/Patterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::InputConversion {

#define GEN_PASS_DEF_IREEIMPORTPUBLICPASS
#include "iree/compiler/InputConversion/Common/Passes.h.inc"

class IREEImportPublicPass final
    : public impl::IREEImportPublicPassBase<IREEImportPublicPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    ConversionTarget conversionTarget(getContext());
    conversionTarget.addLegalDialect<IREE::Util::UtilDialect>();
    conversionTarget.addIllegalOp<mlir::UnrealizedConversionCastOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    conversionTarget.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(&getContext());
    populateFuncToUtilPatterns(context, conversionTarget, typeConverter,
                               patterns, getOperation());

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      getOperation().emitError() << "conversion from input dialects failed";
      return signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler::InputConversion
