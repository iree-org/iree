// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

// A pass converting MLIR Standard operations into the IREE VM dialect.
// Used only for testing as in the common case we only rely on rewrite patterns.
class ConvertStandardToVMTestPass
    : public PassWrapper<ConvertStandardToVMTestPass,
                         OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override {
    return "test-iree-convert-std-to-vm";
  }

  StringRef getDescription() const override {
    return "Convert Standard Ops to the IREE VM dialect";
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<IREE::VM::VMDialect>();
    target.addIllegalDialect<StandardOpsDialect>();

    IREE::VM::TypeConverter typeConverter(
        IREE::VM::getTargetOptionsFromFlags());

    OwningRewritePatternList patterns(&getContext());
    populateStandardToVMPatterns(&getContext(), typeConverter, patterns);

    // NOTE: we allow other dialects besides just VM during this pass as we are
    // only trying to eliminate the std ops. When used as part of a larger set
    // of rewrites a full conversion should be used instead.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

namespace IREE {
namespace VM {
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertStandardToVMTestPass() {
  return std::make_unique<ConvertStandardToVMTestPass>();
}
}  // namespace VM
}  // namespace IREE

static PassRegistration<ConvertStandardToVMTestPass> pass;

}  // namespace iree_compiler
}  // namespace mlir
