// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/ConvertHALToVM.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/hal.imports.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/IREEToVM/ConvertIREEToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

extern void populateHALAllocatorToVMPatterns(
    MLIRContext *context, SymbolTable &importSymbols,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns);
extern void populateHALBufferToVMPatterns(MLIRContext *context,
                                          SymbolTable &importSymbols,
                                          TypeConverter &typeConverter,
                                          OwningRewritePatternList &patterns);
extern void populateHALBufferViewToVMPatterns(
    MLIRContext *context, SymbolTable &importSymbols,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns);
extern void populateHALCommandBufferToVMPatterns(
    MLIRContext *context, SymbolTable &importSymbols,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns);
extern void populateHALConstantToVMPatterns(MLIRContext *context,
                                            SymbolTable &importSymbols,
                                            TypeConverter &typeConverter,
                                            OwningRewritePatternList &patterns);
extern void populateHALControlFlowToVMPatterns(
    MLIRContext *context, SymbolTable &importSymbols,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns);
extern void populateHALDeviceToVMPatterns(MLIRContext *context,
                                          SymbolTable &importSymbols,
                                          TypeConverter &typeConverter,
                                          OwningRewritePatternList &patterns);
extern void populateHALExecutableToVMPatterns(
    MLIRContext *context, SymbolTable &importSymbols,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns);
extern void populateHALExperimentalToVMPatterns(
    MLIRContext *context, SymbolTable &importSymbols,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns);
extern void populateHALSemaphoreToVMPatterns(
    MLIRContext *context, SymbolTable &importSymbols,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns);
extern void populateHALVariableToVMPatterns(MLIRContext *context,
                                            SymbolTable &importSymbols,
                                            TypeConverter &typeConverter,
                                            OwningRewritePatternList &patterns);

void populateHALToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                             OwningRewritePatternList &patterns,
                             TypeConverter &typeConverter) {
  populateHALAllocatorToVMPatterns(context, importSymbols, typeConverter,
                                   patterns);
  populateHALBufferToVMPatterns(context, importSymbols, typeConverter,
                                patterns);
  populateHALBufferViewToVMPatterns(context, importSymbols, typeConverter,
                                    patterns);
  populateHALCommandBufferToVMPatterns(context, importSymbols, typeConverter,
                                       patterns);
  populateHALConstantToVMPatterns(context, importSymbols, typeConverter,
                                  patterns);
  populateHALControlFlowToVMPatterns(context, importSymbols, typeConverter,
                                     patterns);
  populateHALDeviceToVMPatterns(context, importSymbols, typeConverter,
                                patterns);
  populateHALExecutableToVMPatterns(context, importSymbols, typeConverter,
                                    patterns);
  populateHALExperimentalToVMPatterns(context, importSymbols, typeConverter,
                                      patterns);
  populateHALSemaphoreToVMPatterns(context, importSymbols, typeConverter,
                                   patterns);
  populateHALVariableToVMPatterns(context, importSymbols, typeConverter,
                                  patterns);
}

namespace {

// A pass converting the IREE HAL dialect into the IREE VM dialect.
class ConvertHALToVMPass
    : public PassWrapper<ConvertHALToVMPass, OperationPass<ModuleOp>> {
 public:
  explicit ConvertHALToVMPass(IREE::VM::TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEDialect, IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override { return "iree-convert-hal-to-vm"; }

  StringRef getDescription() const override {
    return "Convert the IREE HAL dialect to the IREE VM dialect";
  }

  void runOnOperation() override {
    if (getOperation().getBody()->empty()) return;
    auto *context = &getContext();

    VMConversionTarget conversionTarget(context);
    IREE::VM::TypeConverter typeConverter(targetOptions_);

    mlir::ModuleOp outerModuleOp, innerModuleOp;
    std::tie(outerModuleOp, innerModuleOp) =
        VMConversionTarget::nestModuleForConversion(getOperation());

    (void)appendImportModule(StringRef(iree_hal_imports_create()->data,
                                       iree_hal_imports_create()->size),
                             innerModuleOp);

    OwningRewritePatternList conversionPatterns(&getContext());
    populateStandardToVMPatterns(context, typeConverter, conversionPatterns);
    populateIREEToVMPatterns(context, typeConverter, conversionPatterns);

    SymbolTable importSymbols(innerModuleOp);
    populateHALToVMPatterns(context, importSymbols, conversionPatterns,
                            typeConverter);

    if (failed(applyPartialConversion(outerModuleOp, conversionTarget,
                                      std::move(conversionPatterns)))) {
      outerModuleOp.emitError() << "conversion to vm.module failed";
      return signalPassFailure();
    }
  }

 private:
  IREE::VM::TargetOptions targetOptions_;
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertHALToVMPass(
    IREE::VM::TargetOptions targetOptions) {
  return std::make_unique<ConvertHALToVMPass>(targetOptions);
}

static PassRegistration<ConvertHALToVMPass> pass([] {
  auto options = IREE::VM::getTargetOptionsFromFlags();
  return std::make_unique<ConvertHALToVMPass>(options);
});

}  // namespace iree_compiler
}  // namespace mlir
