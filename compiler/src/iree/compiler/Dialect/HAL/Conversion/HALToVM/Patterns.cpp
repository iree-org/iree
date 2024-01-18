// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/hal.imports.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/Conversion/UtilToVM/ConvertUtilToVM.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

extern void populateHALAllocatorToVMPatterns(MLIRContext *context,
                                             SymbolTable &importSymbols,
                                             TypeConverter &typeConverter,
                                             RewritePatternSet &patterns);
extern void populateHALBufferToVMPatterns(MLIRContext *context,
                                          SymbolTable &importSymbols,
                                          TypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
extern void populateHALBufferViewToVMPatterns(MLIRContext *context,
                                              SymbolTable &importSymbols,
                                              TypeConverter &typeConverter,
                                              RewritePatternSet &patterns);
extern void populateHALChannelToVMPatterns(MLIRContext *context,
                                           SymbolTable &importSymbols,
                                           TypeConverter &typeConverter,
                                           RewritePatternSet &patterns);
extern void populateHALCommandBufferToVMPatterns(MLIRContext *context,
                                                 SymbolTable &importSymbols,
                                                 TypeConverter &typeConverter,
                                                 RewritePatternSet &patterns);
extern void populateHALDeviceToVMPatterns(MLIRContext *context,
                                          SymbolTable &importSymbols,
                                          TypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
extern void populateHALDevicesToVMPatterns(MLIRContext *context,
                                           SymbolTable &importSymbols,
                                           TypeConverter &typeConverter,
                                           RewritePatternSet &patterns);
extern void populateHALExecutableToVMPatterns(MLIRContext *context,
                                              SymbolTable &importSymbols,
                                              TypeConverter &typeConverter,
                                              RewritePatternSet &patterns);
extern void populateHALExperimentalToVMPatterns(MLIRContext *context,
                                                SymbolTable &importSymbols,
                                                TypeConverter &typeConverter,
                                                RewritePatternSet &patterns);
extern void populateHALFenceToVMPatterns(MLIRContext *context,
                                         SymbolTable &importSymbols,
                                         TypeConverter &typeConverter,
                                         RewritePatternSet &patterns);

void populateHALToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                             RewritePatternSet &patterns,
                             TypeConverter &typeConverter) {
  populateHALAllocatorToVMPatterns(context, importSymbols, typeConverter,
                                   patterns);
  populateHALBufferToVMPatterns(context, importSymbols, typeConverter,
                                patterns);
  populateHALBufferViewToVMPatterns(context, importSymbols, typeConverter,
                                    patterns);
  populateHALChannelToVMPatterns(context, importSymbols, typeConverter,
                                 patterns);
  populateHALCommandBufferToVMPatterns(context, importSymbols, typeConverter,
                                       patterns);
  populateHALDeviceToVMPatterns(context, importSymbols, typeConverter,
                                patterns);
  populateHALDevicesToVMPatterns(context, importSymbols, typeConverter,
                                 patterns);
  populateHALExecutableToVMPatterns(context, importSymbols, typeConverter,
                                    patterns);
  populateHALExperimentalToVMPatterns(context, importSymbols, typeConverter,
                                      patterns);
  populateHALFenceToVMPatterns(context, importSymbols, typeConverter, patterns);
}

namespace {

// A pass converting the IREE HAL dialect into the IREE VM dialect.
class ConvertHALToVMPass
    : public PassWrapper<ConvertHALToVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertHALToVMPass)

  explicit ConvertHALToVMPass(IREE::VM::TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override { return "iree-convert-hal-to-vm"; }

  StringRef getDescription() const override {
    return "Convert the IREE HAL dialect to the IREE VM dialect";
  }

  void runOnOperation() override {
    if (getOperation().getBody()->empty())
      return;
    auto *context = &getContext();

    VMConversionTarget conversionTarget(context);
    IREE::VM::TypeConverter typeConverter(targetOptions_);

    mlir::ModuleOp outerModuleOp, innerModuleOp;
    std::tie(outerModuleOp, innerModuleOp) =
        VMConversionTarget::nestModuleForConversion(getOperation());

    (void)appendImportModule(StringRef(iree_hal_imports_create()->data,
                                       iree_hal_imports_create()->size),
                             innerModuleOp);

    RewritePatternSet conversionPatterns(&getContext());
    populateStandardToVMPatterns(context, typeConverter, conversionPatterns);
    populateUtilToVMPatterns(context, conversionTarget, typeConverter,
                             conversionPatterns);

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

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createConvertHALToVMPass(IREE::VM::TargetOptions targetOptions) {
  return std::make_unique<ConvertHALToVMPass>(targetOptions);
}

static PassRegistration<ConvertHALToVMPass> pass([] {
  auto options = IREE::VM::TargetOptions::FromFlags::get();
  return std::make_unique<ConvertHALToVMPass>(options);
});

} // namespace mlir::iree_compiler
