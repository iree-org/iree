// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/ConvertHALToVM.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/hal.imports.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
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

// A pass converting the IREE flow dialect into the IREE HAL dialect.
class ConvertHALToVMPass
    : public PassWrapper<ConvertHALToVMPass, OperationPass<ModuleOp>> {
 public:
  ConvertHALToVMPass()
      : targetOptions_(IREE::VM::getTargetOptionsFromFlags()) {}
  explicit ConvertHALToVMPass(IREE::VM::TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void runOnOperation() override {
    auto *context = &getContext();

    VMConversionTarget conversionTarget(context);
    IREE::VM::TypeConverter typeConverter(targetOptions_);

    mlir::ModuleOp outerModuleOp, innerModuleOp;
    std::tie(outerModuleOp, innerModuleOp) =
        VMConversionTarget::nestModuleForConversion(getOperation());

    appendImportModule(
        StringRef(hal_imports_create()->data, hal_imports_create()->size),
        innerModuleOp);

    OwningRewritePatternList conversionPatterns;
    populateStandardToVMPatterns(context, typeConverter, conversionPatterns);

    SymbolTable importSymbols(innerModuleOp);
    populateHALToVMPatterns(context, importSymbols, conversionPatterns,
                            typeConverter);

    if (failed(applyPartialConversion(outerModuleOp, conversionTarget,
                                      conversionPatterns))) {
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

static PassRegistration<ConvertHALToVMPass> pass(
    "iree-convert-hal-to-vm",
    "Convert the IREE HAL dialect to the IREE VM dialect");

}  // namespace iree_compiler
}  // namespace mlir
