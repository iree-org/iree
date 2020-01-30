// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/VMLA/Conversion/VMLAToVM/ConvertVMLAToVM.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "iree/compiler/Dialect/VMLA/vmla.imports.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

void populateVMLAToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                              OwningRewritePatternList &patterns,
                              TypeConverter &typeConverter) {
  // TODO(benvanik): conversion patterns.
}

namespace {

// A pass converting the IREE flow dialect into the IREE VMLA dialect.
class ConvertVMLAToVMPass : public ModulePass<ConvertVMLAToVMPass> {
 public:
  void runOnModule() override {
    auto *context = &getContext();

    VMConversionTarget conversionTarget(context);
    VMTypeConverter typeConverter;

    mlir::ModuleOp outerModuleOp, innerModuleOp;
    std::tie(outerModuleOp, innerModuleOp) =
        VMConversionTarget::nestModuleForConversion(getModule());

    appendImportModule(
        StringRef(vmla_imports_create()->data, vmla_imports_create()->size),
        innerModuleOp);

    OwningRewritePatternList conversionPatterns;
    populateStandardToVMPatterns(context, conversionPatterns);

    SymbolTable importSymbols(innerModuleOp);
    populateVMLAToVMPatterns(context, importSymbols, conversionPatterns,
                             typeConverter);

    if (failed(applyPartialConversion(outerModuleOp, conversionTarget,
                                      conversionPatterns, &typeConverter))) {
      outerModuleOp.emitError() << "conversion to vm.module failed";
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> createConvertVMLAToVMPass() {
  return std::make_unique<ConvertVMLAToVMPass>();  // NOLINT
}

static PassRegistration<ConvertVMLAToVMPass> pass(
    "iree-convert-vmla-to-vm",
    "Convert the IREE VMLA dialect to the IREE VM dialect");

}  // namespace iree_compiler
}  // namespace mlir
