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

#include <tuple>

#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// TODO(benvanik): import dialect registration.

// Runs conversion with registered input dialects.
class ConversionPass : public OperationPass<ConversionPass, mlir::ModuleOp> {
 public:
  void runOnOperation() override {
    auto *context = &getContext();
    VMConversionTarget conversionTarget(context);
    VMTypeConverter typeConverter;

    mlir::ModuleOp outerModuleOp, innerModuleOp;
    std::tie(outerModuleOp, innerModuleOp) =
        VMConversionTarget::nestModuleForConversion(getOperation());

    // TODO(benvanik): registration system for custom dialects.

    OwningRewritePatternList conversionPatterns;
    populateStandardToVMPatterns(context, conversionPatterns);

    if (failed(applyFullConversion(outerModuleOp, conversionTarget,
                                   conversionPatterns, &typeConverter))) {
      outerModuleOp.emitError() << "conversion to vm.module failed";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OpPassBase<mlir::ModuleOp>> createConversionPass() {
  return std::make_unique<ConversionPass>();
}

static PassRegistration<ConversionPass> pass(
    "iree-vm-conversion", "Converts from various dialects to the VM dialect");

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
