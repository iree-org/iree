// Copyright 2021 Google LLC
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

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/Modules/VMVX/Conversion/HALToVMVX/ConvertHALToVMVX.h"
#include "iree/compiler/Dialect/Modules/VMVX/Conversion/StandardToVMVX/ConvertStandardToVMVX.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXTypes.h"
#include "iree/compiler/Dialect/Modules/VMVX/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/TosaToStandard/TosaToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMVX {

// Runs conversion with registered input dialects.
class ConversionPass
    : public PassWrapper<ConversionPass, OperationPass<mlir::ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEDialect, IREE::HAL::HALDialect, IREE::VM::VMDialect,
                    IREE::VMVX::VMVXDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;

    typeConverter.addConversion([](Type type) { return type; });

    // Run a pre-pass that updates the entry function signature.
    for (auto funcOp : getOperation().getOps<FuncOp>()) {
      if (funcOp.isPublic()) {
        if (failed(updateHALToVMVXEntryFuncOp(funcOp, typeConverter))) {
          return signalPassFailure();
        }
      }
    }

    // Ensure all input dialects go away.
    ConversionTarget conversionTarget(*context);
    conversionTarget.addIllegalDialect<IREE::HAL::HALDialect>();
    conversionTarget.addIllegalDialect<tensor::TensorDialect>();
    conversionTarget.addLegalDialect<IREEDialect>();
    conversionTarget.addLegalDialect<IREE::VMVX::VMVXDialect>();
    conversionTarget.addLegalDialect<mlir::StandardOpsDialect>();
    conversionTarget.addLegalDialect<mlir::AffineDialect>();
    conversionTarget.addLegalDialect<memref::MemRefDialect>();
    conversionTarget.addLegalOp<mlir::UnrealizedConversionCastOp>();

    OwningRewritePatternList conversionPatterns(&getContext());
    populateHALToVMVXPatterns(context, conversionPatterns, typeConverter);
    populateStandardToVMVXPatterns(context, conversionPatterns, typeConverter);

    // Use the default 64-bit lowering for TOSA's ApplyScale operator:
    //   This lowering widens integer types to 64-bit an performs the non-fused
    //   operations, specifically multiply, add, and shift. Bit-widening
    //   is used to guarantee higher-order bits are not truncated during the
    //   multiply or add.
    //
    // TODO(suderman): remove the TOSA layering violation and lower to standard/
    // math ops instead.
    tosa::populateTosaRescaleToStandardConversionPatterns(&conversionPatterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      getOperation().emitError() << "conversion to the VMVX dialect failed";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass() {
  return std::make_unique<ConversionPass>();
}

static PassRegistration<ConversionPass> pass(
    "iree-vmvx-conversion",
    "Converts from various dialects to the VMVX dialect");

}  // namespace VMVX
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
