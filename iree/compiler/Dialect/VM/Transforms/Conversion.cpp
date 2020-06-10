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

#include "iree/compiler/Dialect/IREE/Conversion/PreserveCompilerHints.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/IREEToVM/ConvertIREEToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {
namespace {

// When converting to the VM, it is safe to remove any identity tie_shape
// ops that remain.
class ElideTieShapeOp : public OpConversionPattern<Shape::TieShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Shape::TieShapeOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};

// Returns a stably sorted list of dialect interfaces of T for all dialects used
// within the given module.
template <typename T>
SmallVector<const T *, 4> gatherUsedDialectInterfaces(mlir::ModuleOp moduleOp) {
  SmallPtrSet<const T *, 4> resultSet;
  moduleOp.walk([&](Operation *op) {
    auto *dialect = op->getDialect();
    if (!dialect) return;
    auto *dialectInterface = dialect->getRegisteredInterface<T>();
    if (!dialectInterface) return;
    resultSet.insert(dialectInterface);
  });

  // NOTE: to ensure deterministic output we sort the result so that imports are
  // always added in a consistent order.
  SmallVector<const T *, 4> results = {resultSet.begin(), resultSet.end()};
  llvm::sort(
      results, +[](const T *a, const T *b) {
        return a->getDialect()->getNamespace().compare(
                   b->getDialect()->getNamespace()) < 0;
      });
  return results;
}

}  // namespace

// Runs conversion with registered input dialects.
class ConversionPass
    : public PassWrapper<ConversionPass, OperationPass<mlir::ModuleOp>> {
 public:
  void runOnOperation() override {
    auto *context = &getContext();
    VMConversionTarget conversionTarget(context);
    VMTypeConverter typeConverter;

    mlir::ModuleOp outerModuleOp, innerModuleOp;
    std::tie(outerModuleOp, innerModuleOp) =
        VMConversionTarget::nestModuleForConversion(getOperation());

    // Append all vm.import ops from used dialects so that we can look them up
    // during conversion.
    auto usedDialects =
        gatherUsedDialectInterfaces<VMConversionDialectInterface>(
            innerModuleOp);
    for (auto *dialectInterface : usedDialects) {
      auto outerImportModuleOp = dialectInterface->getVMImportModule();
      for (auto importModuleOp :
           outerImportModuleOp->getOps<IREE::VM::ModuleOp>()) {
        if (failed(appendImportModule(importModuleOp, innerModuleOp))) {
          importModuleOp.emitError() << "failed to import module";
          return signalPassFailure();
        }
      }
    }

    OwningRewritePatternList conversionPatterns;
    populateIREEToVMPatterns(context, conversionPatterns);
    populateStandardToVMPatterns(context, conversionPatterns);
    conversionPatterns.insert<ElideTieShapeOp>(context);

    // Populate patterns from all used dialects, providing the imports they
    // registered earlier.
    SymbolTable importSymbols(innerModuleOp);
    for (auto *dialectInterface : usedDialects) {
      dialectInterface->populateVMConversionPatterns(
          importSymbols, conversionPatterns, typeConverter);
    }
    Shape::populateFoldConversionPatterns(context, conversionPatterns);
    populatePreserveCompilerHintsPatterns(context, conversionPatterns);
    setupCompilerHintsLegality(context, conversionTarget, typeConverter);

    if (failed(applyPartialConversion(outerModuleOp, conversionTarget,
                                      conversionPatterns))) {
      outerModuleOp.emitError() << "conversion to vm.module failed";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass() {
  return std::make_unique<ConversionPass>();
}

static PassRegistration<ConversionPass> pass(
    "iree-vm-conversion", "Converts from various dialects to the VM dialect");

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
