// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <tuple>

#include "iree/compiler/Dialect/IREE/Conversion/PreserveCompilerHints.h"
#include "iree/compiler/Dialect/IREE/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/IREEToVM/ConvertIREEToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/MathToVM/ConvertMathToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/MemRefToVM/ConvertMemRefToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
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
  explicit ConversionPass(TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, IREE::VM::VMDialect,
                    StandardOpsDialect, math::MathDialect, AffineDialect,
                    memref::MemRefDialect>();
  }

  StringRef getArgument() const override { return "iree-vm-conversion"; }

  StringRef getDescription() const override {
    return "Converts from various dialects to the VM dialect";
  }

  void runOnOperation() override {
    if (getOperation().getBody()->empty()) return;

    auto *context = &getContext();
    VMConversionTarget conversionTarget(context);
    IREE::VM::TypeConverter typeConverter(targetOptions_);

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
      if (!outerImportModuleOp) {
        innerModuleOp.emitError()
            << "unable load the VM import module for dialect '"
            << dialectInterface->getDialect()->getNamespace()
            << "'; possibly a bad file structure or malformed vm.import";
        signalPassFailure();
        return;
      }
      for (auto importModuleOp :
           outerImportModuleOp.getOps<IREE::VM::ModuleOp>()) {
        if (failed(appendImportModule(importModuleOp, innerModuleOp))) {
          importModuleOp.emitError() << "failed to import module";
          return signalPassFailure();
        }
      }
    }

    OwningRewritePatternList conversionPatterns(&getContext());
    populateIREEToVMPatterns(context, typeConverter, conversionPatterns);
    populateStandardToVMPatterns(context, typeConverter, conversionPatterns);
    populateMathToVMPatterns(context, typeConverter, conversionPatterns);
    populateMemRefToVMPatterns(context, conversionTarget, typeConverter,
                               conversionPatterns);
    populateAffineToStdConversionPatterns(conversionPatterns);
    conversionPatterns.insert<ElideTieShapeOp>(context);

    conversionTarget.addIllegalDialect<StandardOpsDialect>();
    conversionTarget.addIllegalDialect<AffineDialect>();
    conversionTarget.addIllegalDialect<math::MathDialect>();

    // Populate patterns from all used dialects, providing the imports they
    // registered earlier.
    SymbolTable importSymbols(innerModuleOp);
    for (auto *dialectInterface : usedDialects) {
      dialectInterface->populateVMConversionPatterns(
          importSymbols, conversionPatterns, typeConverter);
    }
    Shape::populateFoldConversionPatterns(context, conversionPatterns);
    IREE::Util::populatePreserveCompilerHintsPatterns(context,
                                                      conversionPatterns);
    IREE::Util::setupCompilerHintsLegality(context, conversionTarget,
                                           typeConverter);

    if (failed(applyPartialConversion(outerModuleOp, conversionTarget,
                                      std::move(conversionPatterns)))) {
      outerModuleOp.emitError() << "conversion to vm.module failed";
      return signalPassFailure();
    }
  }

 private:
  TargetOptions targetOptions_;
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass(
    TargetOptions targetOptions) {
  return std::make_unique<ConversionPass>(targetOptions);
}

static PassRegistration<ConversionPass> pass(

    [] {
      auto options = getTargetOptionsFromFlags();
      return std::make_unique<ConversionPass>(options);
    });

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
