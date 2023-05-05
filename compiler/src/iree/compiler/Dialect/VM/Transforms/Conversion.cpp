// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <tuple>

#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/MathToVM/ConvertMathToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/Conversion/UtilToVM/ConvertUtilToVM.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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

// TODO#(11786): The expansions of integer min and max ops were removed in
// llvm-project@e502f4fc2e25. They are added here for moving integrate forward.
// We should add native VM ops for supporting them.
template <typename OpTy, arith::CmpIPredicate pred>
struct MaxMinIOpConverter : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    Location loc = op.getLoc();
    Value cmp = rewriter.create<arith::CmpIOp>(loc, pred, lhs, rhs);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmp, lhs, rhs);
    return success();
  }
};

// Returns a stably sorted list of dialect interfaces of T for all dialects used
// within the given module.
template <typename T>
SmallVector<const T *, 4> gatherUsedDialectInterfaces(mlir::ModuleOp moduleOp) {
  SmallPtrSet<const T *, 4> resultSet;
  moduleOp.walk([&](Operation *op) {
    // Special case for declarations which may reference builtins.
    // TODO(benvanik): add a linking attribute to the module instead to avoid
    // the walk. All dialects could then indicate they want certain modules
    // linked in.
    Dialect *dialect = nullptr;
    if (auto moduleAttr = op->getAttrOfType<StringAttr>("vm.import.module")) {
      // Specified dialect lookup.
      dialect = op->getContext()->getOrLoadDialect(moduleAttr.getValue());
    } else {
      // Generic dialect lookup.
      dialect = op->getDialect();
    }
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

  StringRef getArgument() const override { return "iree-vm-conversion"; }

  StringRef getDescription() const override {
    return "Converts from various dialects to the VM dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Util::UtilDialect, IREE::VM::VMDialect, func::FuncDialect,
                mlir::arith::ArithDialect, math::MathDialect, AffineDialect>();
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

    RewritePatternSet patterns(&getContext());
    populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                   patterns);
    populateUtilToVMPatterns(context, conversionTarget, typeConverter,
                             patterns);
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    populateStandardToVMPatterns(context, typeConverter, patterns);
    populateMathToVMPatterns(context, typeConverter, patterns);
    populateAffineToStdConversionPatterns(patterns);

    conversionTarget
        .addIllegalDialect<func::FuncDialect, mlir::arith::ArithDialect>();
    conversionTarget.addIllegalDialect<AffineDialect>();
    conversionTarget.addIllegalDialect<math::MathDialect>();

    // Populate patterns from all used dialects, providing the imports they
    // registered earlier.
    SymbolTable importSymbols(innerModuleOp);
    for (auto *dialectInterface : usedDialects) {
      dialectInterface->populateVMConversionPatterns(
          importSymbols, patterns, conversionTarget, typeConverter);
    }

    if (failed(applyPartialConversion(outerModuleOp, conversionTarget,
                                      std::move(patterns)))) {
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
      auto options = TargetOptions::FromFlags::get();
      return std::make_unique<ConversionPass>(options);
    });

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
