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
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

#include "iree/compiler/Dialect/VM/Conversion/ArithToVM/Patterns.h"
#include "iree/compiler/Dialect/VM/Conversion/MathToVM/Patterns.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/Patterns.h"
#include "iree/compiler/Dialect/VM/Conversion/UtilToVM/Patterns.h"

namespace mlir::iree_compiler::IREE::VM {

#define GEN_PASS_DEF_CONVERSIONPASS
#include "iree/compiler/Dialect/VM/Transforms/Passes.h.inc"

// Returns a stably sorted list of dialect interfaces of T for all dialects used
// within the given module.
template <typename T>
static SmallVector<const T *>
gatherUsedDialectInterfaces(mlir::ModuleOp moduleOp) {
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
    if (!dialect)
      return;
    auto *dialectInterface = dialect->getRegisteredInterface<T>();
    if (!dialectInterface)
      return;
    resultSet.insert(dialectInterface);
  });

  // NOTE: to ensure deterministic output we sort the result so that imports are
  // always added in a consistent order.
  auto results = llvm::to_vector_of<const T *, 4>(resultSet);
  llvm::sort(
      results, +[](const T *a, const T *b) {
        return a->getDialect()->getNamespace().compare(
                   b->getDialect()->getNamespace()) < 0;
      });
  return results;
}

// Runs conversion with registered input dialects.
class ConversionPass
    : public IREE::VM::impl::ConversionPassBase<ConversionPass> {
  using Base::Base;
  void runOnOperation() override {
    if (getOperation().getBody()->empty())
      return;

    auto targetOptions = targetOptionsFromConversionPass();

    auto *context = &getContext();
    VMConversionTarget conversionTarget(context);
    IREE::VM::TypeConverter typeConverter(targetOptions);

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

    // Populated below after all type converters are registered.
    ImportTable importTable;

    RewritePatternSet patterns(&getContext());
    populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                   patterns);
    populateUtilToVMPatterns(context, conversionTarget, typeConverter,
                             importTable, patterns);

    conversionTarget.addIllegalDialect<affine::AffineDialect>();
    populateAffineToStdConversionPatterns(patterns);

    conversionTarget.addIllegalDialect<arith::ArithDialect>();
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    populateArithToVMPatterns(context, typeConverter, patterns);

    conversionTarget.addIllegalDialect<math::MathDialect>();
    populateMathToVMPatterns(context, typeConverter, patterns);

    conversionTarget.addIllegalDialect<func::FuncDialect>();
    populateStandardToVMPatterns(context, typeConverter, importTable, patterns);

    // Populate patterns from all used dialects, providing the imports they
    // registered earlier.
    SymbolTable importSymbols(innerModuleOp);
    for (auto *dialectInterface : usedDialects) {
      dialectInterface->populateVMConversionPatterns(
          importSymbols, patterns, conversionTarget, typeConverter);
    }

    // Build an import table so that we can quickly look up import information
    // during conversion.
    if (failed(importTable.build(innerModuleOp, typeConverter))) {
      return signalPassFailure(); // error emitted already
    }

    if (failed(applyPartialConversion(outerModuleOp, conversionTarget,
                                      std::move(patterns)))) {
      outerModuleOp.emitError() << "conversion to vm.module failed";
      return signalPassFailure();
    }
  }

  IREE::VM::TargetOptions targetOptionsFromConversionPass() {
    IREE::VM::TargetOptions targetOptions;
    targetOptions.indexBits = indexBits;
    targetOptions.f32Extension = f32Extension;
    targetOptions.f64Extension = f64Extension;
    targetOptions.truncateUnsupportedFloats = truncateUnsupportedFloats;
    targetOptions.optimizeForStackSize = optimizeForStackSize;
    return targetOptions;
  }
};

} // namespace mlir::iree_compiler::IREE::VM
