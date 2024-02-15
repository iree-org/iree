// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {

struct LLVMCPUAssignImportOrdinalsPass
    : public LLVMCPUAssignImportOrdinalsBase<LLVMCPUAssignImportOrdinalsPass> {
  LLVMCPUAssignImportOrdinalsPass() = default;
  void runOnOperation() override {
    auto variantOp = getOperation();

    auto *context = variantOp.getContext();
    auto unitAttr = UnitAttr::get(context);
    auto importKeyAttr = StringAttr::get(context, "hal.executable.import.key");
    auto importWeakAttr =
        StringAttr::get(context, "hal.executable.import.weak");

    // Scan the module for the used imports and sort by name.
    // This allows us to assign ordinals deterministically regardless of what
    // order the imports are declared which may not always be stable.
    SetVector<StringAttr> uniqueKeys;
    DenseMap<StringAttr, SmallVector<LLVM::GlobalOp>> ordinalGlobals;
    auto moduleOp = variantOp.getInnerModule();
    for (auto globalOp :
         llvm::make_early_inc_range(moduleOp.getOps<LLVM::GlobalOp>())) {
      auto keyAttr = globalOp->getAttrOfType<StringAttr>(importKeyAttr);
      if (!keyAttr)
        continue;
      uniqueKeys.insert(keyAttr);
      ordinalGlobals[keyAttr].push_back(globalOp);
    }
    if (uniqueKeys.empty())
      return;
    auto sortedKeys = uniqueKeys.takeVector();
    llvm::stable_sort(sortedKeys, [](auto lhs, auto rhs) {
      return lhs.getValue() < rhs.getValue();
    });

    // Build the attribute used during serialization to emit the import table.
    SmallVector<Attribute> importAttrs;
    for (auto keyAttr : sortedKeys) {
      auto anyGlobalOp = ordinalGlobals[keyAttr].front();
      bool isWeak = anyGlobalOp->hasAttr(importWeakAttr);
      auto isWeakAttr = BoolAttr::get(context, isWeak);
      importAttrs.push_back(ArrayAttr::get(context, {keyAttr, isWeakAttr}));
    }
    variantOp->setAttr("hal.executable.imports",
                       ArrayAttr::get(context, importAttrs));

    // Update placeholders to hold the concrete ordinal values.
    // Eventually MLIR or LLVM will inline them.
    for (auto [ordinal, keyAttr] : llvm::enumerate(sortedKeys)) {
      for (auto globalOp : ordinalGlobals[keyAttr]) {
        globalOp->removeAttr(importKeyAttr);
        globalOp->removeAttr(importWeakAttr);
        globalOp.setConstantAttr(unitAttr);
        globalOp.setValueAttr(IntegerAttr::get(
            IntegerType::get(globalOp.getContext(), 32), ordinal));
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUAssignImportOrdinalsPass() {
  return std::make_unique<LLVMCPUAssignImportOrdinalsPass>();
}

} // namespace mlir::iree_compiler
