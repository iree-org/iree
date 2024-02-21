// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Util {

// Returns true if |value| is worth outlining (large, etc).
static bool isOutlinableValue(Attribute value) {
  if (auto elementsAttr = llvm::dyn_cast<ElementsAttr>(value)) {
    // Don't outline splats - we want those fused.
    return !elementsAttr.isSplat();
  }
  return false;
}

struct ConstantDef {
  Operation *op;
  Type type;
  ElementsAttr value;
};

// Returns a list of all constant-like shaped data ops in the module.
static SmallVector<ConstantDef> findConstantsInModule(mlir::ModuleOp moduleOp) {
  SmallVector<ConstantDef> results;
  for (auto callableOp : moduleOp.getOps<CallableOpInterface>()) {
    auto *region = callableOp.getCallableRegion();
    if (!region)
      continue;
    for (auto &block : *region) {
      for (auto &op : block.getOperations()) {
        if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
          if (isOutlinableValue(constantOp.getValue())) {
            results.push_back(ConstantDef{
                constantOp,
                constantOp.getType(),
                llvm::cast<ElementsAttr>(constantOp.getValue()),
            });
          }
        }
      }
    }
  }
  return results;
}

class OutlineConstantsPass : public OutlineConstantsBase<OutlineConstantsPass> {
public:
  OutlineConstantsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    SymbolTable moduleSymbols(moduleOp);
    std::string baseName = "_constant";

    // Create all top-level util.globals from constants in the module.
    OpBuilder moduleBuilder(&moduleOp.getBody()->front());
    std::vector<std::pair<Operation *, IREE::Util::GlobalOp>> replacements;
    for (auto &def : findConstantsInModule(moduleOp)) {
      // New immutable global takes the constant attribute in its specified
      // encoding.
      auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
          def.op->getLoc(), baseName, /*isMutable=*/false, def.type, def.value);
      globalOp.setPrivate();
      moduleSymbols.insert(globalOp); // uniques name
      replacements.emplace_back(def.op, globalOp);

      // Prevent the variable from being re-inlined if the canonicalizer runs.
      // By the time we've outlined things here we are sure we want them
      // outlined even if the user runs an arbitrary number of passes between
      // now and when we may use that information (HAL constant pooling, etc).
      globalOp.setInliningPolicyAttr(
          moduleBuilder.getAttr<IREE::Util::InlineNeverAttr>());
    }

    // Replace all of the constants with lookups for the new variables.
    for (auto pair : replacements) {
      auto *originalOp = pair.first;
      auto globalOp = pair.second;
      OpBuilder builder(moduleOp.getContext());
      builder.setInsertionPoint(originalOp);
      auto loadOp = globalOp.createLoadOp(originalOp->getLoc(), builder);

      Value replacement;
      if (auto constantOp = dyn_cast<arith::ConstantOp>(originalOp)) {
        // Directly replace constant with global constant value.
        replacement = loadOp.getLoadedGlobalValue();
      } else {
        assert(false && "unhandled constant op type");
      }

      originalOp->getResult(0).replaceAllUsesWith(replacement);
      originalOp->erase();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createOutlineConstantsPass() {
  return std::make_unique<OutlineConstantsPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
