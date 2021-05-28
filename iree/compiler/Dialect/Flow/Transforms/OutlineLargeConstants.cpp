// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Returns true if |constantOp| is large enough to be considered for pooling.
// Some constants are small enough that inlining them into the ringbuffer is
// more efficient and fewer bindings.
static bool isConstantLarge(ConstantOp constantOp,
                            size_t minLargeConstantSize) {
  auto type = constantOp.getType();
  if (auto shapedType = type.dyn_cast<RankedTensorType>()) {
    size_t unpackedByteLength =
        (shapedType.getNumElements() * shapedType.getElementTypeBitWidth()) / 8;
    if (unpackedByteLength >= minLargeConstantSize) {
      return true;
    }
  }
  return false;
}

// Returns a list of all large constants in the module.
// Only walks top-level functions and ops to avoid pulling constants out of
// executables.
static std::vector<ConstantOp> findLargeConstantsInModule(
    ModuleOp moduleOp, size_t minLargeConstantSize) {
  std::vector<ConstantOp> largeConstantOps;
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    for (auto &block : funcOp.getBlocks()) {
      for (auto constantOp : block.getOps<ConstantOp>()) {
        if (isConstantLarge(constantOp, minLargeConstantSize)) {
          largeConstantOps.push_back(constantOp);
        }
      }
    }
  }
  return largeConstantOps;
}

class OutlineLargeConstantsPass
    : public OutlineLargeConstantsBase<OutlineLargeConstantsPass> {
 public:
  OutlineLargeConstantsPass() = default;
  OutlineLargeConstantsPass(size_t minLargeConstantSize)
      : minLargeConstantSize(minLargeConstantSize){};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) return;

    // For name uniquing.
    SymbolTable moduleSymbols(moduleOp);
    std::string baseName = "_large_const_";
    int uniqueId = 0;

    // Create all top-level flow.variables from large constants in the module.
    OpBuilder moduleBuilder(&moduleOp.getBody()->front());
    std::vector<std::pair<ConstantOp, IREE::Flow::VariableOp>> replacements;
    for (auto &largeConstantOp :
         findLargeConstantsInModule(moduleOp, minLargeConstantSize)) {
      std::string name;
      do {
        name = baseName + std::to_string(uniqueId++);
      } while (moduleSymbols.lookup(name) != nullptr);
      auto variableOp = moduleBuilder.create<IREE::Flow::VariableOp>(
          largeConstantOp.getLoc(), name, /*isMutable=*/false,
          largeConstantOp.getType(), largeConstantOp.getValue());
      variableOp.setPrivate();
      replacements.emplace_back(largeConstantOp, variableOp);

      // Prevent the variable from being re-inlined if the canonicalizer runs.
      // By the time we've outlined things here we are sure we want them
      // outlined even if the user runs an arbitrary number of passes between
      // now and when we may use that information (HAL constant pooling, etc).
      variableOp->setAttr("noinline", moduleBuilder.getUnitAttr());
    }

    // Replace all of the constants with lookups for the new variables.
    for (auto pair : replacements) {
      auto constantOp = pair.first;
      auto variableOp = pair.second;
      OpBuilder builder(moduleOp.getContext());
      builder.setInsertionPoint(constantOp);
      auto lookupOp = builder.create<IREE::Flow::VariableLoadOp>(
          constantOp.getLoc(), constantOp.getType(), variableOp.getName());
      constantOp.getResult().replaceAllUsesWith(lookupOp);
      constantOp.erase();
    }
  }

 private:
  size_t minLargeConstantSize;
};

std::unique_ptr<OperationPass<ModuleOp>> createOutlineLargeConstantsPass(
    size_t minLargeConstantSize) {
  return std::make_unique<OutlineLargeConstantsPass>(minLargeConstantSize);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
