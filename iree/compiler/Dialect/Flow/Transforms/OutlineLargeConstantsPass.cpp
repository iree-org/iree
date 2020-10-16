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

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// NOTE: a total guess :) this feels like about the most per-dispatch-buffer
// data we'd want to embed in the command buffer.
// TODO(benvanik): make a pass option so users can override.
static constexpr size_t kMinLargeConstantSize = 256;

// Returns true if |constantOp| is large enough to be considered for pooling.
// Some constants are small enough that inlining them into the ringbuffer is
// more efficient and fewer bindings.
static bool isConstantLarge(ConstantOp constantOp) {
  auto type = constantOp.getType();
  if (auto shapedType = type.dyn_cast<RankedTensorType>()) {
    size_t unpackedByteLength =
        (shapedType.getNumElements() * shapedType.getElementTypeBitWidth()) / 8;
    if (unpackedByteLength >= kMinLargeConstantSize) {
      return true;
    }
  }
  return false;
}

// Returns a list of all large constants in the module.
// Only walks top-level functions and ops to avoid pulling constants out of
// executables.
static std::vector<ConstantOp> findLargeConstantsInModule(ModuleOp moduleOp) {
  std::vector<ConstantOp> largeConstantOps;
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    for (auto &block : funcOp.getBlocks()) {
      for (auto constantOp : block.getOps<ConstantOp>()) {
        if (isConstantLarge(constantOp)) {
          largeConstantOps.push_back(constantOp);
        }
      }
    }
  }
  return largeConstantOps;
}

class OutlineLargeConstantsPass
    : public PassWrapper<OutlineLargeConstantsPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // For name uniquing.
    SymbolTable moduleSymbols(moduleOp);
    std::string baseName = "_large_const_";
    int uniqueId = 0;

    // Create all top-level flow.variables from large constants in the module.
    OpBuilder moduleBuilder(&moduleOp.getBody()->front());
    std::vector<std::pair<ConstantOp, IREE::Flow::VariableOp>> replacements;
    for (auto &largeConstantOp : findLargeConstantsInModule(moduleOp)) {
      std::string name;
      do {
        name = baseName + std::to_string(uniqueId++);
      } while (moduleSymbols.lookup(name) != nullptr);
      auto variableOp = moduleBuilder.create<IREE::Flow::VariableOp>(
          largeConstantOp.getLoc(), name, /*isMutable=*/false,
          largeConstantOp.getType(), largeConstantOp.getValue());
      SymbolTable::setSymbolVisibility(variableOp,
                                       SymbolTable::Visibility::Private);
      replacements.emplace_back(largeConstantOp, variableOp);

      // Prevent the variable from being re-inlined if the canonicalizer runs.
      // By the time we've outlined things here we are sure we want them
      // outlined even if the user runs an arbitrary number of passes between
      // now and when we may use that information (HAL constant pooling, etc).
      variableOp.setAttr("noinline", moduleBuilder.getUnitAttr());
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
};

std::unique_ptr<OperationPass<ModuleOp>> createOutlineLargeConstantsPass() {
  return std::make_unique<OutlineLargeConstantsPass>();  // NOLINT
}

static PassRegistration<OutlineLargeConstantsPass> pass(
    "iree-flow-outline-large-constants",
    "Outlines large tensor constants into flow.variables at the module level.");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
