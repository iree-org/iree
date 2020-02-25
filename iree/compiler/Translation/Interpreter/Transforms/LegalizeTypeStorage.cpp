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

#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

bool convertOperation(Operation *oldOp, OpBuilder &builder,
                      BlockAndValueMapping *mapping) {
  OperationState state(oldOp->getLoc(), oldOp->getName());
  if (oldOp->getNumSuccessors() == 0) {
    // Non-branching operations can just add all the operands.
    for (auto oldOperand : oldOp->getOperands()) {
      state.operands.push_back(mapping->lookupOrDefault(oldOperand));
    }
  } else {
    // We add the operands separated by nullptr's for each successor.
    unsigned firstSuccOperand = oldOp->getNumSuccessors()
                                    ? oldOp->getSuccessorOperandIndex(0)
                                    : oldOp->getNumOperands();
    auto opOperands = oldOp->getOpOperands();
    unsigned i = 0;
    for (; i != firstSuccOperand; ++i) {
      state.operands.push_back(mapping->lookupOrDefault(opOperands[i].get()));
    }
    for (unsigned succ = 0, e = oldOp->getNumSuccessors(); succ != e; ++succ) {
      state.successors.push_back(
          mapping->lookupOrDefault(oldOp->getSuccessor(succ)));
      // Add sentinel to delineate successor operands.
      state.operands.push_back(nullptr);
      // Remap the successors operands.
      for (auto operand : oldOp->getSuccessorOperands(succ)) {
        state.operands.push_back(mapping->lookupOrDefault(operand));
      }
    }
  }
  for (const auto &oldType : oldOp->getResultTypes()) {
    state.types.push_back(legalizeLegacyType(oldType));
  }
  state.attributes = {oldOp->getAttrs().begin(), oldOp->getAttrs().end()};
  auto newOp = builder.createOperation(state);
  for (int i = 0; i < newOp->getNumResults(); ++i) {
    mapping->map(oldOp->getResult(i), newOp->getResult(i));
  }
  return false;
}

bool convertFunction(FuncOp oldFunction, FuncOp newFunction) {
  OpBuilder builder(newFunction.getBody());
  BlockAndValueMapping mapping;

  // Create new blocks matching the expected arguments of the old ones.
  // This sets up the block mappings to enable us to reference blocks forward
  // during conversion.
  newFunction.getBlocks().clear();
  for (auto &oldBlock : oldFunction.getBlocks()) {
    auto *newBlock = builder.createBlock(&newFunction.getBody());
    mapping.map(&oldBlock, newBlock);
    for (auto oldArg : oldBlock.getArguments()) {
      auto newArg = newBlock->addArgument(legalizeLegacyType(oldArg.getType()));
      mapping.map(oldArg, newArg);
    }
  }

  // Convert all ops in the blocks.
  for (auto &oldBlock : oldFunction.getBlocks()) {
    builder.setInsertionPointToEnd(mapping.lookupOrNull(&oldBlock));
    for (auto &oldOp : oldBlock.getOperations()) {
      if (convertOperation(&oldOp, builder, &mapping)) {
        return true;
      }
    }
  }

  return false;
}

}  // namespace

class LegalizeTypeStoragePass : public ModulePass<LegalizeTypeStoragePass> {
 public:
  void runOnModule() override {
    auto module = getModule();

    // Build a list of (oldFunction, newFunction) for all functions we need to
    // replace. This will ensure that when we go to convert function bodies we
    // have only new functions defined.
    std::vector<std::pair<FuncOp, FuncOp>> convertedFunctions;

    for (auto oldFunction : module.getOps<FuncOp>()) {
      // Create the replacement function, ensuring that we copy attributes.
      auto newFunction = FuncOp::create(
          oldFunction.getLoc(), oldFunction.getName(),
          legalizeLegacyType(oldFunction.getType()).cast<FunctionType>(),
          oldFunction.getDialectAttrs());
      convertedFunctions.push_back({oldFunction, newFunction});

      // Perform the actual body conversion now that we have proper signatures.
      if (convertFunction(oldFunction, newFunction)) {
        return signalPassFailure();
      }
    }

    // Replace functions in the module.
    for (auto &pair : convertedFunctions) {
      pair.first.erase();
      module.push_back(pair.second);
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createLegalizeTypeStoragePass() {
  return std::make_unique<LegalizeTypeStoragePass>();
}

static PassRegistration<LegalizeTypeStoragePass> pass(
    "iree-legalize-type-storage",
    "Legalizes types to ones supported by the IREE VM.");

}  // namespace iree_compiler
}  // namespace mlir
