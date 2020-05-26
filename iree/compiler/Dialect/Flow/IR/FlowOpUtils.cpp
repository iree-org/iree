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

#include "iree/compiler/Dialect/Flow/IR/FlowOpUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

Operation *cloneWithNewResultTypes(Operation *op, TypeRange newResultTypes) {
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(op->getOperands());
  state.addTypes(newResultTypes);
  state.addSuccessors(op->getSuccessors());
  state.addAttributes(op->getAttrs());
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    state.addRegion();
  }
  Operation *newOp = Operation::create(state);
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    newOp->getRegion(i).takeBody(op->getRegion(i));
  }
  return newOp;
}

//------------------------------------------------------------------------------
// ClosureOpDce
//------------------------------------------------------------------------------

ClosureOpDce::ClosureOpDce(Operation *closureOp, Block &entryBlock,
                           unsigned variadicOffset)
    : closureOp(closureOp),
      entryBlock(entryBlock),
      variadicOffset(variadicOffset),
      blockArgReplacements(entryBlock.getNumArguments()) {
  assert(closureOp->getNumOperands() ==
         entryBlock.getNumArguments() + variadicOffset);

  // Build data structure for unused operand elision.
  for (auto it : llvm::enumerate(entryBlock.getArguments())) {
    BlockArgument blockArg = it.value();
    Value opArg = closureOp->getOperand(it.index() + variadicOffset);
    if (blockArg.getUses().empty()) {
      // Not used - Drop.
      needsOperandElision = true;
      blockArgReplacements[it.index()] = BlockArgument();
      continue;
    }
    auto existingIt = argToBlockMap.find(opArg);
    if (existingIt == argToBlockMap.end()) {
      // Not found - Record for deduping.
      argToBlockMap.insert(std::make_pair(opArg, blockArg));
    } else {
      // Found - Replace.
      needsOperandElision = true;
      blockArgReplacements[it.index()] = existingIt->second;
    }
  }

  // Check for unused results.
  for (auto result : closureOp->getResults()) {
    if (result.getUses().empty()) {
      needsResultElision = true;
      break;
    }
  }
}

void ClosureOpDce::elideUnusedOperands(OpBuilder &builder) {
  llvm::SmallVector<Value, 8> newOperands(
      closureOp->operand_begin(), closureOp->operand_begin() + variadicOffset);
  unsigned blockArgIndex = 0;
  for (auto it : llvm::enumerate(blockArgReplacements)) {
    llvm::Optional<BlockArgument> replacement = it.value();
    Value currentOpArg = closureOp->getOperand(it.index() + variadicOffset);
    if (!replacement) {
      // No change.
      newOperands.push_back(currentOpArg);
      blockArgIndex++;
      continue;
    } else if (!replacement.getValue()) {
      // Drop.
      entryBlock.eraseArgument(blockArgIndex);
      continue;
    } else {
      // Replace.
      BlockArgument currentBlockArg = entryBlock.getArgument(blockArgIndex);
      currentBlockArg.replaceAllUsesWith(*replacement);
      entryBlock.eraseArgument(blockArgIndex);
    }
  }

  closureOp->setOperands(newOperands);
}

void ClosureOpDce::elideUnusedResults(OpBuilder &builder, bool eraseOriginal) {
  // Determine the result signature transform needed.
  llvm::SmallVector<unsigned, 4> resultIndexMap;
  llvm::SmallVector<Type, 4> newResultTypes;
  for (auto it : llvm::enumerate(closureOp->getResults())) {
    if (!it.value().getUses().empty()) {
      newResultTypes.push_back(it.value().getType());
      resultIndexMap.push_back(it.index());
    }
  }

  // Re-allocate the op.
  builder.setInsertionPoint(closureOp);
  Operation *newOp =
      builder.insert(cloneWithNewResultTypes(closureOp, newResultTypes));

  // Remap all returns.
  llvm::SmallVector<Value, 4> newReturns(resultIndexMap.size());
  newOp->walk([&](IREE::Flow::ReturnOp terminator) {
    for (unsigned i = 0, e = resultIndexMap.size(); i < e; ++i) {
      newReturns[i] = terminator.getOperand(resultIndexMap[i]);
    }
    terminator.getOperation()->setOperands(newReturns);
  });

  // Replace original uses.
  for (unsigned i = 0, e = resultIndexMap.size(); i < e; ++i) {
    closureOp->getResult(resultIndexMap[i])
        .replaceAllUsesWith(newOp->getResult(i));
  }
  if (eraseOriginal) closureOp->erase();
  closureOp = newOp;
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
