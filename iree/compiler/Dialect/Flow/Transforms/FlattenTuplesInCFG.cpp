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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Given a set of types, unpack to a list of a types, removing all tuples.
void untupleTypes(llvm::ArrayRef<Type> types,
                  llvm::SmallVectorImpl<Type> *newTypes) {
  for (auto &type : types) {
    if (type.isa<TupleType>()) {
      untupleTypes(type.dyn_cast<TupleType>().getTypes(), newTypes);
    } else {
      newTypes->push_back(type);
    }
  }
}

Value processTuple(Type type, Location loc, Block *block, OpBuilder &builder) {
  if (!type.isa<TupleType>()) {
    return block->addArgument(type);
  }

  auto tupleType = type.dyn_cast<TupleType>();
  llvm::SmallVector<Value, 4> values;
  values.reserve(tupleType.size());
  for (auto subtype : tupleType.getTypes()) {
    values.push_back(processTuple(subtype, loc, block, builder));
  }

  return builder.create<mhlo::TupleOp>(loc, tupleType, values);
}

void copyOperationAttrs(Operation *oldOp, Operation *newOp) {
  for (const auto &oldAttr : oldOp->getAttrs()) {
    // Don't copy segment attributes as these correspond to the number operands,
    // which may be different.
    if (oldAttr.first == "operand_segment_sizes" ||
        oldAttr.first == "result_segment_sizes")
      continue;

    newOp->setAttr(oldAttr.first, oldAttr.second);
  }
}

bool recursiveUntuple(Value value, Location loc, OpBuilder &builder,
                      BlockAndValueMapping *mapping,
                      llvm::SmallVectorImpl<Value> *newValues) {
  Type type = value.getType();
  // We can return the value as is.
  if (!type.isa<TupleType>()) {
    newValues->push_back(value);
    return false;
  }

  TupleType tupleType = type.dyn_cast<TupleType>();
  for (int i = 0; i < tupleType.size(); i++) {
    auto subType = tupleType.getType(i);

    auto elementOp = builder.create<mhlo::GetTupleElementOp>(
        loc, subType, value, builder.getI32IntegerAttr(i));
    recursiveUntuple(elementOp.getResult(), loc, builder, mapping, newValues);
  }

  return false;
}

Value recursiveRetuple(Type oldType, Operation::result_range *values,
                       OpBuilder &builder, Location loc) {
  if (!oldType.isa<TupleType>()) {
    Value returnValue = *values->begin();
    *values = {values->begin() + 1, values->end()};
    return returnValue;
  }

  TupleType tupleType = oldType.dyn_cast<TupleType>();
  llvm::SmallVector<Value, 10> subValues;
  for (auto subtype : tupleType.getTypes()) {
    subValues.push_back(recursiveRetuple(subtype, values, builder, loc));
  }

  return builder.create<mhlo::TupleOp>(loc, tupleType, subValues).getResult();
}

template <typename T>
bool untupleAndLookupValues(T values, llvm::SmallVectorImpl<Value> *newValues,
                            OpBuilder &builder, Location loc,
                            BlockAndValueMapping *mapping) {
  for (auto operand : values) {
    auto newValue = mapping->lookupOrNull(operand);
    if (!newValue) {
      return true;
    }

    recursiveUntuple(newValue, loc, builder, mapping, newValues);
  }

  return false;
}

bool convertReturnOp(mlir::ReturnOp *op, OpBuilder &builder,
                     BlockAndValueMapping *mapping) {
  llvm::SmallVector<Value, 10> newOperands;
  if (untupleAndLookupValues(op->getOperands(), &newOperands, builder,
                             op->getLoc(), mapping)) {
    return true;
  }

  builder.create<mlir::ReturnOp>(op->getLoc(), newOperands);
  return false;
}

bool convertCallOp(CallOp *oldOp, OpBuilder &builder,
                   BlockAndValueMapping *mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  if (untupleAndLookupValues(oldOp->getOperands(), &newArgs, builder,
                             oldOp->getLoc(), mapping)) {
    return true;
  }

  SmallVector<Type, 4> resultTypes;
  untupleTypes(oldOp->getOperation()->getResultTypes(), &resultTypes);
  auto newOp = builder.create<CallOp>(oldOp->getLoc(), oldOp->getCallee(),
                                      resultTypes, newArgs);
  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  auto newResults = newOp.getResults();
  for (auto oldResult : oldOp->getResults()) {
    llvm::SmallVector<Value, 10> subValues;
    auto newResult = recursiveRetuple(oldResult.getType(), &newResults, builder,
                                      oldOp->getLoc());
    mapping->map(oldResult, newResult);
  }

  return false;
}

bool convertIndirectCallOp(CallIndirectOp *oldOp, OpBuilder &builder,
                           BlockAndValueMapping *mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  if (untupleAndLookupValues(oldOp->getOperands(), &newArgs, builder,
                             oldOp->getLoc(), mapping)) {
    return true;
  }

  auto newOp = builder.create<CallIndirectOp>(oldOp->getLoc(),
                                              oldOp->getCallee(), newArgs);
  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  for (int i = 0; i < newOp.getNumResults(); ++i) {
    auto oldResult = oldOp->getResult(i);
    auto newResult = newOp.getResult(i);
    mapping->map(oldResult, newResult);
  }

  return false;
}

bool convertBranchOp(BranchOp *oldOp, OpBuilder &builder,
                     BlockAndValueMapping *mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  if (untupleAndLookupValues(oldOp->getOperands(), &newArgs, builder,
                             oldOp->getLoc(), mapping)) {
    return true;
  }

  auto newOp = builder.create<BranchOp>(
      oldOp->getLoc(), mapping->lookupOrNull(oldOp->getDest()), newArgs);

  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  return false;
}

bool convertCondBranchOp(CondBranchOp *oldOp, OpBuilder &builder,
                         BlockAndValueMapping *mapping) {
  llvm::SmallVector<Value, 4> trueArgs;
  if (untupleAndLookupValues(oldOp->getTrueOperands(), &trueArgs, builder,
                             oldOp->getLoc(), mapping)) {
    return true;
  }

  llvm::SmallVector<Value, 4> falseArgs;
  if (untupleAndLookupValues(oldOp->getFalseOperands(), &falseArgs, builder,
                             oldOp->getLoc(), mapping)) {
    return true;
  }

  auto newOp = builder.create<CondBranchOp>(
      oldOp->getLoc(), mapping->lookupOrNull(oldOp->getCondition()),
      mapping->lookupOrNull(oldOp->getTrueDest()), trueArgs,
      mapping->lookupOrNull(oldOp->getFalseDest()), falseArgs);

  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  return false;
}

bool convertOperation(Operation *op, OpBuilder &builder,
                      BlockAndValueMapping *mapping) {
  if (auto returnOp = dyn_cast<mlir::ReturnOp>(op)) {
    return convertReturnOp(&returnOp, builder, mapping);
  } else if (auto callOp = dyn_cast<CallOp>(op)) {
    return convertCallOp(&callOp, builder, mapping);
  } else if (auto callIndirectOp = dyn_cast<CallIndirectOp>(op)) {
    return convertIndirectCallOp(&callIndirectOp, builder, mapping);
  } else if (auto branchOp = dyn_cast<BranchOp>(op)) {
    return convertBranchOp(&branchOp, builder, mapping);
  } else if (auto condBranchOp = dyn_cast<CondBranchOp>(op)) {
    return convertCondBranchOp(&condBranchOp, builder, mapping);
  }

  builder.clone(*op, *mapping);
  return false;
}

bool convertFunction(FuncOp oldFunction, FuncOp newFunction) {
  OpBuilder builder(newFunction.getBody());
  BlockAndValueMapping mapping;

  for (auto attr : oldFunction.getAttrs()) {
    if (attr.first != oldFunction.getTypeAttrName()) {
      newFunction.setAttr(attr.first, attr.second);
    }
  }

  newFunction.getBlocks().clear();
  for (auto &oldBlock : oldFunction.getBlocks()) {
    auto *newBlock = builder.createBlock(&newFunction.getBody());
    for (auto oldArg : oldBlock.getArguments()) {
      llvm::SmallVector<Type, 4> newTypes;
      untupleTypes(oldArg.getType(), &newTypes);

      Value newTuple = processTuple(oldArg.getType(), oldFunction.getLoc(),
                                    newBlock, builder);
      if (!newTuple) {
        return true;
      }

      mapping.map(oldArg, newTuple);
    }
    mapping.map(&oldBlock, newBlock);
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

class FlattenTuplesInCFGPass
    : public PassWrapper<FlattenTuplesInCFGPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    auto module = getOperation();
    Builder builder(module.getContext());

    // Build a list of (oldFunction, newFunction) for all functions we need to
    // replace. This will ensure that when we go to convert function bodies we
    // have only new functions defined.
    std::vector<std::pair<FuncOp, FuncOp>> convertedFunctions;

    for (auto oldFunction : module.getOps<FuncOp>()) {
      auto oldFunctionType = oldFunction.getType();
      llvm::SmallVector<Type, 10> newInputTypes;
      untupleTypes(oldFunctionType.getInputs(), &newInputTypes);

      llvm::SmallVector<Type, 10> newResultTypes;
      untupleTypes(oldFunctionType.getResults(), &newResultTypes);

      auto newFunctionType =
          builder.getFunctionType(newInputTypes, newResultTypes);
      auto newFunction =
          FuncOp::create(oldFunction.getLoc(), oldFunction.getName(),
                         newFunctionType, oldFunction.getDialectAttrs());
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

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createFlattenTuplesInCFGPass() {
  return std::make_unique<FlattenTuplesInCFGPass>();
}

static PassRegistration<FlattenTuplesInCFGPass> pass(
    "iree-flow-flatten-tuples-in-cfg",
    "Convert functions to remove tuples from method signatures and blocks");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
