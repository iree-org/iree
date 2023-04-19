// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

namespace {

// Given a set of types, unpack to a list of a types, removing all tuples.
void untupleTypes(TypeRange types, llvm::SmallVectorImpl<Type> &newTypes) {
  for (Type type : types) {
    if (type.isa<TupleType>()) {
      untupleTypes(type.dyn_cast<TupleType>().getTypes(), newTypes);
    } else {
      newTypes.push_back(type);
    }
  }
}

Value processTuple(Type type, Location loc, Block *block, OpBuilder &builder) {
  if (!type.isa<TupleType>()) {
    return block->addArgument(type, loc);
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
    if (oldAttr.getName() == "operand_segment_sizes" ||
        oldAttr.getName() == "result_segment_sizes")
      continue;

    newOp->setAttr(oldAttr.getName(), oldAttr.getValue());
  }
}

bool recursiveUntuple(Value value, Location loc, OpBuilder &builder,
                      IRMapping *mapping,
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
                            IRMapping *mapping) {
  for (auto operand : values) {
    auto newValue = mapping->lookupOrNull(operand);
    if (!newValue) {
      return true;
    }

    recursiveUntuple(newValue, loc, builder, mapping, newValues);
  }

  return false;
}

bool convertReturnOp(mlir::func::ReturnOp *op, OpBuilder &builder,
                     IRMapping *mapping) {
  llvm::SmallVector<Value, 10> newOperands;
  if (untupleAndLookupValues(op->getOperands(), &newOperands, builder,
                             op->getLoc(), mapping)) {
    return true;
  }

  builder.create<mlir::func::ReturnOp>(op->getLoc(), newOperands);
  return false;
}

bool convertCallOp(func::CallOp *oldOp, OpBuilder &builder,
                   IRMapping *mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  if (untupleAndLookupValues(oldOp->getOperands(), &newArgs, builder,
                             oldOp->getLoc(), mapping)) {
    return true;
  }

  SmallVector<Type, 4> resultTypes;
  untupleTypes(oldOp->getOperation()->getResultTypes(), resultTypes);
  auto newOp = builder.create<func::CallOp>(oldOp->getLoc(), oldOp->getCallee(),
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

bool convertIndirectCallOp(func::CallIndirectOp *oldOp, OpBuilder &builder,
                           IRMapping *mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  if (untupleAndLookupValues(oldOp->getOperands(), &newArgs, builder,
                             oldOp->getLoc(), mapping)) {
    return true;
  }

  auto newOp = builder.create<func::CallIndirectOp>(
      oldOp->getLoc(), oldOp->getCallee(), newArgs);
  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  for (int i = 0; i < newOp.getNumResults(); ++i) {
    auto oldResult = oldOp->getResult(i);
    auto newResult = newOp.getResult(i);
    mapping->map(oldResult, newResult);
  }

  return false;
}

bool convertBranchOp(cf::BranchOp *oldOp, OpBuilder &builder,
                     IRMapping *mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  if (untupleAndLookupValues(oldOp->getOperands(), &newArgs, builder,
                             oldOp->getLoc(), mapping)) {
    return true;
  }

  auto newOp = builder.create<cf::BranchOp>(
      oldOp->getLoc(), mapping->lookupOrNull(oldOp->getDest()), newArgs);

  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  return false;
}

bool convertCondBranchOp(cf::CondBranchOp *oldOp, OpBuilder &builder,
                         IRMapping *mapping) {
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

  auto newOp = builder.create<cf::CondBranchOp>(
      oldOp->getLoc(), mapping->lookupOrNull(oldOp->getCondition()),
      mapping->lookupOrNull(oldOp->getTrueDest()), trueArgs,
      mapping->lookupOrNull(oldOp->getFalseDest()), falseArgs);

  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  return false;
}

bool convertOperation(Operation *op, OpBuilder &builder, IRMapping *mapping) {
  if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op)) {
    return convertReturnOp(&returnOp, builder, mapping);
  } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
    return convertCallOp(&callOp, builder, mapping);
  } else if (auto callIndirectOp = dyn_cast<func::CallIndirectOp>(op)) {
    return convertIndirectCallOp(&callIndirectOp, builder, mapping);
  } else if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
    return convertBranchOp(&branchOp, builder, mapping);
  } else if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
    return convertCondBranchOp(&condBranchOp, builder, mapping);
  }

  builder.clone(*op, *mapping);
  return false;
}

bool convertFunction(func::FuncOp oldFunction, func::FuncOp newFunction) {
  OpBuilder builder(newFunction.getBody());
  IRMapping mapping;

  // Check whether has tuple in signature.
  bool hasTupleSig = (oldFunction.getArgumentTypes().size() !=
                      newFunction.getArgumentTypes().size()) ||
                     (oldFunction.getResultTypes().size() !=
                      newFunction.getResultTypes().size());

  // Cache unused XLA ABI marker names.
  auto xlaAbiParam = StringAttr::get(newFunction.getContext(),
                                     "xla_entry_computation_parameter_layouts"),
       xlaAbiLayout = StringAttr::get(newFunction.getContext(),
                                      "xla_entry_computation_result_layout");

  for (auto attr : oldFunction->getAttrs()) {
    if (attr.getName() == oldFunction.getFunctionTypeAttrName() ||
        // Currently skipping all arg, result and XLA specific ABI attributes.
        attr.getName() == xlaAbiParam || attr.getName() == xlaAbiLayout)
      continue;
    // If it has tuples in sig, then skip arg and res attrs. None of the
    // existing ones along path that produces tuples are used further, so just
    // remove instead of flattening.
    if (hasTupleSig && (attr.getName() == oldFunction.getArgAttrsAttrName() ||
                        attr.getName() == oldFunction.getResAttrsAttrName()))
      continue;
    newFunction->setAttr(attr.getName(), attr.getValue());
  }

  newFunction.getBlocks().clear();
  for (auto &oldBlock : oldFunction.getBlocks()) {
    auto *newBlock = builder.createBlock(&newFunction.getBody());
    for (auto oldArg : oldBlock.getArguments()) {
      llvm::SmallVector<Type, 4> newTypes;
      untupleTypes(oldArg.getType(), newTypes);

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
    : public FlattenTuplesInCFGBase<FlattenTuplesInCFGPass> {
 public:
  void runOnOperation() override {
    auto module = getOperation();
    Builder builder(module.getContext());

    // Build a list of (oldFunction, newFunction) for all functions we need to
    // replace. This will ensure that when we go to convert function bodies we
    // have only new functions defined.
    std::vector<std::pair<func::FuncOp, func::FuncOp>> convertedFunctions;

    for (auto oldFunction : module.getOps<func::FuncOp>()) {
      auto oldFunctionType = oldFunction.getFunctionType();

      llvm::SmallVector<Type, 10> newInputTypes;
      untupleTypes(oldFunctionType.getInputs(), newInputTypes);

      llvm::SmallVector<Type, 10> newResultTypes;
      untupleTypes(oldFunctionType.getResults(), newResultTypes);

      auto newFunctionType =
          builder.getFunctionType(newInputTypes, newResultTypes);
      auto newFunction =
          func::FuncOp::create(oldFunction.getLoc(), oldFunction.getName(),
                               newFunctionType, oldFunction->getDialectAttrs());
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

static PassRegistration<FlattenTuplesInCFGPass> pass;

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir
