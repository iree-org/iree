// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements IREE-specific preprocessing for XLA inputs.

#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h"
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Rewriters.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_FLATTENTUPLESINCFG
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h.inc"

namespace {
// Given a set of types, unpack to a list of a types, removing all tuples.
void untupleTypes(TypeRange types, llvm::SmallVectorImpl<Type> &newTypes) {
  for (Type type : types) {
    if (auto tupleTy = dyn_cast<TupleType>(type)) {
      untupleTypes(tupleTy.getTypes(), newTypes);
    } else {
      newTypes.push_back(type);
    }
  }
}

Value processTuple(Type type, Location loc, Block &block, OpBuilder &builder) {
  auto tupleType = dyn_cast<TupleType>(type);
  if (!tupleType) {
    return block.addArgument(type, loc);
  }

  llvm::SmallVector<Value, 4> values;
  values.reserve(tupleType.size());
  for (Type subtype : tupleType.getTypes()) {
    values.push_back(processTuple(subtype, loc, block, builder));
  }

  return builder.create<mlir::stablehlo::TupleOp>(loc, tupleType, values);
}

void copyOperationAttrs(Operation *oldOp, Operation *newOp) {
  for (NamedAttribute oldAttr : oldOp->getAttrs()) {
    // Don't copy segment attributes as these correspond to the number operands,
    // which may be different.
    if (oldAttr.getName() == "operand_segment_sizes" ||
        oldAttr.getName() == "result_segment_sizes")
      continue;

    newOp->setAttr(oldAttr.getName(), oldAttr.getValue());
  }
}

void recursiveUntuple(Value value, Location loc, OpBuilder &builder,
                      IRMapping &mapping,
                      llvm::SmallVectorImpl<Value> &newValues) {
  auto tupleType = dyn_cast<TupleType>(value.getType());
  if (!tupleType) {
    // We can return the value as is.
    newValues.push_back(value);
    return;
  }

  for (auto [idx, subType] : llvm::enumerate(tupleType.getTypes())) {
    auto elementOp = builder.create<mlir::stablehlo::GetTupleElementOp>(
        loc, subType, value, builder.getI32IntegerAttr(idx));
    recursiveUntuple(elementOp.getResult(), loc, builder, mapping, newValues);
  }
}

Value recursiveRetuple(Type oldType, Operation::result_range *values,
                       OpBuilder &builder, Location loc) {
  auto tupleType = dyn_cast<TupleType>(oldType);
  if (!tupleType) {
    Value returnValue = *values->begin();
    *values = {values->begin() + 1, values->end()};
    return returnValue;
  }

  llvm::SmallVector<Value> subValues;
  for (Type subType : tupleType.getTypes()) {
    subValues.push_back(recursiveRetuple(subType, values, builder, loc));
  }

  return builder.create<mlir::stablehlo::TupleOp>(loc, tupleType, subValues)
      .getResult();
}

template <typename T>
LogicalResult untupleAndLookupValues(T values,
                                     llvm::SmallVectorImpl<Value> &newValues,
                                     OpBuilder &builder, Location loc,
                                     IRMapping &mapping) {
  for (auto operand : values) {
    auto newValue = mapping.lookupOrNull(operand);
    if (!newValue) {
      return failure();
    }

    recursiveUntuple(newValue, loc, builder, mapping, newValues);
  }

  return success();
}

LogicalResult convertOp(mlir::func::ReturnOp op, OpBuilder &builder,
                        IRMapping &mapping) {
  llvm::SmallVector<Value> newOperands;
  if (failed(untupleAndLookupValues(op.getOperands(), newOperands, builder,
                                    op.getLoc(), mapping))) {
    return failure();
  }

  builder.create<mlir::func::ReturnOp>(op->getLoc(), newOperands);
  return success();
}

LogicalResult convertOp(func::CallOp oldOp, OpBuilder &builder,
                        IRMapping &mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  if (failed(untupleAndLookupValues(oldOp.getOperands(), newArgs, builder,
                                    oldOp.getLoc(), mapping))) {
    return failure();
  }

  SmallVector<Type, 4> resultTypes;
  untupleTypes(oldOp.getResultTypes(), resultTypes);
  auto newOp = builder.create<func::CallOp>(oldOp->getLoc(), oldOp.getCallee(),
                                            resultTypes, newArgs);
  copyOperationAttrs(oldOp, newOp);

  auto newResults = newOp.getResults();
  for (auto oldResult : oldOp.getResults()) {
    llvm::SmallVector<Value, 10> subValues;
    auto newResult = recursiveRetuple(oldResult.getType(), &newResults, builder,
                                      oldOp->getLoc());
    mapping.map(oldResult, newResult);
  }

  return success();
}

LogicalResult convertOp(func::CallIndirectOp oldOp, OpBuilder &builder,
                        IRMapping &mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  if (failed(untupleAndLookupValues(oldOp.getOperands(), newArgs, builder,
                                    oldOp.getLoc(), mapping))) {
    return failure();
  }

  auto newOp = builder.create<func::CallIndirectOp>(oldOp.getLoc(),
                                                    oldOp.getCallee(), newArgs);
  copyOperationAttrs(oldOp, newOp);

  for (auto [oldResult, newResult] :
       llvm::zip_equal(oldOp.getResults(), newOp.getResults())) {
    mapping.map(oldResult, newResult);
  }

  return success();
}

LogicalResult convertOp(cf::BranchOp oldOp, OpBuilder &builder,
                        IRMapping &mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  if (failed(untupleAndLookupValues(oldOp.getOperands(), newArgs, builder,
                                    oldOp.getLoc(), mapping))) {
    return failure();
  }

  auto newOp = builder.create<cf::BranchOp>(
      oldOp.getLoc(), mapping.lookupOrNull(oldOp.getDest()), newArgs);

  copyOperationAttrs(oldOp, newOp);
  return success();
}

LogicalResult convertOp(cf::CondBranchOp oldOp, OpBuilder &builder,
                        IRMapping &mapping) {
  llvm::SmallVector<Value, 4> trueArgs;
  if (failed(untupleAndLookupValues(oldOp.getTrueOperands(), trueArgs, builder,
                                    oldOp.getLoc(), mapping))) {
    return failure();
  }

  llvm::SmallVector<Value, 4> falseArgs;
  if (failed(untupleAndLookupValues(oldOp.getFalseOperands(), falseArgs,
                                    builder, oldOp.getLoc(), mapping))) {
    return failure();
  }

  auto newOp = builder.create<cf::CondBranchOp>(
      oldOp.getLoc(), mapping.lookupOrNull(oldOp.getCondition()),
      mapping.lookupOrNull(oldOp.getTrueDest()), trueArgs,
      mapping.lookupOrNull(oldOp.getFalseDest()), falseArgs);

  copyOperationAttrs(oldOp, newOp);
  return success();
}

LogicalResult convertOperation(Operation *operation, OpBuilder &builder,
                               IRMapping &mapping) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(operation)
      .Case<func::ReturnOp, func::CallOp, func::CallIndirectOp, cf::BranchOp,
            cf::CondBranchOp>(
          [&](auto op) { return convertOp(op, builder, mapping); })
      .Default([&](Operation *op) {
        builder.clone(*operation, mapping);
        return success();
      });
}

LogicalResult convertFunction(func::FuncOp oldFunction,
                              func::FuncOp newFunction) {
  OpBuilder builder(newFunction.getBody());
  IRMapping mapping;

  // Check whether has tuple in signature.
  bool hasTupleSig = (oldFunction.getArgumentTypes().size() !=
                      newFunction.getArgumentTypes().size()) ||
                     (oldFunction.getResultTypes().size() !=
                      newFunction.getResultTypes().size());

  auto xlaAbiParam = StringAttr::get(newFunction.getContext(),
                                     "xla_entry_computation_parameter_layouts");
  auto xlaAbiLayout = StringAttr::get(newFunction.getContext(),
                                      "xla_entry_computation_result_layout");

  for (NamedAttribute attr : oldFunction->getAttrs()) {
    // Currently skipping all arg, result and XLA specific ABI attributes.
    if (llvm::is_contained(
            {oldFunction.getFunctionTypeAttrName(), xlaAbiParam, xlaAbiLayout},
            attr.getName())) {
      continue;
    }

    // If it has tuples in sig, then skip arg and res attrs. None of the
    // existing ones along path that produces tuples are used further, so just
    // remove instead of flattening.
    if (hasTupleSig && (attr.getName() == oldFunction.getArgAttrsAttrName() ||
                        attr.getName() == oldFunction.getResAttrsAttrName()))
      continue;
    newFunction->setAttr(attr.getName(), attr.getValue());
  }

  newFunction.getBlocks().clear();
  for (Block &oldBlock : oldFunction.getBlocks()) {
    Block *newBlock = builder.createBlock(&newFunction.getBody());
    for (BlockArgument oldArg : oldBlock.getArguments()) {
      llvm::SmallVector<Type> newTypes;
      untupleTypes(oldArg.getType(), newTypes);

      Value newTuple = processTuple(oldArg.getType(), oldFunction.getLoc(),
                                    *newBlock, builder);
      if (!newTuple) {
        return failure();
      }

      mapping.map(oldArg, newTuple);
    }
    mapping.map(&oldBlock, newBlock);
  }

  // Convert all ops in the blocks.
  for (Block &oldBlock : oldFunction.getBlocks()) {
    builder.setInsertionPointToEnd(mapping.lookupOrNull(&oldBlock));
    for (Operation &oldOp : oldBlock.getOperations()) {
      if (failed(convertOperation(&oldOp, builder, mapping))) {
        return failure();
      }
    }
  }

  return success();
}

struct FlattenTuplesInCFG final
    : impl::FlattenTuplesInCFGBase<FlattenTuplesInCFG> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    Builder builder(ctx);

    // Build a list of (oldFunction, newFunction) for all functions we need to
    // replace. This will ensure that when we go to convert function bodies we
    // have only new functions defined.
    SmallVector<std::pair<func::FuncOp, func::FuncOp>> convertedFunctions;
    for (auto oldFunction : module.getOps<func::FuncOp>()) {
      FunctionType oldFunctionType = oldFunction.getFunctionType();

      llvm::SmallVector<Type> newInputTypes;
      untupleTypes(oldFunctionType.getInputs(), newInputTypes);

      llvm::SmallVector<Type> newResultTypes;
      untupleTypes(oldFunctionType.getResults(), newResultTypes);

      FunctionType newFunctionType =
          builder.getFunctionType(newInputTypes, newResultTypes);
      func::FuncOp newFunction =
          func::FuncOp::create(oldFunction.getLoc(), oldFunction.getName(),
                               newFunctionType, oldFunction->getDialectAttrs());
      convertedFunctions.push_back({oldFunction, newFunction});

      // Perform the actual body conversion now that we have proper signatures.
      if (failed(convertFunction(oldFunction, newFunction))) {
        return signalPassFailure();
      }
    }

    // Replace functions in the module.
    for (auto [oldFunction, newFunction] : convertedFunctions) {
      oldFunction.erase();
      module.push_back(newFunction);
    }

    // Run canonicalization patterns to cancel out remaining tuple ops. We need
    // to run these manually here because StableHLO does not define
    // folds/canonicalization patterns for its ops.
    RewritePatternSet patterns(ctx);
    populateCanonicalizationPatterns(ctx, &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::iree_compiler::stablehlo
