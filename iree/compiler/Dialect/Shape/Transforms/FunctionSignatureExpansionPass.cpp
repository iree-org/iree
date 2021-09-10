// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Utils/TypeConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

static LogicalResult expandCallGraphTypesForCall(
    CallOp callOp, const TypeExpander &typeExpander) {
  Location loc = callOp.getLoc();

  OpBuilder builder(callOp);
  // Create the operands list of the new `CallOp`.
  SmallVector<Value> newOperands;
  if (failed(typeExpander.expandSourceValuesToTarget(
          loc, llvm::to_vector<6>(callOp.getOperands()), newOperands,
          builder))) {
    return failure();
  }

  // Create the new result types for the new `CallOp` and track the indices
  // in the new call op's results that correspond to the old call op's
  // results.
  //
  // expandedResultIndices[i] = "list of new result indices that old result
  // i expanded to".
  SmallVector<Type> newResultTypes;
  SmallVector<SmallVector<unsigned>> expandedResultIndices;
  for (Type resultType : callOp.getResultTypes()) {
    unsigned oldSize = newResultTypes.size();
    if (failed(typeExpander.convertType(resultType, newResultTypes))) {
      return failure();
    }
    auto &resultMapping = expandedResultIndices.emplace_back();
    for (unsigned i = oldSize, e = newResultTypes.size(); i < e; i++)
      resultMapping.push_back(i);
  }

  CallOp newCallOp = builder.create<CallOp>(loc, callOp.getCallee(),
                                            newResultTypes, newOperands);

  // Build a replacement value for each result to replace its uses. If a
  // result has multiple mapping values, it needs to be materialized as a
  // single value.
  SmallVector<Value, 2> replacedValues;
  replacedValues.reserve(callOp.getNumResults());
  for (unsigned i = 0, e = callOp.getNumResults(); i < e; ++i) {
    auto decomposedValues = llvm::to_vector<6>(
        llvm::map_range(expandedResultIndices[i],
                        [&](unsigned i) { return newCallOp.getResult(i); }));
    Value materialized = typeExpander.castToSource(loc, callOp.getType(i),
                                                   decomposedValues, builder);
    replacedValues.push_back(materialized);
  }
  callOp.replaceAllUsesWith(replacedValues);
  return success();
}

static LogicalResult expandCallGraphTypes(ModuleOp module,
                                          const TypeExpander &typeExpander) {
  for (auto funcOp : module.getOps<FuncOp>()) {
    OpBuilder builder(funcOp);
    if (failed(typeExpander.expandFunctionSignature(funcOp, builder)) ||
        failed(typeExpander.expandAllReturnLikeTerminators<mlir::ReturnOp>(
            funcOp, builder))) {
      return funcOp.emitError()
             << "couldn't expand function signature or return";
    }
    bool hadError = false;
    SmallVector<Operation *> toErase;
    funcOp.walk([&](CallOp callOp) {
      if (failed(expandCallGraphTypesForCall(callOp, typeExpander))) {
        hadError = true;
        return;
      }
      toErase.push_back(callOp);
    });
    if (hadError) return failure();
    for (Operation *op : toErase) op->erase();
  }
  return success();
}

namespace {

class ExpandFunctionDynamicDimsPass
    : public PassWrapper<ExpandFunctionDynamicDimsPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ShapeDialect>();
  }

  StringRef getArgument() const override {
    return "iree-shape-expand-function-dynamic-dims";
  }

  StringRef getDescription() const override {
    return "Expands dynamic dimensions in function signatures.";
  }

  void runOnOperation() override {
    auto &typeExpander = getDynamicShapeTypeExpander();
    if (failed(expandCallGraphTypes(getOperation(), typeExpander)))
      signalPassFailure();
  }
};

class ExpandFunctionRankedShapeDimsPass
    : public PassWrapper<ExpandFunctionRankedShapeDimsPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ShapeDialect>();
  }

  StringRef getArgument() const override {
    return "iree-shape-expand-function-ranked-shape-dims";
  }

  StringRef getDescription() const override {
    return "Expands ranked_shape types at function boundaries to loose dims.";
  }

  void runOnOperation() override {
    auto &typeExpander = getShapeToPrimitiveTypeExpander();
    if (failed(expandCallGraphTypes(getOperation(), typeExpander)))
      signalPassFailure();
  }
};

}  // namespace

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OperationPass<ModuleOp>> createExpandFunctionDynamicDimsPass() {
  return std::make_unique<Shape::ExpandFunctionDynamicDimsPass>();
}

// For any function which contains ranked_shape argument/result types,
// expands them to individual dynamic dimensions, inserting appropriate casts
// within the function.
std::unique_ptr<OperationPass<ModuleOp>>
createExpandFunctionRankedShapeDimsPass() {
  return std::make_unique<Shape::ExpandFunctionRankedShapeDimsPass>();
}

static PassRegistration<Shape::ExpandFunctionDynamicDimsPass> pass_dynamic;
static PassRegistration<Shape::ExpandFunctionRankedShapeDimsPass> pass_rs;

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
