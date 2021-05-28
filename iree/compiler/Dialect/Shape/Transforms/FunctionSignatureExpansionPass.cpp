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
namespace {

class ExpandFunctionDynamicDimsPass
    : public PassWrapper<ExpandFunctionDynamicDimsPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ShapeDialect>();
  }

  void runOnFunction() override {
    auto funcOp = getFunction();
    auto &typeExpander = getDynamicShapeTypeExpander();
    OpBuilder builder(funcOp);
    if (failed(typeExpander.expandFunctionSignature(funcOp, builder)) ||
        failed(typeExpander.expandAllReturnLikeTerminators<mlir::ReturnOp>(
            funcOp, builder))) {
      return signalPassFailure();
    }
  }
};

class ExpandFunctionRankedShapeDimsPass
    : public PassWrapper<ExpandFunctionRankedShapeDimsPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ShapeDialect>();
  }

  void runOnFunction() override {
    auto funcOp = getFunction();
    auto &typeExpander = getShapeToPrimitiveTypeExpander();
    OpBuilder builder(funcOp);
    if (failed(typeExpander.expandFunctionSignature(funcOp, builder)) ||
        failed(typeExpander.expandAllReturnLikeTerminators<mlir::ReturnOp>(
            funcOp, builder))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OperationPass<FuncOp>> createExpandFunctionDynamicDimsPass() {
  return std::make_unique<Shape::ExpandFunctionDynamicDimsPass>();
}

// For any function which contains ranked_shape argument/result types,
// expands them to individual dynamic dimensions, inserting appropriate casts
// within the function.
std::unique_ptr<OperationPass<FuncOp>>
createExpandFunctionRankedShapeDimsPass() {
  return std::make_unique<Shape::ExpandFunctionRankedShapeDimsPass>();
}

static PassRegistration<Shape::ExpandFunctionDynamicDimsPass> pass_dynamic(
    "iree-shape-expand-function-dynamic-dims",
    "Expands dynamic dimensions in function signatures.");

static PassRegistration<Shape::ExpandFunctionRankedShapeDimsPass> pass_rs(
    "iree-shape-expand-function-ranked-shape-dims",
    "Expands ranked_shape types at function boundaries to loose dims.");

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
