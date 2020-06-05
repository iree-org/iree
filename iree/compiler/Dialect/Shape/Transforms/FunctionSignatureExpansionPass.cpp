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

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Utils/TypeConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace {

class ExpandFunctionDynamicDimsPass
    : public PassWrapper<ExpandFunctionDynamicDimsPass, FunctionPass> {
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
