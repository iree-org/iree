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

#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Plugins/XLA/XlaHloShapeBuilder.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Patterns.h"
#include "iree/compiler/Utils/PatternUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace {

class MaterializeShapeCalculationsPass
    : public PassWrapper<MaterializeShapeCalculationsPass, FunctionPass> {
 public:
  void runOnFunction() override {
    auto *context = &getContext();
    // First run conversion.
    ConversionTarget target(*context);

    // Allow any Shape dialect ops to persist.
    target.addLegalDialect<ShapeDialect>();

    // Shape functions and runtime resolution generate ops in the standard
    // dialect.
    target.addLegalDialect<StandardOpsDialect>();

    setupMaterializeShapeCalculationsLegality(target);
    OwningRewritePatternList conversionPatterns;
    populateMaterializeShapeCalculationsConversionPatterns(conversionPatterns,
                                                           context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      conversionPatterns))) {
      signalPassFailure();
      return;
    }

    // And then canonicalize shape ops.
    // TODO(laurenzo): I would prefer to get the list of ops in the dialect
    // versus doing this, but I don't know that is possible.
    OwningRewritePatternList patterns;
    CastCompatibleShapeOp::getCanonicalizationPatterns(patterns, context);
    GetRankedShapeOp::getCanonicalizationPatterns(patterns, context);
    MakeRankedShapeOp::getCanonicalizationPatterns(patterns, context);
    RankedDimOp::getCanonicalizationPatterns(patterns, context);
    TieShapeOp::getCanonicalizationPatterns(patterns, context);
    applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

}  // namespace

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OperationPass<FuncOp>>
createMaterializeShapeCalculationsPass() {
  return std::make_unique<Shape::MaterializeShapeCalculationsPass>();
}

static PassRegistration<Shape::MaterializeShapeCalculationsPass> pass(
    "iree-shape-materialize-calculations", "Materializes shape calculations.");

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
