// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Plugins/XLA/XlaHloShapeBuilder.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Patterns.h"
#include "iree/compiler/Utils/PatternUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace {

class MaterializeShapeCalculationsPass
    : public PassWrapper<MaterializeShapeCalculationsPass, FunctionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnFunction() override {
    auto *context = &getContext();
    // First run conversion.
    ConversionTarget target(*context);

    // Allow any Shape dialect ops to persist.
    target.addLegalDialect<ShapeDialect>();

    // Shape functions and runtime resolution generate ops in the memref and
    // standard dialect.
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<StandardOpsDialect>();

    setupMaterializeShapeCalculationsLegality(target);
    OwningRewritePatternList conversionPatterns(&getContext());
    populateMaterializeShapeCalculationsConversionPatterns(conversionPatterns,
                                                           context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(conversionPatterns)))) {
      signalPassFailure();
      return;
    }

    // And then canonicalize shape ops.
    // TODO(laurenzo): I would prefer to get the list of ops in the dialect
    // versus doing this, but I don't know that is possible.
    OwningRewritePatternList patterns(&getContext());
    CastCompatibleShapeOp::getCanonicalizationPatterns(patterns, context);
    GetRankedShapeOp::getCanonicalizationPatterns(patterns, context);
    MakeRankedShapeOp::getCanonicalizationPatterns(patterns, context);
    RankedDimOp::getCanonicalizationPatterns(patterns, context);
    RankedDimsOp::getCanonicalizationPatterns(patterns, context);
    TieShapeOp::getCanonicalizationPatterns(patterns, context);
    FromExtentTensorOp::getCanonicalizationPatterns(patterns, context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
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
