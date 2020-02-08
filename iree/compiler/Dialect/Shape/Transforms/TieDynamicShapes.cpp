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

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

namespace {

class TieDynamicShapesPass : public FunctionPass<TieDynamicShapesPass> {
  void runOnFunction() override {
    getFunction().walk([&](Operation *nestedOp) {
      for (auto result : nestedOp->getResults()) {
        rewriteOperationResult(nestedOp, result);
      }
    });
  }

  void rewriteOperationResult(Operation *op, Value result) {
    if (llvm::isa<TieShapeOp>(op)) return;
    auto shapedType = result.getType().dyn_cast<ShapedType>();
    if (!shapedType || shapedType.hasStaticShape()) return;

    // Only ranked is supported currently.
    if (!shapedType.hasRank()) return;

    // Skip un-ambiguous ties that already exist.
    if (result.hasOneUse() &&
        llvm::dyn_cast_or_null<TieShapeOp>(result.use_begin()->getOwner())) {
      return;
    }

    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(op);
    auto getShapeOp = builder.create<GetRankedShapeOp>(op->getLoc(), result);
    auto tieOp = builder.create<TieShapeOp>(op->getLoc(), result, getShapeOp);

    // Replace: {result} -> {tieOp, getShapeOp, ...origUses}
    // With: {result} -> {tieOp -> ...origUses, getShapeOp}
    result.replaceAllUsesWith(tieOp);
    tieOp.getOperation()->replaceUsesOfWith(tieOp, result);
    getShapeOp.getOperation()->replaceUsesOfWith(tieOp, result);
  }
};

}  // namespace
}  // namespace Shape

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OpPassBase<FuncOp>> createTieDynamicShapesPass() {
  return std::make_unique<Shape::TieDynamicShapesPass>();
}

static PassRegistration<Shape::TieDynamicShapesPass> pass(
    "iree-shape-tie-dynamic", "Ties any dynamic shapes in a function.");

}  // namespace iree_compiler
}  // namespace mlir
