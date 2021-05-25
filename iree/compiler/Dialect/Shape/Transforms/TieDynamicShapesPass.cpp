// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Return true if `op` has a proper ancestor whose op name is in `opNames`.
static bool hasProperAncestorWithOpName(Operation *op,
                                        const DenseSet<StringRef> &opNames) {
  while ((op = op->getParentOp())) {
    if (opNames.contains(op->getName().getStringRef())) {
      return true;
    }
  }
  return false;
}

namespace {

class TieDynamicShapesPass
    : public PassWrapper<TieDynamicShapesPass, FunctionPass> {
 public:
  TieDynamicShapesPass() = default;
  TieDynamicShapesPass(const TieDynamicShapesPass &) {}
  explicit TieDynamicShapesPass(ArrayRef<std::string> doNotRecurseOpNames_) {
    doNotRecurseOpNames = doNotRecurseOpNames_;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ShapeDialect>();
  }

  void runOnFunction() override {
    DenseSet<StringRef> doNotRecurseOpNameSet;
    for (auto &s : doNotRecurseOpNames) {
      doNotRecurseOpNameSet.insert(s);
    }
    getFunction().walk([&](Operation *nestedOp) {
      if (hasProperAncestorWithOpName(nestedOp, doNotRecurseOpNameSet)) {
        return;
      }
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
    auto tieOp = builder.create<TieShapeOp>(op->getLoc(), result.getType(),
                                            result, getShapeOp);

    // Replace: {result} -> {tieOp, getShapeOp, ...origUses}
    // With: {result} -> {tieOp -> ...origUses, getShapeOp}
    result.replaceAllUsesWith(tieOp);
    tieOp.getOperation()->replaceUsesOfWith(tieOp, result);
    getShapeOp.getOperation()->replaceUsesOfWith(tieOp, result);
  }

  ListOption<std::string> doNotRecurseOpNames{
      *this, "do-not-recurse-op-names",
      llvm::cl::desc("comma separated list of op names (e.g. "
                     "`my_dialect.my_op,my_dialect.my_other_op`) whose regions "
                     "should not have their ops tied"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};

}  // namespace

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OperationPass<FuncOp>> createTieDynamicShapesPass(
    ArrayRef<std::string> doNotRecurseOpNames) {
  return std::make_unique<Shape::TieDynamicShapesPass>(doNotRecurseOpNames);
}

static PassRegistration<Shape::TieDynamicShapesPass> pass(
    "iree-shape-tie-dynamic", "Ties any dynamic shapes in a function.");

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
