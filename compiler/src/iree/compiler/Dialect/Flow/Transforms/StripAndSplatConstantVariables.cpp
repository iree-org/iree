// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class StripAndSplatConstantVariablesPass
    : public StripAndSplatConstantVariablesBase<
          StripAndSplatConstantVariablesPass> {
 public:
  StripAndSplatConstantVariablesPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    // Use a heuristic to space out splat values in hopes of avoiding NaN and
    // INF values at runtime:
    //   floats: 1/1, 1/2, 1/3, ...
    //   ints: 1, 2, 3, 4, ...
    // TODO(scotttodd): flags to control numbers used (all 0, all 1, increasing)
    int replaceIndex = 1;

    auto getSplatAttr = [&](TensorType tensorType) {
      auto elementType = tensorType.getElementType();
      TypedAttr newAttr;
      if (llvm::isa<FloatType>(elementType)) {
        newAttr = DenseElementsAttr::get(
            tensorType, FloatAttr::get(elementType, 1.0 / replaceIndex));
      } else {
        assert(elementType.isa<IntegerType>());
        newAttr = DenseElementsAttr::get(
            tensorType, IntegerAttr::get(elementType, replaceIndex));
      }

      replaceIndex++;
      return newAttr;
    };

    moduleOp.walk([&](Operation *op) {
      if (auto globalOp = dyn_cast<Util::GlobalOp>(op)) {
        // Only strip constant variables.
        if (globalOp.getIsMutable()) return;

        // Only strip tensor type constants (to replace with dense<>).
        if (!llvm::isa<TensorType>(globalOp.getType())) return;

        auto tensorType = llvm::cast<TensorType>(globalOp.getType());
        TypedAttr newValue = getSplatAttr(tensorType);

        builder.setInsertionPoint(globalOp);
        auto newOp = builder.create<IREE::Util::GlobalOp>(
            globalOp.getLoc(), globalOp.getSymName(), globalOp.getIsMutable(),
            globalOp.getType(), newValue);
        newOp.setVisibility(globalOp.getVisibility());
        newOp->setAttr("noinline", UnitAttr::get(builder.getContext()));
        globalOp.erase();
      } else if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
        if (!llvm::isa<TensorType>(cstOp.getType())) return;

        auto tensorType = llvm::cast<TensorType>(cstOp.getType());
        TypedAttr newValue = getSplatAttr(tensorType);
        builder.setInsertionPoint(cstOp);
        auto newOp =
            builder.create<arith::ConstantOp>(cstOp.getLoc(), newValue);
        cstOp->replaceAllUsesWith(newOp);
        cstOp.erase();
      }
    });
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createStripAndSplatConstantVariablesPass() {
  return std::make_unique<StripAndSplatConstantVariablesPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
