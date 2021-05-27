// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class ExpandVariableDynamicDimsPass
    : public ExpandVariableDynamicDimsBase<ExpandVariableDynamicDimsPass> {
 public:
  ExpandVariableDynamicDimsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<ShapeDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Gathers all of the flow.variables containing shapes.
    SmallVector<VariableOp, 4> shapeVarOps;
    moduleOp.walk([&](VariableOp op) {
      if (op.type().isa<Shape::RankedShapeType>()) {
        shapeVarOps.push_back(op);
      }
    });

    // Split each variable into one variable per dimension.
    for (auto shapeVarOp : shapeVarOps) {
      expandShapeVariable(moduleOp, shapeVarOp);
    }
  }

 private:
  // Expands a flow.variable representing a shape with one variable per dim.
  // Uses of the variable will be replaced with the per-dim ones.
  void expandShapeVariable(ModuleOp moduleOp, VariableOp shapeVarOp) {
    // Create one flow.variable per dimension (static or dynamic).
    OpBuilder moduleBuilder(shapeVarOp);
    auto shapeType = shapeVarOp.type().cast<Shape::RankedShapeType>();
    SmallVector<VariableOp, 4> dimVarOps;
    for (int i = 0; i < shapeType.getRank(); ++i) {
      Attribute initialDimValue;
      if (shapeType.isDimDynamic(i)) {
        // Right now we choose zero for initial dynamic dim values but this
        // needs to agree with bindings that may have expectations on query.
        // 0 is at least easier to gracefully bail on when values are never
        // overridden.
        initialDimValue = moduleBuilder.getIndexAttr(0);
      } else {
        initialDimValue = moduleBuilder.getIndexAttr(shapeType.getStaticDim(i));
      }
      auto dimVarOp = moduleBuilder.create<VariableOp>(
          shapeVarOp.getLoc(),
          (shapeVarOp.getName() + "_d" + std::to_string(i)).str(),
          /*isMutable=*/shapeType.isDimDynamic(i), moduleBuilder.getIndexType(),
          initialDimValue);
      dimVarOp.setPrivate();
      dimVarOps.push_back(dimVarOp);
    }

    // Replace all uses of the single variable with the split ones.
    replaceShapeVariableUses(moduleOp, shapeType, shapeVarOp, dimVarOps);

    // Erase the original variable.
    shapeVarOp.erase();
  }

  // Replaces uses of |shapeVarOp| in |moduleOp| with the expanded |dimVarOps|.
  void replaceShapeVariableUses(ModuleOp moduleOp,
                                Shape::RankedShapeType shapeType,
                                VariableOp shapeVarOp,
                                ArrayRef<VariableOp> dimVarOps) {
    auto allUses = SymbolTable::getSymbolUses(shapeVarOp, moduleOp)
                       .getValueOr(SymbolTable::UseRange({}));
    for (auto use : allUses) {
      if (auto loadOp = dyn_cast<VariableLoadOp>(use.getUser())) {
        OpBuilder builder(loadOp);
        SmallVector<Value, 4> dynamicDimValues;
        for (int i = 0; i < shapeType.getRank(); ++i) {
          if (!shapeType.isDimDynamic(i)) continue;
          VariableOp dimVarOp = dimVarOps[i];
          dynamicDimValues.push_back(builder.create<VariableLoadOp>(
              loadOp.getLoc(), builder.getIndexType(), dimVarOp.getName()));
        }
        auto shapeValue = builder.create<Shape::MakeRankedShapeOp>(
            loadOp.getLoc(), shapeType, dynamicDimValues);
        loadOp->replaceAllUsesWith(shapeValue);
        loadOp.erase();
      } else if (auto storeOp = dyn_cast<VariableStoreOp>(use.getUser())) {
        OpBuilder builder(storeOp);
        auto shapeValue = storeOp.value();
        for (int i = 0; i < shapeType.getRank(); ++i) {
          if (!shapeType.isDimDynamic(i)) continue;
          VariableOp dimVarOp = dimVarOps[i];
          auto dynamicDimValue = builder.createOrFold<Shape::RankedDimOp>(
              storeOp.getLoc(), shapeValue, i);
          builder.create<VariableStoreOp>(storeOp.getLoc(), dynamicDimValue,
                                          dimVarOp.getName());
        }
        storeOp.erase();
      } else {
        // TODO(benvanik): support indirection/addressing - should be fairly
        // easy to do by splitting the address ops to each dim.
        use.getUser()->emitError()
            << "variable action on shape is not yet supported";
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createExpandVariableDynamicDimsPass() {
  return std::make_unique<ExpandVariableDynamicDimsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
