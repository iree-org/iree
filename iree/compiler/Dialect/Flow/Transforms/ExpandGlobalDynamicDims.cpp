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
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class ExpandGlobalDynamicDimsPass
    : public ExpandGlobalDynamicDimsBase<ExpandGlobalDynamicDimsPass> {
 public:
  ExpandGlobalDynamicDimsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, IREE::Util::UtilDialect>();
    registry.insert<ShapeDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Gathers all of the util.globals containing shapes.
    SmallVector<IREE::Util::GlobalOp, 4> shapeGlobalOps;
    moduleOp.walk([&](IREE::Util::GlobalOp op) {
      if (op.type().isa<Shape::RankedShapeType>()) {
        shapeGlobalOps.push_back(op);
      }
    });

    // Split each global into one global per dimension.
    for (auto shapeGlobalOp : shapeGlobalOps) {
      expandShapeGlobal(moduleOp, shapeGlobalOp);
    }
  }

 private:
  // Expands a util.global representing a shape with one global per dim.
  // Uses of the global will be replaced with the per-dim ones.
  void expandShapeGlobal(mlir::ModuleOp moduleOp,
                         IREE::Util::GlobalOp shapeGlobalOp) {
    // Create one util.global per dimension (static or dynamic).
    OpBuilder moduleBuilder(shapeGlobalOp);
    auto shapeType = shapeGlobalOp.type().cast<Shape::RankedShapeType>();
    SmallVector<IREE::Util::GlobalOp, 4> dimGlobalOps;
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
      auto dimGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
          shapeGlobalOp.getLoc(),
          (shapeGlobalOp.getName() + "_d" + std::to_string(i)).str(),
          /*isMutable=*/shapeType.isDimDynamic(i), moduleBuilder.getIndexType(),
          initialDimValue);
      dimGlobalOp.setPrivate();
      dimGlobalOps.push_back(dimGlobalOp);
    }

    // Replace all uses of the single global with the split ones.
    replaceShapeGlobalUses(moduleOp, shapeType, shapeGlobalOp, dimGlobalOps);

    // Erase the original global.
    shapeGlobalOp.erase();
  }

  // Replaces uses of |shapeGlobalOp| in |moduleOp| with the expanded
  // |dimGlobalOps|.
  void replaceShapeGlobalUses(mlir::ModuleOp moduleOp,
                              Shape::RankedShapeType shapeType,
                              IREE::Util::GlobalOp shapeGlobalOp,
                              ArrayRef<IREE::Util::GlobalOp> dimGlobalOps) {
    auto allUses = SymbolTable::getSymbolUses(shapeGlobalOp, moduleOp)
                       .getValueOr(SymbolTable::UseRange({}));
    for (auto use : allUses) {
      if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOp>(use.getUser())) {
        OpBuilder builder(loadOp);
        SmallVector<Value, 4> dynamicDimValues;
        for (int i = 0; i < shapeType.getRank(); ++i) {
          if (!shapeType.isDimDynamic(i)) continue;
          IREE::Util::GlobalOp dimGlobalOp = dimGlobalOps[i];
          dynamicDimValues.push_back(builder.create<IREE::Util::GlobalLoadOp>(
              loadOp.getLoc(), builder.getIndexType(), dimGlobalOp.getName()));
        }
        auto shapeValue = builder.create<Shape::MakeRankedShapeOp>(
            loadOp.getLoc(), shapeType, dynamicDimValues);
        loadOp->replaceAllUsesWith(shapeValue);
        loadOp.erase();
      } else if (auto storeOp =
                     dyn_cast<IREE::Util::GlobalStoreOp>(use.getUser())) {
        OpBuilder builder(storeOp);
        auto shapeValue = storeOp.value();
        for (int i = 0; i < shapeType.getRank(); ++i) {
          if (!shapeType.isDimDynamic(i)) continue;
          IREE::Util::GlobalOp dimGlobalOp = dimGlobalOps[i];
          auto dynamicDimValue = builder.createOrFold<Shape::RankedDimOp>(
              storeOp.getLoc(), shapeValue, i);
          builder.create<IREE::Util::GlobalStoreOp>(
              storeOp.getLoc(), dynamicDimValue, dimGlobalOp.getName());
        }
        storeOp.erase();
      } else {
        // TODO(benvanik): support indirection/addressing - should be fairly
        // easy to do by splitting the address ops to each dim.
        use.getUser()->emitError()
            << "global action on shape is not yet supported";
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createExpandGlobalDynamicDimsPass() {
  return std::make_unique<ExpandGlobalDynamicDimsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
