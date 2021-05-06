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

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
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
    registry.insert<IREE::Flow::FlowDialect>();
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

    moduleOp.walk([&](VariableOp op) {
      // Only strip constant variables.
      if (op.is_mutable()) {
        return;
      }

      // Only strip tensor type constants (to replace with dense<>).
      if (!op.type().isa<TensorType>()) {
        return;
      }

      auto tensorType = op.type().cast<TensorType>();
      auto elementType = tensorType.getElementType();
      DenseElementsAttr newValue;
      if (elementType.isa<FloatType>()) {
        newValue = DenseElementsAttr::get(
            tensorType, FloatAttr::get(elementType, 1.0 / replaceIndex));
      } else {
        newValue = DenseElementsAttr::get(
            tensorType, IntegerAttr::get(elementType, replaceIndex));
      }

      builder.setInsertionPointAfter(op);
      auto newOp = builder.create<VariableOp>(
          op.getLoc(), op.sym_name(), op.is_mutable(), op.type(), newValue);
      newOp.setVisibility(op.getVisibility());
      newOp->setAttr("noinline", UnitAttr::get(builder.getContext()));
      op.erase();

      replaceIndex++;
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createStripAndSplatConstantVariablesPass() {
  return std::make_unique<StripAndSplatConstantVariablesPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
