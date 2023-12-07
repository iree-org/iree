// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Util {

class StripAndSplatConstantsPass
    : public StripAndSplatConstantsBase<StripAndSplatConstantsPass> {
public:
  StripAndSplatConstantsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Give each splatted value a module-unique byte value so that it's easier
    // to track back to where it came from in the final output.
    int replaceIndex = 1;
    auto getSplatAttr = [&](ShapedType type) {
      return IREE::Util::BytePatternAttr::get(moduleOp.getContext(), type,
                                              replaceIndex++);
    };

    moduleOp.walk([&](Operation *op) {
      if (auto globalOp = dyn_cast<Util::GlobalOp>(op)) {
        if (auto initialValue = globalOp.getInitialValueAttr()) {
          if (auto shapedType = dyn_cast<ShapedType>(initialValue.getType())) {
            globalOp.setInitialValueAttr(getSplatAttr(shapedType));
          }
        }
      }
    });
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createStripAndSplatConstantsPass() {
  return std::make_unique<StripAndSplatConstantsPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
