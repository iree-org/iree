// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

struct StripExecutableContentsPass
    : public PassWrapper<StripExecutableContentsPass,
                         OperationPass<mlir::ModuleOp>> {
  StringRef getArgument() const override {
    return "iree-hal-strip-executable-contents";
  }

  StringRef getDescription() const override {
    return "Strips executable module contents for reducing IR size during "
           "debugging.";
  }

  void runOnOperation() override {
    for (auto executableOp : getOperation().getOps<IREE::HAL::ExecutableOp>()) {
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        if (auto innerModuleOp = variantOp.getInnerModule()) {
          innerModuleOp.erase();
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createStripExecutableContentsPass() {
  return std::make_unique<StripExecutableContentsPass>();
}

static PassRegistration<StripExecutableContentsPass> pass;

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
