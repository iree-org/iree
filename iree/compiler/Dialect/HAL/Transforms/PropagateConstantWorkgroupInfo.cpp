// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class PropagateConstantWorkgroupInfoPass
    : public PassWrapper<PropagateConstantWorkgroupInfoPass,
                         OperationPass<IREE::HAL::ExecutableVariantOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-hal-propagate-constant-workgroup-info";
  }

  StringRef getDescription() const override {
    return "Propagates constant hal.interface.workgroup.* queries when known";
  }

  void runOnOperation() override {
    auto variantOp = getOperation();

    SymbolTable targetSymbolTable(variantOp);
    for (auto funcOp : variantOp.getInnerModule().getOps<FuncOp>()) {
      auto entryPointOp =
          targetSymbolTable.lookup<IREE::HAL::ExecutableEntryPointOp>(
              funcOp.getName());
      if (!entryPointOp) continue;
      if (!entryPointOp.workgroup_size().hasValue()) continue;
      auto workgroupSizeAttr = entryPointOp.workgroup_sizeAttr();
      auto workgroupSizeOps = llvm::to_vector<4>(
          funcOp.getOps<IREE::HAL::InterfaceWorkgroupSizeOp>());
      for (auto workgroupSizeOp : workgroupSizeOps) {
        OpBuilder builder(workgroupSizeOp);
        auto dimValue = builder.createOrFold<ConstantIndexOp>(
            workgroupSizeOp.getLoc(),
            workgroupSizeAttr[workgroupSizeOp.dimension().getZExtValue()]
                .cast<IntegerAttr>()
                .getInt());
        workgroupSizeOp.replaceAllUsesWith(dimValue);
        workgroupSizeOp.erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createPropagateConstantWorkgroupInfoPass() {
  return std::make_unique<PropagateConstantWorkgroupInfoPass>();
}

static PassRegistration<PropagateConstantWorkgroupInfoPass> pass;

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
