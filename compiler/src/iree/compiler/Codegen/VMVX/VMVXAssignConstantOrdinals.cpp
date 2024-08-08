// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VMVXASSIGNCONSTANTORDINALSPASS
#include "iree/compiler/Codegen/VMVX/Passes.h.inc"

namespace {

struct VMVXAssignConstantOrdinalsPass
    : public impl::VMVXAssignConstantOrdinalsPassBase<
          VMVXAssignConstantOrdinalsPass> {
  void runOnOperation() override {
    auto variantOp = getOperation();

    // Ignore non-VMVX variants.
    // TODO(benvanik): a way to nest this in the pipeline via dynamic passes.
    if (variantOp.getTarget().getBackend().getValue() != "vmvx")
      return;

    // Get a constant key -> ordinal mapping.
    auto keyOrdinals = variantOp.gatherConstantOrdinals();
    if (keyOrdinals.empty())
      return;

    // Update placeholders to hold the concrete ordinal values.
    // Eventually the VM global folding passes will inline them.
    for (auto moduleOp :
         variantOp.getInnerModule().getOps<IREE::VM::ModuleOp>()) {
      for (auto globalOp : llvm::make_early_inc_range(
               moduleOp.getOps<IREE::VM::GlobalI32Op>())) {
        auto keyAttr = globalOp->getAttr(
            IREE::HAL::ExecutableConstantBlockOp::getKeyAttrName());
        if (!keyAttr)
          continue;
        auto it = keyOrdinals.find(keyAttr);
        if (it == keyOrdinals.end()) {
          globalOp.emitOpError()
              << "no constant block providing key '" << keyAttr << "'";
          return signalPassFailure();
        }
        globalOp->removeAttr(
            IREE::HAL::ExecutableConstantBlockOp::getKeyAttrName());
        globalOp.setGlobalMutable(false);
        globalOp.setGlobalInitialValue(IntegerAttr::get(
            IntegerType::get(globalOp.getContext(), 32), it->second));
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
