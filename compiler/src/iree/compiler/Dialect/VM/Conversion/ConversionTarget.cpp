// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"

namespace mlir::iree_compiler {

// static
std::pair<mlir::ModuleOp, mlir::ModuleOp>
VMConversionTarget::nestModuleForConversion(mlir::ModuleOp outerModuleOp) {
  if (isa<IREE::VM::ModuleOp>(outerModuleOp.getBody()->front())) {
    // Already have a VM module; no need for the nesting.
    return std::make_pair(outerModuleOp, outerModuleOp);
  }
  auto innerModuleOp = dyn_cast<ModuleOp>(outerModuleOp.getBody()->front());
  if (!innerModuleOp) {
    innerModuleOp =
        ModuleOp::create(outerModuleOp.getLoc(), outerModuleOp.getName());
    innerModuleOp.getBodyRegion().takeBody(outerModuleOp.getBodyRegion());
    outerModuleOp.getBodyRegion().getBlocks().push_back(new Block());
    outerModuleOp.push_back(innerModuleOp);
  }

  outerModuleOp->setAttr("vm.toplevel",
                         UnitAttr::get(outerModuleOp.getContext()));
  return std::make_pair(outerModuleOp, innerModuleOp);
}

// static
bool VMConversionTarget::isTopLevelModule(mlir::ModuleOp moduleOp) {
  return !moduleOp->getParentOp() || moduleOp->hasAttr("vm.toplevel");
}

VMConversionTarget::VMConversionTarget(MLIRContext *context)
    : ConversionTarget(*context) {
  addLegalDialect<IREE::VM::VMDialect>();

  // NOTE: we need to allow the outermost std.module to be legal to support the
  // double-nesting (module { vm.module { ... } }).
  addDynamicallyLegalOp<mlir::ModuleOp>(
      +[](mlir::ModuleOp op) { return isTopLevelModule(op); });
}

} // namespace mlir::iree_compiler
