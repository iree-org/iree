// Copyright 2024 The IREE Authors
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

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_HOISTEXECUTABLEOBJECTSPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-hoist-executable-objects
//===----------------------------------------------------------------------===//

struct HoistExecutableObjectsPass
    : public IREE::HAL::impl::HoistExecutableObjectsPassBase<
          HoistExecutableObjectsPass> {
  void runOnOperation() override {
    // Note that some executables may be external and not have any contents.
    if (getOperation().isExternal()) {
      return;
    }

    auto objectsAttrName =
        StringAttr::get(&getContext(), "hal.executable.objects");

    // Seed with existing variant-level object attrs, if any present.
    SetVector<Attribute> allObjectAttrs;
    if (auto existingAttr = getOperation().getObjectsAttr()) {
      allObjectAttrs.insert(existingAttr.begin(), existingAttr.end());
    }

    // Move all op-level attributes into a unique set. Note that order can be
    // important so we use an ordered set.
    //
    // We could do this first as a gather step in parallel if this walk gets too
    // expensive.
    bool foundAnyAttrs = false;
    getOperation().getInnerModule().walk([&](Operation *op) {
      auto objectsAttr = op->getAttrOfType<ArrayAttr>(objectsAttrName);
      if (objectsAttr) {
        allObjectAttrs.insert(objectsAttr.begin(), objectsAttr.end());
        op->removeAttr(objectsAttrName);
        foundAnyAttrs = true;
      }
    });

    // Update the variant if any changes were made.
    if (foundAnyAttrs) {
      getOperation().setObjectsAttr(
          ArrayAttr::get(&getContext(), allObjectAttrs.getArrayRef()));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
