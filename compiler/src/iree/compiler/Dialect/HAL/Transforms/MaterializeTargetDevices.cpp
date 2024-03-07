// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_MATERIALIZETARGETDEVICESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-materialize-target-devices
//===----------------------------------------------------------------------===//

struct MaterializeTargetDevicesPass
    : public IREE::HAL::impl::MaterializeTargetDevicesPassBase<
          MaterializeTargetDevicesPass> {
  using IREE::HAL::impl::MaterializeTargetDevicesPassBase<
      MaterializeTargetDevicesPass>::MaterializeTargetDevicesPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Only run if there's a module-level attribute specified.
    auto deviceTargetAttrs =
        moduleOp->getAttrOfType<ArrayAttr>("hal.device.targets");
    if (!deviceTargetAttrs || deviceTargetAttrs.empty())
      return;
    moduleOp->removeAttr("hal.device.targets");

    // Create the default device global.
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    auto deviceType = moduleBuilder.getType<IREE::HAL::DeviceType>();
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        moduleOp.getLoc(), "__device.0", /*isMutable=*/false, deviceType);
    globalOp.setPrivate();
    if (deviceTargetAttrs.size() == 1) {
      auto typedAttr =
          dyn_cast<TypedAttr>(deviceTargetAttrs.getValue().front());
      if (typedAttr && isa<IREE::HAL::DeviceType>(typedAttr.getType())) {
        globalOp.setInitialValueAttr(typedAttr);
      } else {
        moduleOp.emitOpError()
            << "has invalid device targets specified; "
               "expect hal.device.targets to be an "
               "ArrayAttr of !hal.device initialization attributes";
        return signalPassFailure();
      }
    } else {
      globalOp.setInitialValueAttr(
          moduleBuilder.getAttr<IREE::HAL::DeviceSelectAttr>(
              deviceType, deviceTargetAttrs));
    }

    // Assign affinities to all top level ops that don't already have one set.
    auto affinityName = StringAttr::get(&getContext(), "stream.affinity");
    auto affinityAttr = moduleBuilder.getAttr<IREE::HAL::DeviceAffinityAttr>(
        FlatSymbolRefAttr::get(globalOp), /*queue_mask=*/-1ll);
    auto isAnnotatableType = [](Type type) {
      return isa<TensorType>(type) || isa<IREE::Stream::ResourceType>(type);
    };
    for (auto &op : moduleOp.getOps()) {
      bool shouldAnnotate = true;
      if (auto globalOp = dyn_cast<IREE::Util::GlobalOpInterface>(op)) {
        if (!isAnnotatableType(globalOp.getGlobalType()))
          shouldAnnotate = false;
      } else if (op.hasTrait<OpTrait::SymbolTable>()) {
        // Symbol table ops can't reference parent symbols properly.
        shouldAnnotate = false;
      }
      if (!shouldAnnotate)
        continue;
      if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
        if (!affinityOp.getAffinity())
          affinityOp.setAffinity(affinityAttr);
      } else {
        if (!op.hasAttr(affinityName))
          op.setAttr(affinityName, affinityAttr);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
