// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_RESOLVEDEVICEALIASESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-resolve-device-aliases
//===----------------------------------------------------------------------===//

static FailureOr<Attribute>
resolveAliasAttr(Operation *forOp, IREE::HAL::DeviceAliasAttr aliasAttr,
                 const TargetRegistry &targetRegistry) {
  // Lookup device in the registry.
  auto targetDevice =
      targetRegistry.getTargetDevice(aliasAttr.getDeviceID().getValue());
  if (!targetDevice) {
    auto diagnostic = forOp->emitError();
    diagnostic << "unregistered device alias " << aliasAttr.getDeviceID()
               << "; ensure it is linked into the compiler (available = [ ";
    for (const auto &targetName : targetRegistry.getRegisteredTargetDevices()) {
      diagnostic << "'" << targetName << "' ";
    }
    diagnostic << "])";
    return diagnostic;
  }

  // Query the default device target.
  auto defaultAttr =
      targetDevice->getDefaultDeviceTarget(forOp->getContext(), targetRegistry);
  assert(defaultAttr && "expected a default device target attr");

  // Merge in any additional configuration from the alias attr.
  if (aliasAttr.getOrdinal().has_value() ||
      (aliasAttr.getConfiguration() && !aliasAttr.getConfiguration().empty())) {
    NamedAttrList configAttrs;
    if (auto defaultConfigAttr = defaultAttr.getConfiguration()) {
      for (auto existingAttr : defaultConfigAttr) {
        configAttrs.push_back(existingAttr);
      }
    }
    if (auto overrideConfigAttr = aliasAttr.getConfiguration()) {
      for (auto overrideAttr : overrideConfigAttr) {
        configAttrs.set(overrideAttr.getName(), overrideAttr.getValue());
      }
    }
    if (aliasAttr.getOrdinal().has_value()) {
      configAttrs.set("ordinal",
                      IntegerAttr::get(IndexType::get(forOp->getContext()),
                                       aliasAttr.getOrdinal().value()));
    }
    defaultAttr = IREE::HAL::DeviceTargetAttr::get(
        forOp->getContext(), defaultAttr.getDeviceID(),
        DictionaryAttr::get(forOp->getContext(), configAttrs),
        defaultAttr.getExecutableTargets());
  }

  return defaultAttr;
}

static FailureOr<Attribute>
resolveNestedAliasAttrs(Operation *forOp, Attribute attr,
                        const TargetRegistry &targetRegistry) {
  if (auto aliasAttr = dyn_cast<IREE::HAL::DeviceAliasAttr>(attr)) {
    return resolveAliasAttr(forOp, aliasAttr, targetRegistry);
  } else if (auto selectAttr = dyn_cast<IREE::HAL::DeviceSelectAttr>(attr)) {
    SmallVector<Attribute> resolvedAttrs;
    bool didChange = false;
    for (auto deviceAttr : selectAttr.getDevices()) {
      auto resolvedAttr =
          resolveNestedAliasAttrs(forOp, deviceAttr, targetRegistry);
      if (failed(resolvedAttr)) {
        return failure();
      }
      didChange = didChange || *resolvedAttr != deviceAttr;
      resolvedAttrs.push_back(*resolvedAttr);
    }
    return didChange ? IREE::HAL::DeviceSelectAttr::get(attr.getContext(),
                                                        resolvedAttrs)
                     : attr;
  } else {
    return attr; // pass-through
  }
}

struct ResolveDeviceAliasesPass
    : public IREE::HAL::impl::ResolveDeviceAliasesPassBase<
          ResolveDeviceAliasesPass> {
  using IREE::HAL::impl::ResolveDeviceAliasesPassBase<
      ResolveDeviceAliasesPass>::ResolveDeviceAliasesPassBase;
  void runOnOperation() override {
    // Walks all device globals and resolve any aliases found.
    auto moduleOp = getOperation();
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      if (!isa<IREE::HAL::DeviceType>(globalOp.getGlobalType())) {
        continue;
      }
      auto initialValue = globalOp.getGlobalInitialValue();
      if (!initialValue) {
        continue;
      }
      auto resolvedValue = resolveNestedAliasAttrs(globalOp, initialValue,
                                                   *targetRegistry.value);
      if (failed(resolvedValue)) {
        return signalPassFailure();
      }
      globalOp.setGlobalInitialValue(*resolvedValue);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
