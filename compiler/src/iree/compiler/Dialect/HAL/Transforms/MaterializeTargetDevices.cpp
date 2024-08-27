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

// Returns the canonical name for a device by ordinal:
// device ordinal `N` -> `@__device_N`
static FlatSymbolRefAttr makeDefaultDeviceOrdinalRef(MLIRContext *context,
                                                     int64_t ordinal) {
  return FlatSymbolRefAttr::get(
      context, (StringRef("__device_") + std::to_string(ordinal)).str());
}

// Returns the canonical name for a device by name:
// device name `NAME` -> `@NAME`
static FlatSymbolRefAttr makeDefaultDeviceNameRef(MLIRContext *context,
                                                  StringRef name) {
  return FlatSymbolRefAttr::get(context, name);
}

// Returns a symbol ref constructed to reference the specified device.
// Supports:
//   integer attrs: device ordinal `N` -> `@__device_N`
//   string attrs: device name `NAME` -> `@NAME`
static FailureOr<FlatSymbolRefAttr>
makeDefaultDeviceAttrRef(Attribute defaultDeviceAttr) {
  if (auto stringAttr = dyn_cast<StringAttr>(defaultDeviceAttr)) {
    return makeDefaultDeviceNameRef(stringAttr.getContext(), stringAttr);
  } else if (auto integerAttr = dyn_cast<IntegerAttr>(defaultDeviceAttr)) {
    return makeDefaultDeviceOrdinalRef(integerAttr.getContext(),
                                       integerAttr.getInt());
  }
  return failure();
}

// Creates a named device global with the given attribute.
static FailureOr<FlatSymbolRefAttr>
createDeviceGlobal(Location loc, StringAttr name, Attribute targetAttr,
                   OpBuilder &moduleBuilder) {
  auto deviceType = moduleBuilder.getType<IREE::HAL::DeviceType>();
  auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
      loc, name, /*isMutable=*/false, deviceType);
  globalOp.setPrivate();

  TypedAttr attrValue;
  if (auto arrayAttr = dyn_cast<ArrayAttr>(targetAttr)) {
    if (arrayAttr.size() == 1) {
      auto typedAttr = dyn_cast<TypedAttr>(arrayAttr.getValue().front());
      if (typedAttr && isa<IREE::HAL::DeviceType>(typedAttr.getType())) {
        // Don't care exactly what the attribute is, only that it's a device.
        attrValue = typedAttr;
      }
    } else {
      // Expand arrays to selects.
      attrValue = moduleBuilder.getAttr<IREE::HAL::DeviceSelectAttr>(deviceType,
                                                                     arrayAttr);
    }
  } else if (auto typedAttr = dyn_cast<TypedAttr>(targetAttr)) {
    if (isa<IREE::HAL::DeviceType>(typedAttr.getType())) {
      // Don't care exactly what the attribute is, only that it's a device.
      attrValue = typedAttr;
    }
  }
  if (!attrValue) {
    return mlir::emitError(loc)
           << "module has invalid device targets specified; "
              "expected hal.device.targets to be an array of !hal.device "
              "initialization attributes or a dictionary with named values";
  }

  globalOp.setInitialValueAttr(attrValue);
  return FlatSymbolRefAttr::get(globalOp);
}

// Creates one or more device globals based on the specified targets and returns
// the "default" device (usually just the first one specified).
static FailureOr<FlatSymbolRefAttr> createDeviceGlobals(mlir::ModuleOp moduleOp,
                                                        Attribute targetsAttr) {
  auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

  FlatSymbolRefAttr firstDeviceRef;
  if (auto dictAttr = dyn_cast<DictionaryAttr>(targetsAttr)) {
    for (auto namedTargetsAttr : dictAttr.getValue()) {
      auto deviceRefOr =
          createDeviceGlobal(moduleOp.getLoc(), namedTargetsAttr.getName(),
                             namedTargetsAttr.getValue(), moduleBuilder);
      if (failed(deviceRefOr)) {
        return failure();
      } else if (!firstDeviceRef) {
        firstDeviceRef = *deviceRefOr;
      }
    }
  } else if (auto arrayAttr = dyn_cast<ArrayAttr>(targetsAttr)) {
    for (auto [i, ordinalTargetsAttr] : llvm::enumerate(arrayAttr.getValue())) {
      auto deviceRefOr =
          createDeviceGlobal(moduleOp.getLoc(),
                             moduleBuilder.getStringAttr(
                                 StringRef("__device_") + std::to_string(i)),
                             ordinalTargetsAttr, moduleBuilder);
      if (failed(deviceRefOr)) {
        return failure();
      } else if (!firstDeviceRef) {
        firstDeviceRef = *deviceRefOr;
      }
    }
  } else {
    return moduleOp.emitError()
           << "unexpected `hal.device.targets` attribute; must be a dictionary "
              "of named devices or an array of devices to use by ordinal";
  }

  return firstDeviceRef;
}

// Assigns the default device affinity to all top level ops that don't already
// have one set.
static void assignDefaultDeviceAffinity(mlir::ModuleOp moduleOp,
                                        FlatSymbolRefAttr defaultDeviceRef) {
  auto affinityAttr = IREE::HAL::DeviceAffinityAttr::get(
      moduleOp.getContext(), defaultDeviceRef, /*queue_mask=*/-1ll);

  // Default on the module that applies to any ops that don't otherwise have a
  // placement. Ideally we never need this but some programs may take/return no
  // tensors or have tensors come from unattributed containers (lists/dicts).
  moduleOp->setAttr("stream.affinity.default", affinityAttr);

  // Set all arg/results to route through the default device unless they've
  // already been assigned.
  auto affinityName = StringAttr::get(moduleOp.getContext(), "stream.affinity");
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    if (funcOp.isPublic()) {
      for (auto arg : funcOp.getArguments()) {
        if (isa<IREE::Stream::AffinityTypeInterface>(arg.getType())) {
          if (!funcOp.getArgAttr(arg.getArgNumber(), affinityName)) {
            funcOp.setArgAttr(arg.getArgNumber(), affinityName, affinityAttr);
          }
        }
      }
      for (auto result : llvm::enumerate(funcOp.getResultTypes())) {
        if (isa<IREE::Stream::AffinityTypeInterface>(result.value())) {
          if (!funcOp.getResultAttr(result.index(), affinityName)) {
            funcOp.setResultAttr(result.index(), affinityName, affinityAttr);
          }
        }
      }
    }
  }
}

struct MaterializeTargetDevicesPass
    : public IREE::HAL::impl::MaterializeTargetDevicesPassBase<
          MaterializeTargetDevicesPass> {
  using IREE::HAL::impl::MaterializeTargetDevicesPassBase<
      MaterializeTargetDevicesPass>::MaterializeTargetDevicesPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Only materialize devices if there's a module-level attribute specified.
    FlatSymbolRefAttr defaultDeviceRef;
    auto deviceTargetAttrs = moduleOp->getAttr("hal.device.targets");
    if (deviceTargetAttrs) {
      moduleOp->removeAttr("hal.device.targets");

      // Create the globals and get the default device.
      auto firstDeviceOr = createDeviceGlobals(moduleOp, deviceTargetAttrs);
      if (failed(firstDeviceOr)) {
        // Fails if invalid attributes.
        return signalPassFailure();
      }
      defaultDeviceRef = *firstDeviceOr;
    }

    // Select the default device from what the user specified or from the first
    // created.
    auto defaultDeviceAttr = moduleOp->getAttr("hal.device.default");
    if (defaultDeviceAttr) {
      // Always prefer the explicitly specified default device.
      moduleOp->removeAttr("hal.device.default");
      auto defaultDeviceRefOr = makeDefaultDeviceAttrRef(defaultDeviceAttr);
      if (failed(defaultDeviceRefOr)) {
        moduleOp.emitError() << "invalid `hal.device.default` value, must be "
                                "an ordinal or a name";
        return signalPassFailure();
      }
      defaultDeviceRef = *defaultDeviceRefOr;
    } else if (!defaultDevice.empty()) {
      // Fallback to the option specified, if any provided.
      long long defaultDeviceOrdinal = 0;
      if (!llvm::getAsSignedInteger(defaultDevice, 10, defaultDeviceOrdinal)) {
        defaultDeviceRef =
            makeDefaultDeviceOrdinalRef(&getContext(), defaultDeviceOrdinal);
      } else {
        defaultDeviceRef =
            makeDefaultDeviceNameRef(&getContext(), defaultDevice);
      }
    }

    // Assign affinities to all top level ops that don't already have one set.
    if (defaultDeviceRef) {
      assignDefaultDeviceAffinity(moduleOp, defaultDeviceRef);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
