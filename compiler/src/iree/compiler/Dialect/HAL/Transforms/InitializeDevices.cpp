// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_INITIALIZEDEVICESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

// Converts an initialized device global to one with a util.initializer that
// performs the device initialization. The initializer is added immediately
// following the global in its parent op.
static void initializeDeviceGlobal(
    IREE::Util::GlobalOpInterface globalOp,
    IREE::HAL::DeviceInitializationAttrInterface initialValue,
    const IREE::HAL::TargetRegistry &targetRegistry) {
  auto loc = globalOp.getLoc();

  // Clear the initial value as we'll be initializing from the initializer.
  globalOp.setGlobalInitialValue({});

  // Build a new util.initializer.
  OpBuilder moduleBuilder(globalOp);
  moduleBuilder.setInsertionPointAfter(globalOp);
  auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
  auto *block = moduleBuilder.createBlock(&initializerOp.getBody());
  auto initializerBuilder = OpBuilder::atBlockBegin(block);

  // Get the device from the attribute builder; note that it may be null.
  Value enumeratedDevice = initialValue.buildDeviceEnumeration(
      loc,
      [&](Location loc, Value device, IREE::HAL::DeviceTargetAttr targetAttr,
          OpBuilder &builder) {
        auto targetDevice =
            targetRegistry.getTargetDevice(targetAttr.getDeviceID());
        return targetDevice ? targetDevice->buildDeviceTargetMatch(
                                  loc, device, targetAttr, builder)
                            : Value{};
      },
      initializerBuilder);

  // Check if the device is null and error out. We could support optional
  // devices that are allowed to be null but don't support that anywhere else in
  // the compiler today and may never want to. If selecting from multiple
  // devices queries can be used to detect what the selected device was and
  // those will be memoized.
  Value nullDevice = initializerBuilder.create<IREE::Util::NullOp>(
      loc, enumeratedDevice.getType());
  Value isNull = initializerBuilder.create<IREE::Util::CmpEQOp>(
      loc, enumeratedDevice, nullDevice);
  initializerBuilder.create<scf::IfOp>(
      loc, isNull, [&](OpBuilder &thenBuilder, Location thenLoc) {
        Value status = thenBuilder.create<arith::ConstantIntOp>(
            thenLoc, static_cast<int64_t>(IREE::Util::StatusCode::Incompatible),
            32);
        std::string str;
        {
          llvm::raw_string_ostream os(str);
          os << "HAL device `" << globalOp.getGlobalName().getValue()
             << "` not found or unavailable: ";
          initialValue.printStatusDescription(os);
        }
        thenBuilder.create<IREE::Util::StatusCheckOkOp>(thenLoc, status, str);
        thenBuilder.create<scf::YieldOp>(thenLoc);
      });

  // Store the device back to the global to complete initialization.
  globalOp.createStoreOp(loc, enumeratedDevice, initializerBuilder);
  initializerBuilder.create<IREE::Util::ReturnOp>(loc);
}

//===----------------------------------------------------------------------===//
// --iree-hal-initialize-devices
//===----------------------------------------------------------------------===//

struct InitializeDevicesPass
    : public IREE::HAL::impl::InitializeDevicesPassBase<InitializeDevicesPass> {
  using IREE::HAL::impl::InitializeDevicesPassBase<
      InitializeDevicesPass>::InitializeDevicesPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      auto initialValue =
          dyn_cast_if_present<IREE::HAL::DeviceInitializationAttrInterface>(
              globalOp.getGlobalInitialValue());
      if (initialValue) {
        initializeDeviceGlobal(globalOp, initialValue, *targetRegistry.value);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
