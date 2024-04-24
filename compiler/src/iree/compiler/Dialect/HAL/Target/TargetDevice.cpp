// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::iree_compiler::IREE::HAL {

// virtual
Value TargetDevice::buildDeviceTargetMatch(
    Location loc, Value device, IREE::HAL::DeviceTargetAttr targetAttr,
    OpBuilder &builder) const {
  return buildDeviceIDAndExecutableFormatsMatch(
      loc, device, targetAttr.getDeviceID(), targetAttr.getExecutableTargets(),
      builder);
}

Value buildDeviceIDAndExecutableFormatsMatch(
    Location loc, Value device, StringRef deviceIDPattern,
    ArrayRef<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs,
    OpBuilder &builder) {
  // Match first on the device ID, as that's the top-level filter.
  Value idMatch = IREE::HAL::DeviceQueryOp::createI1(
      loc, device, "hal.device.id", deviceIDPattern, builder);

  // If there are executable formats defined we should check at least one of
  // them is supported.
  if (executableTargetAttrs.empty()) {
    return idMatch; // just device ID
  } else {
    auto ifOp = builder.create<scf::IfOp>(loc, builder.getI1Type(), idMatch,
                                          true, true);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    Value anyFormatMatch = buildExecutableFormatMatch(
        loc, device, executableTargetAttrs, thenBuilder);
    thenBuilder.create<scf::YieldOp>(loc, anyFormatMatch);
    auto elseBuilder = ifOp.getElseBodyBuilder();
    Value falseValue = elseBuilder.create<arith::ConstantIntOp>(loc, 0, 1);
    elseBuilder.create<scf::YieldOp>(loc, falseValue);
    return ifOp.getResult(0);
  }
}

Value buildExecutableFormatMatch(
    Location loc, Value device,
    ArrayRef<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs,
    OpBuilder &builder) {
  if (executableTargetAttrs.empty())
    return builder.create<arith::ConstantIntOp>(loc, 1, 1);
  Value anyFormatMatch;
  for (auto executableTargetAttr : executableTargetAttrs) {
    Value formatMatch = IREE::HAL::DeviceQueryOp::createI1(
        loc, device, "hal.executable.format",
        executableTargetAttr.getFormat().getValue(), builder);
    if (!anyFormatMatch) {
      anyFormatMatch = formatMatch;
    } else {
      anyFormatMatch =
          builder.create<arith::OrIOp>(loc, anyFormatMatch, formatMatch);
    }
  }
  return anyFormatMatch;
}

} // namespace mlir::iree_compiler::IREE::HAL
