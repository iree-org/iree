// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

namespace mlir::iree_compiler::IREE::HAL {

// virtual
Value TargetDevice::buildDeviceTargetMatch(
    Location loc, Value device, IREE::HAL::DeviceTargetAttr targetAttr,
    OpBuilder &builder) const {
  return IREE::HAL::DeviceTargetAttr::buildDeviceIDAndExecutableFormatsMatch(
      loc, device, targetAttr.getDeviceID(), targetAttr.getExecutableTargets(),
      builder);
}

LogicalResult TargetDevice::setSharedUsageBits(
    const SetVector<IREE::HAL::DeviceTargetAttr> &targets,
    IREE::HAL::BufferUsageBitfield &bufferUsage) const {
  // If the TargetDevice does not implement this function, default to failure.
  return failure();
}

} // namespace mlir::iree_compiler::IREE::HAL
