// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_TARGETDEVICE_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_TARGETDEVICE_H_

#include <optional>

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler::IREE::HAL {

class TargetRegistry;

// HAL device target interface.
class TargetDevice {
public:
  virtual ~TargetDevice() = default;

  // Returns the default device this backend targets. This may involve setting
  // defaults from flags and other environmental sources, and it may be
  // cross-targeting in a way that is not compatible with the host.
  virtual IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const = 0;

  // Similar to getDefaultDeviceTarget, but always returns a DeviceTargetAttr
  // that is configured for the host, regardless of if flags/environment were
  // configured to cross-target in some way.
  virtual std::optional<IREE::HAL::DeviceTargetAttr>
  getHostDeviceTarget(MLIRContext *context,
                      const TargetRegistry &targetRegistry) const {
    return {};
  }

  // Builds an expression that returns an i1 indicating whether the given
  // |device| matches the |targetAttr| requirements.
  virtual Value buildDeviceTargetMatch(Location loc, Value device,
                                       IREE::HAL::DeviceTargetAttr targetAttr,
                                       OpBuilder &builder) const;

  // Sets the shared usage bits for the given device set.
  virtual LogicalResult
  setSharedUsageBits(const SetVector<IREE::HAL::DeviceTargetAttr> &targets,
                     IREE::HAL::BufferUsageBitfield &bufferUsage) const;

  // TODO(benvanik): pipeline registration for specialization of host code at
  // various stages.
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETDEVICE_H_
