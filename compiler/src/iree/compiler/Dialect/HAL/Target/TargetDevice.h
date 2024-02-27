// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_TARGETDEVICE_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_TARGETDEVICE_H_

#include <optional>
#include <string>

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler::IREE::HAL {

// HAL device target interface.
class TargetDevice {
public:
  virtual ~TargetDevice() = default;

  // Returns a name for the backend used to differentiate between other targets.
  virtual std::string name() const = 0;

  // Returns the name of the runtime device for this backend.
  // TODO(benvanik): remove this once we can properly specify targets.
  virtual std::string deviceID() const { return name(); }

  // Registers dependent dialects for the TargetDevice.
  // Mirrors the method on mlir::Pass of the same name. A TargetDevice is
  // expected to register the dialects it will create entities for (Operations,
  // Types, Attributes).
  virtual void getDependentDialects(DialectRegistry &registry) const {}

  // Returns the default device this backend targets. This may involve setting
  // defaults from flags and other environmental sources, and it may be
  // cross-targeting in a way that is not compatible with the host.
  virtual IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context) const = 0;

  // Similar to getDefaultDeviceTarget, but always returns a DeviceTargetAttr
  // that is configured for the host, regardless of if flags/environment were
  // configured to cross-target in some way.
  //
  virtual std::optional<IREE::HAL::DeviceTargetAttr>
  getHostDeviceTarget(MLIRContext *context) const {
    return {};
  }
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETDEVICE_H_
