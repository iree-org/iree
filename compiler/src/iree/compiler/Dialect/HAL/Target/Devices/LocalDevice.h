// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_DEVICE_LOCALDEVICE_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_DEVICE_LOCALDEVICE_H_

#include <string>
#include <vector>

#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler::IREE::HAL {

class LocalDevice final : public TargetDevice {
public:
  struct Options {
    // A list of default target backends for local devices.
    std::vector<std::string> defaultTargetBackends;
    // A list of default host backends for local devices.
    std::vector<std::string> defaultHostBackends;

    void bindOptions(OptionsBinder &binder);
    using FromFlags = OptionsFromFlags<Options>;
  };

  explicit LocalDevice(const Options options);

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override;

  std::optional<IREE::HAL::DeviceTargetAttr>
  getHostDeviceTarget(MLIRContext *context,
                      const TargetRegistry &targetRegistry) const override;

  Value buildDeviceTargetMatch(Location loc, Value device,
                               IREE::HAL::DeviceTargetAttr targetAttr,
                               OpBuilder &builder) const override;
  LogicalResult setSharedUsageBits(
      const SetVector<IREE::HAL::DeviceTargetAttr> &targets,
      IREE::HAL::BufferUsageBitfield &bufferUsage) const override;

private:
  const Options options;
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_DEVICE_LOCALDEVICE_H_
