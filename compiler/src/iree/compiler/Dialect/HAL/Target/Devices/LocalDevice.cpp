// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/Devices/LocalDevice.h"

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::IREE::HAL::LocalDevice::Options);

namespace mlir::iree_compiler::IREE::HAL {

void LocalDevice::Options::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory optionsCategory(
      "IREE HAL local device options");

  binder.list<std::string>(
      "iree-hal-local-target-device-backends", defaultTargetBackends,
      llvm::cl::desc(
          "Default target backends for local device executable compilation."),
      llvm::cl::ZeroOrMore, llvm::cl::cat(optionsCategory));
  binder.list<std::string>(
      "iree-hal-local-host-device-backends", defaultHostBackends,
      llvm::cl::desc(
          "Default host backends for local device executable compilation."),
      llvm::cl::ZeroOrMore, llvm::cl::cat(optionsCategory));
}

LocalDevice::LocalDevice(const LocalDevice::Options options)
    : options(options) {}

IREE::HAL::DeviceTargetAttr LocalDevice::getDefaultDeviceTarget(
    MLIRContext *context, const TargetRegistry &targetRegistry) const {
  Builder b(context);
  SmallVector<NamedAttribute> configItems;

  // TODO(benvanik): flags for common queries.

  auto configAttr = b.getDictionaryAttr(configItems);

  SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  for (auto backendName : options.defaultTargetBackends) {
    auto targetBackend = targetRegistry.getTargetBackend(backendName);
    if (!targetBackend) {
      llvm::errs() << "Default target backend not registered: " << backendName
                   << "\n";
      return {};
    }
    targetBackend->getDefaultExecutableTargets(context, "local", configAttr,
                                               executableTargetAttrs);
  }

  return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("local"),
                                          configAttr, executableTargetAttrs);
}

std::optional<IREE::HAL::DeviceTargetAttr>
LocalDevice::getHostDeviceTarget(MLIRContext *context,
                                 const TargetRegistry &targetRegistry) const {
  Builder b(context);
  SmallVector<NamedAttribute> configItems;

  // TODO(benvanik): flags for overrides or ask LLVM for info about the host.

  auto configAttr = b.getDictionaryAttr(configItems);

  SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  for (auto backendName : options.defaultHostBackends) {
    auto targetBackend = targetRegistry.getTargetBackend(backendName);
    if (!targetBackend) {
      llvm::errs() << "Default host backend not registered: " << backendName
                   << "\n";
      return std::nullopt;
    }
    targetBackend->getHostExecutableTargets(context, "local", configAttr,
                                            executableTargetAttrs);
  }

  return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("local"),
                                          configAttr, executableTargetAttrs);
}

Value LocalDevice::buildDeviceTargetMatch(
    Location loc, Value device, IREE::HAL::DeviceTargetAttr targetAttr,
    OpBuilder &builder) const {
  return buildDeviceIDAndExecutableFormatsMatch(
      loc, device, "local*", targetAttr.getExecutableTargets(), builder);
}

} // namespace mlir::iree_compiler::IREE::HAL
