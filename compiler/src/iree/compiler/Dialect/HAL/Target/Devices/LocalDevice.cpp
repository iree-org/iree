// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/Devices/LocalDevice.h"

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"

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
  auto deviceConfigAttr = b.getDictionaryAttr({});
  auto executableConfigAttr = b.getDictionaryAttr({});

  SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  for (auto backendName : options.defaultTargetBackends) {
    auto targetBackend = targetRegistry.getTargetBackend(backendName);
    if (!targetBackend) {
      llvm::errs() << "Default target backend not registered: " << backendName
                   << "\n";
      return {};
    }
    targetBackend->getDefaultExecutableTargets(
        context, "local", executableConfigAttr, executableTargetAttrs);
  }

  return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("local"),
                                          deviceConfigAttr,
                                          executableTargetAttrs);
}

std::optional<IREE::HAL::DeviceTargetAttr>
LocalDevice::getHostDeviceTarget(MLIRContext *context,
                                 const TargetRegistry &targetRegistry) const {
  Builder b(context);
  auto deviceConfigAttr = b.getDictionaryAttr({});
  auto executableConfigAttr = b.getDictionaryAttr({});

  // Use the specified default host backends, if any.
  // If no host backends are specified we try to find the first that is.
  // **This is likely to be wrong**.
  //
  // TODO(benvanik): add a secondary registry on the LocalDevice for its
  // backend lists. We shouldn't have to ask the registry and use the legacy
  // device ID from the backend.
  //
  // TODO(benvanik): add an "isHostCompatible" on TargetBackend that returns
  // true when the provided executable target is able to be run on the host
  // (probably). We'd still need to filter to those compatible with the local
  // device (instead of say using the SPIR-V backend with the local device) but
  // would at least be able to handle multiple backends deterministically.
  std::vector<std::string> targetBackends = options.defaultHostBackends;
  if (targetBackends.empty()) {
    for (auto targetBackendName :
         targetRegistry.getRegisteredTargetBackends()) {
      auto targetBackend = targetRegistry.getTargetBackend(targetBackendName);
      if (targetBackend->getLegacyDefaultDeviceID() == "local") {
        targetBackends.push_back(targetBackendName);
        break; // first only
      }
    }
  }

  // Query the chosen target backends for their executable targets.
  SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  for (auto backendName : targetBackends) {
    auto targetBackend = targetRegistry.getTargetBackend(backendName);
    if (!targetBackend) {
      llvm::errs() << "Default host backend not registered: " << backendName
                   << "\n";
      return std::nullopt;
    }
    targetBackend->getHostExecutableTargets(
        context, "local", executableConfigAttr, executableTargetAttrs);
  }

  return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("local"),
                                          deviceConfigAttr,
                                          executableTargetAttrs);
}

Value LocalDevice::buildDeviceTargetMatch(
    Location loc, Value device, IREE::HAL::DeviceTargetAttr targetAttr,
    OpBuilder &builder) const {
  return IREE::HAL::DeviceTargetAttr::buildDeviceIDAndExecutableFormatsMatch(
      loc, device, "local*", targetAttr.getExecutableTargets(), builder);
}

LogicalResult LocalDevice::setSharedUsageBits(
    const SetVector<IREE::HAL::DeviceTargetAttr> &targets,
    IREE::HAL::BufferUsageBitfield &bufferUsage) const {
  for (auto targetAttr : targets) {
    // if the target is not local (self), we need to add the mapping persistent
    // usage bit.
    if (targetAttr.getDeviceID().getValue() != "local") {
      bufferUsage =
          bufferUsage | IREE::HAL::BufferUsageBitfield::MappingPersistent;
    }
  }
  return success();
}

} // namespace mlir::iree_compiler::IREE::HAL
