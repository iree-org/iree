// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"

#include <algorithm>

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// TargetRegistration
//===----------------------------------------------------------------------===//

// Returns the static registry of translator names to translation functions.
static TargetRegistry &getMutableTargetRegistry() {
  static TargetRegistry global;
  return global;
}

TargetBackendRegistration::TargetBackendRegistration(
    llvm::StringRef name, TargetFactoryFn<TargetBackend> fn,
    bool registerStaticGlobal)
    : TargetRegistration<TargetBackend>(std::move(fn)) {
  if (registerStaticGlobal) {
    auto &registry = getMutableTargetRegistry();
    if (registry.backendRegistrations.contains(name)) {
      llvm::report_fatal_error(
          "Attempting to overwrite an existing translation backend");
    }
    assert(initFn &&
           "Attempting to register an empty backend factory function");
    registry.backendRegistrations[name] = this;
  }
}

TargetDeviceRegistration::TargetDeviceRegistration(
    llvm::StringRef name, TargetFactoryFn<TargetDevice> fn,
    bool registerStaticGlobal)
    : TargetRegistration<TargetDevice>(std::move(fn)) {
  if (registerStaticGlobal) {
    auto &registry = getMutableTargetRegistry();
    if (registry.deviceRegistrations.contains(name)) {
      llvm::report_fatal_error(
          "Attempting to overwrite an existing target device");
    }
    assert(initFn && "Attempting to register an empty device factory function");
    registry.deviceRegistrations[name] = this;
  }
}

//===----------------------------------------------------------------------===//
// TargetRegistry
//===----------------------------------------------------------------------===//

const TargetRegistry &TargetRegistry::getGlobal() {
  return getMutableTargetRegistry();
}

void TargetRegistry::mergeFrom(const TargetBackendList &targetBackends) {
  for (auto &it : targetBackends.entries) {
    if (backendRegistrations.contains(it.first)) {
      llvm::report_fatal_error(
          "Attempting to overwrite an existing translation backend");
    }
    auto registration = std::make_unique<TargetBackendRegistration>(
        it.first, it.second, /*registerStaticGlobal=*/false);
    backendRegistrations[it.first] = registration.get();
    ownedBackendRegistrations.push_back(std::move(registration));
  }
}

void TargetRegistry::mergeFrom(const TargetDeviceList &targetDevices) {
  for (auto &it : targetDevices.entries) {
    if (deviceRegistrations.contains(it.first)) {
      llvm::report_fatal_error("Attempting to overwrite an existing device");
    }
    auto registration = std::make_unique<TargetDeviceRegistration>(
        it.first, it.second, /*registerStaticGlobal=*/false);
    deviceRegistrations[it.first] = registration.get();
    ownedDeviceRegistrations.push_back(std::move(registration));
  }
}

void TargetRegistry::mergeFrom(const TargetRegistry &registry) {
  for (auto &it : registry.deviceRegistrations) {
    if (deviceRegistrations.contains(it.first())) {
      llvm::report_fatal_error("Attempting to overwrite an existing device");
    }
    deviceRegistrations[it.first()] = it.second;
  }
  for (auto &it : registry.backendRegistrations) {
    if (backendRegistrations.contains(it.first())) {
      llvm::report_fatal_error(
          "Attempting to overwrite an existing translation backend");
    }
    backendRegistrations[it.first()] = it.second;
  }
}

std::vector<std::string> TargetRegistry::getRegisteredTargetBackends() const {
  std::vector<std::string> result;
  for (auto &entry : backendRegistrations) {
    result.push_back(entry.getKey().str());
  }
  std::sort(result.begin(), result.end(),
            [](const auto &a, const auto &b) { return a < b; });
  return result;
}

std::vector<std::string> TargetRegistry::getRegisteredTargetDevices() const {
  std::vector<std::string> result;
  for (auto &entry : deviceRegistrations) {
    result.push_back(entry.getKey().str());
  }
  std::sort(result.begin(), result.end(),
            [](const auto &a, const auto &b) { return a < b; });
  return result;
}

std::shared_ptr<TargetBackend>
TargetRegistry::getTargetBackend(StringRef targetName) const {
  for (auto &entry : backendRegistrations) {
    if (entry.getKey() == targetName) {
      return entry.getValue()->acquire();
    }
  }
  return {};
}

std::shared_ptr<TargetDevice>
TargetRegistry::getTargetDevice(StringRef targetName) const {
  for (auto &entry : deviceRegistrations) {
    if (entry.getKey() == targetName) {
      return entry.getValue()->acquire();
    }
  }
  return {};
}

SmallVector<std::shared_ptr<TargetBackend>>
TargetRegistry::getTargetBackends(ArrayRef<std::string> targetNames) const {
  SmallVector<std::pair<std::string, std::shared_ptr<TargetBackend>>> matches;
  for (auto &targetName : targetNames) {
    auto targetBackend = getTargetBackend(targetName);
    if (targetBackend) {
      matches.push_back(std::make_pair(targetName, std::move(targetBackend)));
    }
  }
  // To ensure deterministic builds we sort matches by name.
  std::sort(matches.begin(), matches.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  return llvm::to_vector(llvm::map_range(
      matches, [](auto match) { return std::move(match.second); }));
}

SmallVector<std::shared_ptr<TargetDevice>>
TargetRegistry::getTargetDevices(ArrayRef<std::string> targetNames) const {
  SmallVector<std::pair<std::string, std::shared_ptr<TargetDevice>>> matches;
  for (auto &targetName : targetNames) {
    auto targetDevice = getTargetDevice(targetName);
    if (targetDevice) {
      matches.push_back(std::make_pair(targetName, std::move(targetDevice)));
    }
  }
  // To ensure deterministic builds we sort matches by name.
  std::sort(matches.begin(), matches.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  return llvm::to_vector(llvm::map_range(
      matches, [](auto match) { return std::move(match.second); }));
}

} // namespace mlir::iree_compiler::IREE::HAL

//===----------------------------------------------------------------------===//
// TargetRegistryRef
//===----------------------------------------------------------------------===//

namespace llvm::cl {
template class basic_parser<TargetRegistryRef>;
} // namespace llvm::cl

using TargetRegistryRef = llvm::cl::TargetRegistryRef;

// Return true on error.
bool llvm::cl::parser<TargetRegistryRef>::parse(Option &O, StringRef ArgName,
                                                StringRef Arg,
                                                TargetRegistryRef &Val) {
  // We ignore Arg here and just use the global registry. We could parse a list
  // of target backends and create a new registry with just that subset but
  // ownership gets tricky.
  if (Arg != "global")
    return true;
  Val.value = &mlir::iree_compiler::IREE::HAL::TargetRegistry::getGlobal();
  return false;
}

void llvm::cl::parser<TargetRegistryRef>::printOptionDiff(
    const Option &O, TargetRegistryRef V, const OptVal &Default,
    size_t GlobalWidth) const {
  printOptionName(O, GlobalWidth);
  std::string Str = "global";
  outs() << "= " << Str;
  outs().indent(2) << " (default: global)\n";
}

void llvm::cl::parser<TargetRegistryRef>::anchor() {}
