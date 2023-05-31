// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"

#include <algorithm>

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Returns the static registry of translator names to translation functions.
static TargetBackendRegistry &getMutableTargetRegistry() {
  static TargetBackendRegistry global;
  return global;
}

const TargetBackendRegistry &TargetBackendRegistry::getGlobal() {
  return getMutableTargetRegistry();
}

TargetBackendRegistration::TargetBackendRegistration(llvm::StringRef name,
                                                     CreateTargetBackendFn fn,
                                                     bool registerStaticGlobal)
    : initFn(std::move(fn)) {
  if (registerStaticGlobal) {
    auto &registry = getMutableTargetRegistry();
    if (registry.registrations.contains(name)) {
      llvm::report_fatal_error(
          "Attempting to overwrite an existing translation backend");
    }
    assert(initFn && "Attempting to register an empty translation function");
    registry.registrations[name] = this;
  }
}

std::shared_ptr<TargetBackend> TargetBackendRegistration::acquire() {
  std::call_once(initFlag, [&]() { cachedValue = initFn(); });
  return cachedValue;
}

void TargetBackendRegistry::mergeFrom(const TargetBackendList &targets) {
  for (auto &it : targets.entries) {
    if (registrations.contains(it.first)) {
      llvm::report_fatal_error(
          "Attempting to overwrite an existing translation backend");
    }
    auto registration = std::make_unique<TargetBackendRegistration>(
        it.first, it.second, /*registerStaticGlobal=*/false);
    registrations[it.first] = registration.get();
    ownedRegistrations.push_back(std::move(registration));
  }
}

void TargetBackendRegistry::mergeFrom(const TargetBackendRegistry &registry) {
  for (auto &it : registry.registrations) {
    if (registrations.contains(it.first())) {
      llvm::report_fatal_error(
          "Attempting to overwrite an existing translation backend");
    }
    registrations[it.first()] = it.second;
  }
}

std::vector<std::string> TargetBackendRegistry::getRegisteredTargetBackends()
    const {
  std::vector<std::string> result;
  for (auto &entry : registrations) {
    result.push_back(entry.getKey().str());
  }
  std::sort(result.begin(), result.end(),
            [](const auto &a, const auto &b) { return a < b; });
  return result;
}

std::shared_ptr<TargetBackend> TargetBackendRegistry::getTargetBackend(
    StringRef targetName) const {
  for (auto &entry : registrations) {
    if (entry.getKey() == targetName) {
      return entry.getValue()->acquire();
    }
  }
  return {};
}

SmallVector<std::shared_ptr<TargetBackend>>
TargetBackendRegistry::getTargetBackends(
    ArrayRef<std::string> targetNames) const {
  SmallVector<std::shared_ptr<TargetBackend>> matches;
  for (auto targetName : targetNames) {
    auto targetBackend = getTargetBackend(targetName);
    if (targetBackend) {
      matches.push_back(std::move(targetBackend));
    }
  }
  // To ensure deterministic builds we sort matches by name.
  std::sort(matches.begin(), matches.end(),
            [](const auto &a, const auto &b) { return a->name() < b->name(); });
  return matches;
}

SmallVector<std::string> gatherExecutableTargetNames(
    IREE::HAL::ExecutableOp executableOp) {
  SmallVector<std::string> targetNames;
  llvm::SmallDenseSet<StringRef> targets;
  executableOp.walk([&](IREE::HAL::ExecutableVariantOp variantOp) {
    auto targetName = variantOp.getTarget().getBackend().getValue();
    if (targets.insert(targetName).second) {
      targetNames.push_back(targetName.str());
    }
  });
  llvm::stable_sort(targetNames);
  return targetNames;
}

SmallVector<std::string> gatherExecutableTargetNames(mlir::ModuleOp moduleOp) {
  SmallVector<std::string> targetNames;
  llvm::stable_sort(targetNames);
  llvm::SmallDenseSet<StringRef> targets;
  moduleOp.walk([&](IREE::HAL::ExecutableOp executableOp) {
    executableOp.walk([&](IREE::HAL::ExecutableVariantOp variantOp) {
      auto targetName = variantOp.getTarget().getBackend().getValue();
      if (targets.insert(targetName).second) {
        targetNames.push_back(targetName.str());
      }
    });
  });
  return targetNames;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
