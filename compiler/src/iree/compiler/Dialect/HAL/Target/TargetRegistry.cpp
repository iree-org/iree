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
static llvm::StringMap<TargetBackendRegistration *>
    &getMutableTargetRegistry() {
  static llvm::StringMap<TargetBackendRegistration *> registry;
  return registry;
}

TargetBackendRegistration::TargetBackendRegistration(llvm::StringRef name,
                                                     CreateTargetBackendFn fn)
    : initFn(std::move(fn)) {
  auto &registry = getMutableTargetRegistry();
  if (registry.count(name) > 0) {
    llvm::report_fatal_error(
        "Attempting to overwrite an existing translation backend");
  }
  assert(initFn && "Attempting to register an empty translation function");
  registry[name] = this;
}

std::shared_ptr<TargetBackend> TargetBackendRegistration::acquire() {
  std::call_once(initFlag, [&]() { cachedValue = initFn(); });
  return cachedValue;
}

const llvm::StringMap<TargetBackendRegistration *> &getTargetRegistry() {
  return getMutableTargetRegistry();
}

std::vector<std::string> getRegisteredTargetBackends() {
  std::vector<std::string> result;
  for (auto &entry : getTargetRegistry()) {
    result.push_back(entry.getKey().str());
  }
  std::sort(result.begin(), result.end(),
            [](const auto &a, const auto &b) { return a < b; });
  return result;
}

std::shared_ptr<TargetBackend> getTargetBackend(StringRef targetName) {
  for (auto &entry : getTargetRegistry()) {
    if (entry.getKey() == targetName) {
      return entry.getValue()->acquire();
    }
  }
  return {};
}

SmallVector<std::shared_ptr<TargetBackend>> getTargetBackends(
    ArrayRef<std::string> targetNames) {
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
