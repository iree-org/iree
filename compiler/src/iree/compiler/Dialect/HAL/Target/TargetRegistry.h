// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_TARGETREGISTRY_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_TARGETREGISTRY_H_

#include <mutex>
#include <string>
#include <vector>

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::iree_compiler::IREE::HAL {

using CreateTargetBackendFn = std::function<std::shared_ptr<TargetBackend>()>;

// Registers an executable translation target backend creation function.
//
// For example:
//   llvm-aot-x86_64
//   llvm-aot-armv8-dotprod
//   llvm-jit
//   vulkan-v1.1-low
//   vulkan-v1.1-high
class TargetBackendRegistration {
public:
  // TODO: Remove the registerStaticGlobal mode once callers are migrated.
  TargetBackendRegistration(StringRef name, CreateTargetBackendFn fn,
                            bool registerStaticGlobal = true);

  std::shared_ptr<TargetBackend> acquire();

private:
  CreateTargetBackendFn initFn;
  std::once_flag initFlag;
  std::shared_ptr<TargetBackend> cachedValue;
};

// A registry of target
class TargetBackendList {
public:
  void add(llvm::StringRef name, CreateTargetBackendFn fn) {
    entries.push_back(std::make_pair(name, fn));
  }

private:
  llvm::SmallVector<std::pair<llvm::StringRef, CreateTargetBackendFn>> entries;
  friend class TargetBackendRegistry;
};

// A concrete target backend registry.
class TargetBackendRegistry {
public:
  // Merge from a list of of targets. The registry will own the registration
  // entries.
  void mergeFrom(const TargetBackendList &targets);
  // Initialize from an existing registry. This registry will not own the
  // backing registration entries. The source registry must remain live for the
  // life of this.
  void mergeFrom(const TargetBackendRegistry &registry);

  // Returns the read-only global registry. This is used by passes which depend
  // on it from their default constructor.
  static const TargetBackendRegistry &getGlobal();

  // Returns a list of registered target backends.
  std::vector<std::string> getRegisteredTargetBackends() const;

  // Returns the target backend with the given name.
  std::shared_ptr<TargetBackend> getTargetBackend(StringRef targetName) const;

  // Returns one backend per entry in |targetNames|.
  SmallVector<std::shared_ptr<TargetBackend>>
  getTargetBackends(ArrayRef<std::string> targetNames) const;

private:
  llvm::StringMap<TargetBackendRegistration *> registrations;
  llvm::SmallVector<std::unique_ptr<TargetBackendRegistration>>
      ownedRegistrations;

  friend class TargetBackendRegistration;
};

// Returns a sorted uniqued set of target backends used in the executable.
SmallVector<std::string>
gatherExecutableTargetNames(IREE::HAL::ExecutableOp executableOp);

// Returns a sorted uniqued set of target backends used in the entire module.
SmallVector<std::string> gatherExecutableTargetNames(mlir::ModuleOp moduleOp);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETREGISTRY_H_
