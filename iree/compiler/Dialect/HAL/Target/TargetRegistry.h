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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

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
  TargetBackendRegistration(StringRef name, CreateTargetBackendFn fn);

  std::shared_ptr<TargetBackend> acquire();

 private:
  CreateTargetBackendFn initFn;
  std::once_flag initFlag;
  std::shared_ptr<TargetBackend> cachedValue;
};

// Returns a list of registered target backends.
std::vector<std::string> getRegisteredTargetBackends();

// Returns the target backend with the given name.
std::shared_ptr<TargetBackend> getTargetBackend(StringRef targetName);

// Returns one backend per entry in |targetNames|.
SmallVector<std::shared_ptr<TargetBackend>> getTargetBackends(
    ArrayRef<std::string> targetNames);

// Returns a sorted uniqued set of target backends used in the executable.
SmallVector<std::string> gatherExecutableTargetNames(
    IREE::HAL::ExecutableOp executableOp);

// Returns a sorted uniqued set of target backends used in the entire module.
SmallVector<std::string> gatherExecutableTargetNames(mlir::ModuleOp moduleOp);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETREGISTRY_H_
