// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_TARGETREGISTRY_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_TARGETREGISTRY_H_

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

using CreateTargetBackendFn = std::function<std::unique_ptr<TargetBackend>()>;

// Registers an executable translation target backend creation function.
// The name will be matched using `matchTargetBackends` and convention is that
// the name is namespaced with -'s.
//
// For example:
//   llvm-aot-x86_64
//   llvm-aot-armv8-dotprod
//   llvm-jit
//   vulkan-v1.1-low
//   vulkan-v1.1-high
struct TargetBackendRegistration {
  TargetBackendRegistration(StringRef name, CreateTargetBackendFn fn);
};

// Returns a list of registered target backends.
std::vector<std::string> getRegisteredTargetBackends();

// Matches a set of |patterns| against the registry to return zero or more
// backends for each pattern.
//
// For example,
// 'foo-*-bar' matches: 'foo-123-bar', 'foo-456-789-bar'
// 'foo-10?' matches: 'foo-101', 'foo-102'
std::vector<std::unique_ptr<TargetBackend>> matchTargetBackends(
    ArrayRef<std::string> patterns);

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
