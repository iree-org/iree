// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_EXECUTABLETARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_EXECUTABLETARGET_H_

#include <functional>
#include <string>
#include <vector>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Controls executable translation targets.
struct ExecutableTargetOptions {
  // TODO(benvanik): multiple targets of the same type, etc.
  std::vector<std::string> targets;
};

// Returns a ExecutableTargetOptions struct initialized with the
// --iree-hal-target-* flags.
ExecutableTargetOptions getExecutableTargetOptionsFromFlags();

// Registered function that given a |sourceOp| flow.executable will produce one
// or more serialized versions added to |targetOp|.
using ExecutableTargetFn = std::function<LogicalResult(
    IREE::Flow::ExecutableOp sourceOp, IREE::HAL::ExecutableOp targetOp,
    ExecutableTargetOptions executableOptions)>;

// Registers an executable translation function.
struct ExecutableTargetRegistration {
  ExecutableTargetRegistration(llvm::StringRef name,
                               const ExecutableTargetFn &fn);
};

// Returns a read-only reference to the translator registry.
const llvm::StringMap<ExecutableTargetFn> &getExecutableTargetRegistry();

// Returns executable target backend names matching the given pattern.
// This accepts wildcards in the form of '*' and '?' for any delimited value.
// '*' will match zero or more of any character and '?' will match exactly one
// of any character.
//
// For example,
// 'foo-*-bar' matches: 'foo-123-bar', 'foo-456-789-bar'
// 'foo-10?' matches: 'foo-101', 'foo-102'
std::vector<std::string> matchExecutableTargetNames(llvm::StringRef pattern);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_EXECUTABLETARGET_H_
