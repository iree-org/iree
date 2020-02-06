// Copyright 2020 Google LLC
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
//

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_TARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_TARGET_H_

#include "iree/compiler/Dialect/HAL/Target/ExecutableTarget.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

struct LLVMTargetOptions {
  // TODO(ataei): Add LLVM compilation options.
};

// Returns LLVMTargetOptions struct intialized with the
// iree-hal-llvm-ir-* flags.
LLVMTargetOptions getLLVMTargetOptionsFromFlags();

// Translates an executable to the LLVM backends with the given options.
LogicalResult translateToLLVMExecutable(
    IREE::Flow::ExecutableOp sourceOp, IREE::HAL::ExecutableOp targetOp,
    ExecutableTargetOptions executableOptions, LLVMTargetOptions targetOptions);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_TARGET_H_
