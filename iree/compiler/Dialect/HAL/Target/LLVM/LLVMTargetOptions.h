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

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMTARGETOPTIONS_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMTARGETOPTIONS_H_

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetOptions.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

struct LLVMTargetOptions {
  llvm::PipelineTuningOptions pipelineTuningOptions;
  llvm::PassBuilder::OptimizationLevel optLevel;
  llvm::TargetOptions options;
  std::string targetTriple;
};

// Returns LLVMTargetOptions struct intialized with the
// iree-hal-llvm-ir-* flags.
LLVMTargetOptions getLLVMTargetOptionsFromFlags();

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMTARGETOPTIONS_H_
