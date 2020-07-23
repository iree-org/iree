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

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMIRPASSES_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMIRPASSES_H_

#include <memory>

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Creates target machine form target options.
std::unique_ptr<llvm::TargetMachine> createTargetMachine(
    const LLVMTargetOptions& options);

// Creates and runs LLVMIR optimization passes defined in LLVMTargetOptions.
LogicalResult runLLVMIRPasses(const LLVMTargetOptions& options,
                              llvm::TargetMachine* machine,
                              llvm::Module* module);

// Emits compiled module obj for the target machine.
LogicalResult runEmitObjFilePasses(llvm::TargetMachine* machine,
                                   llvm::Module* module, std::string* objData);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMIRPASSES_H_
