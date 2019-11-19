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

#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"
#include "iree/compiler/Translation/IREEVM.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Translation.h"

namespace mlir {
namespace iree_compiler {

static LogicalResult translateFromMLIRToVMBytecodeModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  // TODO(benvanik): parse flags.
  auto executableTargetOptions = IREE::HAL::ExecutableTargetOptions{};
  auto bytecodeTargetOptions = IREE::VM::getBytecodeTargetOptionsFromFlags();
  return translateFromMLIRToVMBytecodeModule(moduleOp, executableTargetOptions,
                                             bytecodeTargetOptions, output);
}

static TranslateFromMLIRRegistration toVMBytecodeModuleWithFlags(
    "iree-mlir-to-vm-bytecode-module",
    translateFromMLIRToVMBytecodeModuleWithFlags);

}  // namespace iree_compiler
}  // namespace mlir
