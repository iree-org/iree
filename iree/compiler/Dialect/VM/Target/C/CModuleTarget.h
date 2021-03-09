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

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_C_CMODULETARGET_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_C_CMODULETARGET_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// Defines the output format of the c module.
enum class COutputFormat {
  // C code.
  kCode,
  // MLIR text of the VM module mixed with emitc operations.
  kMlirText,
};

// Options that can be provided to c code translation.
struct CTargetOptions {
  // Format of the module written to the output stream.
  COutputFormat outputFormat = COutputFormat::kCode;

  // Run basic CSE/inlining/etc passes prior to serialization.
  bool optimize = true;

  // Strips vm ops with the VM_DebugOnly trait.
  bool stripDebugOps = false;
};

// Translates a vm.module to a c module.
//
// Exposed via the --iree-vm-ir-to-c-module translation.
LogicalResult translateModuleToC(IREE::VM::ModuleOp moduleOp,
                                 CTargetOptions targetOptions,
                                 llvm::raw_ostream &output);
LogicalResult translateModuleToC(mlir::ModuleOp outerModuleOp,
                                 CTargetOptions targetOptions,
                                 llvm::raw_ostream &output);

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_TARGET_C_CMODULETARGET_H_
