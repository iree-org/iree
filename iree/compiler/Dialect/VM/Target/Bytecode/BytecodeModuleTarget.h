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

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_BYTECODEMODULETARGET_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_BYTECODEMODULETARGET_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

// Defines the output format of the bytecode module.
enum class BytecodeOutputFormat {
  // FlatBuffer binary with a BytecodeModuleDef as the root.
  kFlatBufferBinary,
  // FlatBuffer text using reflection. Not designed to be deserialized.
  kFlatBufferText,
  // MLIR text with annotations approximating what is in the FlatBuffer binary.
  // Useful for debugging and testing without needing to do filechecks against
  // the FlatBuffer text format (which can be non-deterministic).
  kMlirText,
};

// Options that can be provided to bytecode translation.
struct BytecodeTargetOptions {
  // Format of the module written to the output stream.
  BytecodeOutputFormat outputFormat = BytecodeOutputFormat::kFlatBufferBinary;

  // Run basic CSE/inlining/etc passes prior to serialization.
  bool optimize = true;

  // Strips all internal symbol names. Import and export names will remain.
  bool stripSymbols = false;
  // Strips source map information.
  bool stripSourceMap = false;
  // Strips vm ops with the VM_DebugOnly trait.
  bool stripDebugOps = false;
};

// Translates a vm.module to a bytecode module flatbuffer.
// See iree/schemas/bytecode_module_def.fbs for the description of the
// serialized module format.
//
// Exposed via the --vm-mlir-to-bytecode-module translation.
LogicalResult translateModuleToBytecode(BytecodeTargetOptions targetOptions,
                                        IREE::VM::ModuleOp moduleOp,
                                        llvm::raw_ostream &output);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_BYTECODEMODULETARGET_H_
