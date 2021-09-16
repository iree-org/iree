// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_BYTECODEMODULETARGET_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_BYTECODEMODULETARGET_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// Defines the output format of the bytecode module.
enum class BytecodeOutputFormat {
  // FlatBuffer binary with a BytecodeModuleDef as the root.
  kFlatBufferBinary,
  // FlatBuffer text using reflection. Not designed to be deserialized.
  kFlatBufferText,
  // MLIR text of the VM module.
  kMlirText,
  // MLIR text with annotations approximating what is in the FlatBuffer binary.
  // Useful for debugging and testing without needing to do filechecks against
  // the FlatBuffer text format (which can be non-deterministic).
  kAnnotatedMlirText,
};

// Options that can be provided to bytecode translation.
struct BytecodeTargetOptions {
  // Format of the module written to the output stream.
  BytecodeOutputFormat outputFormat = BytecodeOutputFormat::kFlatBufferBinary;

  // Run basic CSE/inlining/etc passes prior to serialization.
  bool optimize = true;

  // Dump a VM MLIR file and annotate source locations with it.
  // This allows for the runtime to serve stack traces referencing both the
  // original source locations and the VM IR.
  std::string sourceListing;

  // Strips source map information.
  bool stripSourceMap = false;
  // Strips vm ops with the VM_DebugOnly trait.
  bool stripDebugOps = false;

  // Enables the output .vmfb to be inspected as a ZIP file.
  // This is only useful for debugging and should be disabled otherwise.
  bool emitPolyglotZip = false;
};

// Translates a vm.module to a bytecode module flatbuffer.
// See iree/schemas/bytecode_module_def.fbs for the description of the
// serialized module format.
//
// Exposed via the --iree-vm-ir-to-bytecode-module translation.
LogicalResult translateModuleToBytecode(IREE::VM::ModuleOp moduleOp,
                                        BytecodeTargetOptions targetOptions,
                                        llvm::raw_ostream &output);
LogicalResult translateModuleToBytecode(mlir::ModuleOp outerModuleOp,
                                        BytecodeTargetOptions targetOptions,
                                        llvm::raw_ostream &output);

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_BYTECODEMODULETARGET_H_
