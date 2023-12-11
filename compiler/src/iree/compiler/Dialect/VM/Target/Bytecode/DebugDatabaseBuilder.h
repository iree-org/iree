// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_DEBUGDATABASEBUILDER_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_DEBUGDATABASEBUILDER_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/bytecode_module_def_builder.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"

namespace mlir::iree_compiler::IREE::VM {

struct BytecodeLocation {
  int32_t bytecodeOffset;
  Location location;
};

struct FunctionSourceMap {
  std::string localName;
  SmallVector<BytecodeLocation> locations;
};

class DebugDatabaseBuilder {
public:
  // Appends a function source map entry to the debug database.
  void addFunctionSourceMap(IREE::VM::FuncOp funcOp,
                            FunctionSourceMap sourceMap);

  // Finishes construction of the debug database and emits it to the FlatBuffer.
  iree_vm_DebugDatabaseDef_ref_t build(FlatbufferBuilder &fbb);

private:
  // Function source maps ordered by function ordinal.
  SmallVector<FunctionSourceMap> functionSourceMaps;
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_DEBUGDATABASEBUILDER_H_
