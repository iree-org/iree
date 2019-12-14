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

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_BYTECODEENCODER_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_BYTECODEENCODER_H_

#include "iree/compiler/Dialect/VM/IR/VMFuncEncoder.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

struct EncodedBytecodeFunction {
  std::vector<uint8_t> bytecodeData;
  int8_t i32RegisterCount = 0;
  int8_t refRegisterCount = 0;
};

// Abstract encoder used for function bytecode encoding.
class BytecodeEncoder : public VMFuncEncoder {
 public:
  // Encodes a vm.func to bytecode and returns the result.
  // Returns None on failure.
  static Optional<EncodedBytecodeFunction> encodeFunction(
      IREE::VM::FuncOp funcOp, llvm::DenseMap<Type, int> &typeTable,
      SymbolTable &symbolTable);

  BytecodeEncoder() = default;
  ~BytecodeEncoder() = default;
};

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_BYTECODEENCODER_H_
