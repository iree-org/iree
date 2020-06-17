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

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMFUNCENCODER_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMFUNCENCODER_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {

// Interface for encoding of VM operations within functions.
// This base manages source map construction and vm.func walking while
// subclasses provide actual emission.
class VMFuncEncoder {
 public:
  virtual ~VMFuncEncoder() = default;

  // Begins encoding the contents of a block.
  virtual LogicalResult beginBlock(Block *block) = 0;

  // Ends encoding the contents of a block.
  virtual LogicalResult endBlock(Block *block) = 0;

  // Begins encoding an operation.
  virtual LogicalResult beginOp(Operation *op) = 0;

  // Ends encoding an operation.
  virtual LogicalResult endOp(Operation *op) = 0;

  // Encodes an 8-bit integer. Fails if |value| cannot be represented as a byte.
  virtual LogicalResult encodeI8(int value) = 0;

  // Encodes an opcode of the given value.
  virtual LogicalResult encodeOpcode(StringRef name, int opcode) = 0;

  // Encodes a global or function symbol ordinal.
  virtual LogicalResult encodeSymbolOrdinal(SymbolTable &syms,
                                            StringRef name) = 0;

  // Encodes a value type as an integer kind.
  virtual LogicalResult encodeType(Value value) = 0;
  virtual LogicalResult encodeType(Type type) = 0;

  // Encodes an integer attribute as a fixed byte length based on bitwidth.
  virtual LogicalResult encodeIntAttr(IntegerAttr value) = 0;

  // Encodes a variable-length integer array attribute.
  virtual LogicalResult encodeIntArrayAttr(DenseIntElementsAttr value) = 0;

  // Encodes a string attribute as a B-string.
  virtual LogicalResult encodeStrAttr(StringAttr value) = 0;

  // Encodes a branch target and the operand mappings.
  virtual LogicalResult encodeBranch(Block *targetBlock,
                                     Operation::operand_range operands,
                                     int successorIndex) = 0;

  // Encodes an operand value (by reference).
  virtual LogicalResult encodeOperand(Value value, int ordinal) = 0;

  // Encodes a variable list of operands (by reference), including a count.
  virtual LogicalResult encodeOperands(Operation::operand_range values) = 0;

  // Encodes a result value (by reference).
  virtual LogicalResult encodeResult(Value value) = 0;

  // Encodes a variable list of results (by reference), including a count.
  virtual LogicalResult encodeResults(Operation::result_range values) = 0;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_IR_VMFUNCENCODER_H_
