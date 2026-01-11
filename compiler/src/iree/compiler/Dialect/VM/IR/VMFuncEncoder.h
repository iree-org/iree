// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMFUNCENCODER_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMFUNCENCODER_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::iree_compiler {

// Optional register allocation analysis interface used during encoding.
// Encoders that can provide this (such as the bytecode encoder) may use it to
// enable encoding-time optimizations (MOVE bit propagation, discard elision,
// etc). Other encoders (such as EmitC) return nullptr and ops must
// conservatively encode without analysis.
class VMRegisterAllocation {
public:
  virtual ~VMRegisterAllocation() = default;

  // Returns true if the operand at |operandIndex| of the given discard op can
  // be elided because it was already released via MOVE on a preceding op.
  virtual bool isDiscardOperandElidable(Operation *op,
                                        unsigned operandIndex) const = 0;

  // Maps an SSA |value| to its allocated register ordinal.
  // Returns a packed ordinal (matching the VM register encoding) such that
  // ref registers have the type bit set.
  virtual int mapValueToRegisterOrdinal(Value value) const = 0;
};

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

  // Encodes an integer or floating-point primitive attribute as a fixed byte
  // length based on bitwidth.
  virtual LogicalResult encodePrimitiveAttr(TypedAttr value) = 0;

  // Encodes a variable-length integer or floating-point array attribute.
  virtual LogicalResult encodePrimitiveArrayAttr(DenseElementsAttr value) = 0;

  // Encodes a string attribute as a B-string.
  virtual LogicalResult encodeStrAttr(StringAttr value) = 0;

  // Encodes a branch target and the operand mappings.
  virtual LogicalResult encodeBranch(Block *targetBlock,
                                     Operation::operand_range operands,
                                     int successorIndex) = 0;

  // Encodes just a branch target (PC offset) without operand mappings.
  virtual LogicalResult encodeBranchTarget(Block *targetBlock) = 0;

  // Encodes a branch table.
  virtual LogicalResult encodeBranchTable(SuccessorRange caseSuccessors,
                                          OperandRangeRange caseOperands,
                                          int baseSuccessorIndex) = 0;

  // Encodes an operand value (by reference).
  virtual LogicalResult encodeOperand(Value value, int ordinal) = 0;

  // Encodes a variable list of operands (by reference), including a count.
  virtual LogicalResult encodeOperands(Operation::operand_range values) = 0;

  // Encodes a filtered list of operands (by reference), including a count.
  // Each pair contains the value and its original operand index for MOVE bit
  // computation.
  virtual LogicalResult
  encodeOperands(ArrayRef<std::pair<Value, int>> valuesWithIndices) = 0;

  // Encodes a result value (by reference).
  virtual LogicalResult encodeResult(Value value) = 0;

  // Encodes a variable list of results (by reference), including a count.
  virtual LogicalResult encodeResults(Operation::result_range values) = 0;

  // Encodes result destination registers from successor block arguments.
  // Used for operations like CallYieldable where call results go to block args.
  virtual LogicalResult encodeBlockArgResults(Block *targetBlock) = 0;

  // Returns the register allocation analysis, if available.
  // Bytecode encoder provides this; other encoders (e.g., EmitC) return
  // nullptr. Ops can use this for register-allocation-aware encoding decisions
  // such as per-operand elision.
  virtual const VMRegisterAllocation *getRegisterAllocation() const {
    return nullptr;
  }
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VM_IR_VMFUNCENCODER_H_
