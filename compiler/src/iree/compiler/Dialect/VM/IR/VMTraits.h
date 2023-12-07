// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMTRAITS_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::IREE::VM {

template <typename ConcreteType>
class DebugOnly : public OpTrait::TraitBase<ConcreteType, DebugOnly> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify debug-only.
    return success();
  }
};

template <typename ConcreteType>
class FullBarrier : public OpTrait::TraitBase<ConcreteType, FullBarrier> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify full barrier.
    return success();
  }
};

template <typename ConcreteType>
class PseudoOp : public OpTrait::TraitBase<ConcreteType, PseudoOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify pseudo op (not serializable?).
    return success();
  }
};

template <typename ConcreteType>
class AssignmentOp : public OpTrait::TraitBase<ConcreteType, AssignmentOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    if (op->getNumOperands() != op->getNumResults()) {
      return op->emitOpError()
             << "must have a matching number of operands and results";
    }
    return success();
  }
};

template <typename ConcreteType>
class ExtF32 : public OpTrait::TraitBase<ConcreteType, ExtF32> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify f32 ext is supported.
    return success();
  }
};

template <typename ConcreteType>
class ExtF64 : public OpTrait::TraitBase<ConcreteType, ExtF64> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify f64 ext is supported.
    return success();
  }
};

} // namespace mlir::OpTrait::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_IR_VMTRAITS_H_
