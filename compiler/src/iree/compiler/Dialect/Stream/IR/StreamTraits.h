// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_IR_STREAMTRAITS_H_
#define IREE_COMPILER_DIALECT_STREAM_IR_STREAMTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir ::OpTrait::IREE::Stream {

template <typename ConcreteType>
class TensorPhaseOp : public OpTrait::TraitBase<ConcreteType, TensorPhaseOp> {
public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

template <typename ConcreteType>
class AsyncPhaseOp : public OpTrait::TraitBase<ConcreteType, AsyncPhaseOp> {
public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

template <typename ConcreteType>
class CmdPhaseOp : public OpTrait::TraitBase<ConcreteType, CmdPhaseOp> {
public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

// Marks ops where tied operand->result pairs are identity passthroughs: the op
// manages scheduling or synchronization but does not access or modify the
// resource data. This distinguishes passthrough ties (timepoint.barrier,
// timepoint.await) from mutating ties (dispatch, fill, copy).
template <typename ConcreteType>
class TiedResourcePassthrough
    : public OpTrait::TraitBase<ConcreteType, TiedResourcePassthrough> {};

} // namespace mlir::OpTrait::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_IR_STREAMTRAITS_H_
