// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_IR_HALTRAITS_H_
#define IREE_COMPILER_DIALECT_HAL_IR_HALTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir ::OpTrait::IREE::HAL {

template <typename ConcreteType>
class ExecutableInterfaceOp
    : public OpTrait::TraitBase<ConcreteType, ExecutableInterfaceOp> {
public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

} // namespace mlir::OpTrait::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_IR_HALTRAITS_H_
