// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_IR_IREETRAITS_H_
#define IREE_COMPILER_DIALECT_IREE_IR_IREETRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace IREE {

template <typename ConcreteType>
class YieldPoint : public OpTrait::TraitBase<ConcreteType, YieldPoint> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify yield point.
    return success();
  }
};

template <typename ConcreteType>
class Unsafe : public OpTrait::TraitBase<ConcreteType, Unsafe> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify that entire tree is marked unsafe.
    return success();
  }
};

}  // namespace IREE
}  // namespace OpTrait
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_IR_IREETRAITS_H_
