// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTILTRAITS_H_
#define IREE_COMPILER_DIALECT_UTIL_IR_UTILTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace IREE {
namespace Util {

template <typename ConcreteType>
struct YieldPoint : public OpTrait::TraitBase<ConcreteType, YieldPoint> {
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify yield point.
    return success();
  }
};

template <typename ConcreteType>
struct Unsafe : public OpTrait::TraitBase<ConcreteType, Unsafe> {
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify that entire tree is marked unsafe.
    return success();
  }
};

template <typename ConcreteType>
struct DebugOnly : public OpTrait::TraitBase<ConcreteType, DebugOnly> {
  // TODO(benvanik): helper for eliding safely on ops that return values.

  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

template <typename ConcreteType>
struct ImplicitlyCaptured
    : public OpTrait::TraitBase<ConcreteType, ImplicitlyCaptured> {
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

} // namespace Util
} // namespace IREE
} // namespace OpTrait
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_UTIL_IR_UTILTRAITS_H_
