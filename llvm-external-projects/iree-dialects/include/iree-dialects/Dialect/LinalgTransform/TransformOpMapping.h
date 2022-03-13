//===-- TransformOpMapping.h - Map trasform dialect values to ops - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALGTRANSFORMS_TRANSFORMOPMAPPING_H
#define MLIR_DIALECT_LINALGTRANSFORMS_TRANSFORMOPMAPPING_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;
class Value;

using TransformOpMapping = DenseMap<Value, SmallVector<Operation *>>;
}  // namespace mlir

#endif  // MLIR_DIALECT_LINALGTRANSFORMS_TRANSFORMOPMAPPING_H
