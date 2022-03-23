// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_DIALECT_LINALGTRANSFORMS_TRANSFORMOPMAPPING_H
#define MLIR_DIALECT_LINALGTRANSFORMS_TRANSFORMOPMAPPING_H

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
class Value;

using TransformOpMapping = DenseMap<Value, SmallVector<Operation *>>;
} // namespace mlir

#endif // MLIR_DIALECT_LINALGTRANSFORMS_TRANSFORMOPMAPPING_H
