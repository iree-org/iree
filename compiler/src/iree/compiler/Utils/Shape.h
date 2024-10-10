// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_SHAPE_H_
#define IREE_COMPILER_UTILS_SHAPE_H_

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

namespace mlir::iree_compiler {

// Unranked shapes are always considered to have more dynamic dimensions than
// ranked.
inline bool shapeHasLessOrEqualDynamicDimensions(ShapedType t1, ShapedType t2) {
  if (!t2.hasRank()) {
    return true;
  }
  if (!t1.hasRank()) {
    return false;
  }

  return llvm::count_if(t1.getShape(), &ShapedType::isDynamic) <=
         llvm::count_if(t2.getShape(), &ShapedType::isDynamic);
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_SHAPE_H_
