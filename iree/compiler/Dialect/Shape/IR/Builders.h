// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Builds a ranked_shape for the given |shapedValue| with zero or more dynamic
// dims with the values taken from |dynamicDims|.
// Value buildRankedShapeForValue(Location loc, Value shapedValue,
//  ValueRange dynamicDims, OpBuilder &builder);

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_
