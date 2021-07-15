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
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Builds a ranked_shape for the given |shapedValue| with zero or more dynamic
// dims with the values taken from |dynamicDims|.
Value buildRankedShapeForValue(Location loc, Value shapedValue,
                               ValueRange dynamicDims, OpBuilder &builder);

// As with buildRankedShapeForValue but by selecting out the appropriate dims
// from a flattened set of values and dynamic dims.
Value buildRankedShapeForValueInList(Location loc, unsigned index,
                                     ValueRange flatValues,
                                     ValueRange flatDynamicDims,
                                     OpBuilder &builder);

// Returns dimension values for each dimension of the given |value|.
// |value| must be a ShapedType and may optionally have a ranked_shape tied.
Optional<SmallVector<Value, 4>> buildOrFindDimsForValue(Location loc,
                                                        Value value,
                                                        OpBuilder &builder);

// Returns dimension values for each dynamic dimension of the given |value|.
// |value| must be a ShapedType and may optionally have a ranked_shape tied.
// The returned value range will be empty if the shape is fully static.
SmallVector<Value, 4> buildOrFindDynamicDimsForValue(Location loc, Value value,
                                                     OpBuilder &builder);

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_
