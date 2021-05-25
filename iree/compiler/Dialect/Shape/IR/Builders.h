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

// Given an arbitrary list of inputs, builds IR to obtain their shapes and
// cast them to a given !shapex.ranked_shape. Statically verifiable invariants
// will be checked within this call and runtime code will be emitted to verify
// the rest.
// Returns nullptr and emits an error on violation of an invariant.
Value buildCastInputsToResultShape(Location loc,
                                   RankedShapeType resultShapeType,
                                   ArrayRef<Value> inputs, OpBuilder &builder);

// Given a src ranked_shape value and a destination rank, broadcasts the
// src shape to the given rank by adding degenerate '1' dimensions for any
// dimensions not mapped by broadcastDims (which has the same semantics as
// the XLA broadcast_in_dim op).
// Note specifically that this does not do a full broadcast to a final
// shape since '1' dims are inserted for expanded dimensions. Such an
// intermediate can typically be passed to some form of 'tile' op to arrive
// at a fully broadcasted value.
// If broadcastDims is empty, then degenerate dimensions are added on the left
// up to the dstRank.
// The srcShape must have a type of ranked_shape and nullptr will be returned
// if violated.
Value buildDegenerateBroadcastRankedShape(
    Value srcShape, int dstRank, SmallVectorImpl<int64_t> &broadcastDims,
    OpBuilder &builder);

// Given a value representing a ShapedType (i.e. tensor or otherwise), attempts
// to locate a computed RankedShape for it by examining uses for a corresponding
// tie_shape op, returning the associated RankedShape.
// In the case of a static shape, a const_ranked_shape will be created and
// returned. If dimType is provided, then any returned shape will have the
// given dimType (defaults to IndexType), returning nullptr if this is not
// possible.
Value buildOrFindRankedShapeForValue(Location loc, Value value, Type dimType,
                                     OpBuilder &builder);

// Returns dimension values for each dynamic dimension of the given |value|.
// |value| must be a ShapedType and may optionally have a ranked_shape tied.
// The returned value range will be empty if the shape is fully static.
SmallVector<Value, 4> buildOrFindDynamicDimsForValue(Location loc, Value value,
                                                     OpBuilder &builder);

// Given a RankedShapeType'd value |rsValue|, populate values for all
// dimensions. If |createIntermediateOps|, then if the dims cannot be resolved
// by walking the IR, then a RankedDimsOp is created. If false, then the dims
// will either be resolved to discrete dimension SSA values (that already
// exist) or index consts for static dims.
LogicalResult getRankedDimsFromRankedShape(Location loc, Value rsValue,
                                           bool createIntermediateOps,
                                           SmallVectorImpl<Value> &outDims,
                                           OpBuilder &builder);

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_
