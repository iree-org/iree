// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

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
