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

#ifndef IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_TYPECONVERSION_H_
#define IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_TYPECONVERSION_H_

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Utility class for expanding types that are common in shape lowerings.
// All such expansions follow the pattern where there is one source type
// which is expanded to 0 or more target types at a signature boundary (i.e.
// function, call, block, etc).
//
// Subclasses implement the rules for a single type transition, and the
// parent class has helpers for applying it at various levels of granularity.
//
// It is expected that individual passes will use one or more TypeExpanders.
// Since it can be used standalone in this way, and only performs local
// transformations, it may be usable in situations that make the conversion
// framework less suitable.
class TypeExpander {
 public:
  virtual ~TypeExpander() = default;

  // Converts a single source type to >= 1 target types.
  virtual LogicalResult convertType(
      Type sourceType, SmallVectorImpl<Type> &targetTypes) const = 0;

  // Inserts casts to convert a list of targetValues corresponding to a
  // previous call to convertType to the given sourceType.
  virtual Value castToSource(Location loc, Type sourceType,
                             ArrayRef<Value> targetValues,
                             OpBuilder &builder) const = 0;

  // Inserts casts to convert a sourceValue into a list of target values where
  // it is expected that there will be one targetValue produced for each
  // type populated in a corresponding call to convertType.
  virtual LogicalResult castToTarget(Location loc, Value sourceValue,
                                     ArrayRef<Type> targetTypes,
                                     SmallVectorImpl<Value> &targetValues,
                                     OpBuilder &builder) const = 0;

  // Performs type expansion on a function signature, expanding types in
  // both its operands and results. In order to be generally valid, additional
  // expansions will need to be done:
  //   - expandAllReturnLikeTerminators
  //
  // In the case of type expansions, the arg or result attribute will be
  // applied to the first type and defaults will be used for subsequent.
  LogicalResult expandFunctionSignature(FuncOp funcOp,
                                        OpBuilder &builder) const;

  // Expands the signature of a block, inserting source->target casts as
  // needed using the given builder (which will have its insertion point
  // reset). This does nothing to terminators or branches into the block.
  LogicalResult expandBlockSignature(Location loc, Block *block,
                                     OpBuilder &builder) const;

  // Expands a list of source values to target values, inserting necessary
  // casts.
  LogicalResult expandSourceValuesToTarget(Location loc,
                                           ArrayRef<Value> sourceValues,
                                           SmallVectorImpl<Value> &targetValues,
                                           OpBuilder &builder) const;

  // Expands a return-like terminator by inserting casts to targets.
  // This is usually paired with a expandFunctionSignature in order to
  // make terminators consistent with the hosting function signature.
  template <typename OpTy>
  LogicalResult expandReturnLikeTerminator(OpTy op, OpBuilder &builder) const {
    Operation *genericOp = op.getOperation();
    if (genericOp->getNumResults() != 0) {
      return op.emitOpError() << "unsupported return like op";
    }
    return expandGenericReturnLikeTerminator(genericOp, builder);
  }

  // For a parent, expands all return-like terminators.
  template <typename OpTy>
  LogicalResult expandAllReturnLikeTerminators(Operation *parentOp,
                                               OpBuilder &builder) const {
    LogicalResult result = success();
    parentOp->walk([&](OpTy op) {
      if (failed(expandReturnLikeTerminator(op, builder))) {
        result = failure();
      }
    });
    return result;
  }

 private:
  LogicalResult expandGenericReturnLikeTerminator(Operation *op,
                                                  OpBuilder &builder) const;
};

// Gets a TypeExpander which will expand any dynamic tensor to also include
// its dynamic components explicitly.
// This will insert tie_shape ops for target->source casts and get_*_shape ops
// for source->target.
//
// Dynamic tensors are handled as:
//   - T = RankedTensorType: The tensor is expanded to (T, RankedShapeType) with
//     one index type for each dynamic dimension.
//   - T = UnrankedTensorType: Not currently supported. Will eventually expand
//         to (T, UnrankedShapeType).
//
// This expansion is typically done early in compilation to establish signatures
// that operate on types in the shape hierarchy.
const TypeExpander &getDynamicShapeTypeExpander();

// Gets a TypeExpander which will expand any shape types to corresponding
// primitive primitive types:
//
//   - T = RankedShapeType: Expanded to individual dynamic dimensions of the
//     dimension type T::getDimType().
const TypeExpander &getShapeToPrimitiveTypeExpander();

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_TYPECONVERSION_H_
